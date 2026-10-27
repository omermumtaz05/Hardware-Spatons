#!/usr/bin/env python3
"""
act_scale_survey.py — pick FIXED per-boundary activation scales for the A12 MLP,
and verify the full 10-bit-weight / 12-bit-activation integer pipeline against FP32.

Context:
  - Weight scales are already exported (data_files/a12_scales.txt), one per layer.
    Javad's scheme: symmetric per-tensor, scale = abs_max / qmax  (matches quant_core.uniform_quantize).
  - Activation scales are NOT exported because his forward computes them dynamically
    per batch. This script surveys real activation ranges (f, h1, h2) across the
    capture, proposes ONE fixed scale per boundary (with headroom), then runs the
    whole int pipeline (int MAC -> dequantize -> bias+ReLU in float -> requantize)
    and reports final-z error vs FP32.

Run:
  python act_scale_survey.py --crop 0        # full 256x256
  python act_scale_survey.py --crop 64       # quick 64x64
  python act_scale_survey.py --headroom 1.3  # more clipping margin
"""

import argparse, struct
import numpy as np
import torch, torch.nn as nn

# ---- bit widths (match your RTL) ----
W_BITS   = 10
A_BITS   = 12
W_QMAX   = (1 << (W_BITS - 1)) - 1     # 511  (symmetric signed)
W_QMIN   = -(1 << (W_BITS - 1))        # -512
A_QMAX   = (1 << (A_BITS - 1)) - 1     # 2047
A_QMIN   = -(1 << (A_BITS - 1))        # -2048

# Weight scales exported by Javad (data_files/a12_scales.txt). Baked in so this
# script uses the SAME weight quantization the hardware will use.
WEIGHT_SCALE = {
    'layer1':   1.577570830306e-03,
    'layer2':   1.469165145068e-02,
    'layer3':   3.551398704663e-03,
    'residual': 3.478375433009e-04,
}

def hex_f32(h): return struct.unpack('>f', bytes.fromhex(h))[0]

class ParametricModel(nn.Module):
    def __init__(self, i, o, h):
        super().__init__()
        self.layer1=nn.Linear(i,h); self.layer2=nn.Linear(h,h)
        self.layer3=nn.Linear(h,o); self.residual=nn.Linear(i,o)
        self.act=nn.ReLU()
    def forward(self,x):
        t=self.act(self.layer1(x)); t=self.act(self.layer2(t))
        return self.layer3(t)+self.residual(x)

# ---- quantize helpers (symmetric, round-to-nearest, saturating) ----
def quant_w(w, scale):
    raw = np.round(w.astype(np.float64) / scale)
    return np.clip(raw, W_QMIN, W_QMAX).astype(np.int64)

def quant_a(a, scale):
    raw = np.round(a.astype(np.float64) / scale)
    return np.clip(raw, A_QMIN, A_QMAX).astype(np.int64)

def int_matvec_dequant(w_int, x_int, w_scale, a_scale, bias_fp):
    """
    One layer, hardware-faithful:
      acc = sum_i w_int[o,i] * x_int[i]           (wide int accumulator)
      real = acc * (w_scale * a_scale)             (dequantize to float)
      out  = real + bias_fp                        (bias-add in float)
    Returns float pre-activation (caller applies ReLU / requantize).
    """
    acc = w_int.astype(np.int64) @ x_int.astype(np.int64)   # (O,)
    real = acc.astype(np.float64) * (w_scale * a_scale)
    return real + bias_fp.astype(np.float64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='#86-NN2_best.pth')
    ap.add_argument('--capture', default='data_files/capture_array.hex')
    ap.add_argument('--crop', type=int, default=0, help='NxN crop (0=full 256)')
    ap.add_argument('--headroom', type=float, default=1.2, help='scale headroom over p99.9')
    ap.add_argument('--pctile', type=float, default=99.9, help='percentile for scale (vs abs max)')
    ap.add_argument('--photo-golden', default=None,
                    help='path to photo_out_golden.hex; if set, applies post_mlp and diffs against it')
    args = ap.parse_args()

    IN_DIM, HIDDEN, OUT_DIM = 29, 48, 3
    FULL = 256

    m = ParametricModel(IN_DIM, OUT_DIM, HIDDEN)
    sd = torch.load(args.weights, map_location='cpu')['model_state_dict']
    for n in ['layer1','layer2','layer3','residual']:
        getattr(m,n).weight.data = sd[f'models.1.model.{n}.weight']
        getattr(m,n).bias.data   = sd[f'models.1.model.{n}.bias']
    m.eval()

    # pre-quantize all weights with the exported scales
    W = {n: getattr(m,n).weight.data.numpy() for n in ['layer1','layer2','layer3','residual']}
    B = {n: getattr(m,n).bias.data.numpy()   for n in ['layer1','layer2','layer3','residual']}
    Wq = {n: quant_w(W[n], WEIGHT_SCALE[n]) for n in W}

    # load capture
    img=np.zeros((FULL,FULL,3),np.float32)
    with open(args.capture) as f:
        lines=[l.strip() for l in f if l.strip() and not l.strip().startswith('//')]
    for idx,line in enumerate(lines):
        line=line.zfill(24)
        B_=hex_f32(line[0:8]);G_=hex_f32(line[8:16]);R_=hex_f32(line[16:24])
        img[idx//FULL,idx%FULL]=[R_,G_,B_]
    img=np.clip(img,0,1)

    xs=np.linspace(0,1,FULL,np.float32); ys=np.linspace(0,1,FULL,np.float32)
    def refl(c,N): return -c if c<0 else (2*(N-1)-c if c>=N else c)
    def px(x,y): return img[refl(y,FULL),refl(x,FULL)]
    def feat(x,y):
        f=[xs[x],ys[y]]; f.extend(img[y,x])
        for dy in(-1,0,1):
            for dx in(-1,0,1):
                if dy==0 and dx==0: continue
                f.extend(px(x+dx,y+dy))
        return np.array(f,np.float32)

    N = args.crop if args.crop>0 else FULL

    # ---- pass 1: collect FP32 activation ranges + reference z ----
    feats=[]; z_ref=[]; centers=[]
    f_abs=[]; h1_abs=[]; h2_abs=[]
    for y in range(N):
        for x in range(N):
            ft=feat(x,y); feats.append(ft)
            centers.append(img[y,x].copy())   # center RGB, needed for post_mlp
            with torch.no_grad():
                xt=torch.tensor(ft).view(1,IN_DIM)
                t1=torch.relu(m.layer1(xt))
                t2=torch.relu(m.layer2(t1))
                zz=(m.layer3(t2)+m.residual(xt)).view(3).numpy()
            f_abs.append(np.abs(ft).max())
            h1_abs.append(t1.abs().max().item())
            h2_abs.append(t2.abs().max().item())
            z_ref.append(zz)
    z_ref=np.array(z_ref); feats=np.array(feats); centers=np.array(centers)
    f_abs=np.array(f_abs); h1_abs=np.array(h1_abs); h2_abs=np.array(h2_abs)

    def pick_scale(absvals):
        p = np.percentile(absvals, args.pctile)
        return (args.headroom * p) / A_QMAX, p, absvals.max()

    f_scale, f_p, f_m   = pick_scale(f_abs)
    h1_scale, h1_p, h1_m= pick_scale(h1_abs)
    h2_scale, h2_p, h2_m= pick_scale(h2_abs)

    print(f"=== activation ranges over {N}x{N} ({N*N} pixels) ===")
    print(f"  f : p{args.pctile}={f_p:.4f}  absmax={f_m:.4f}")
    print(f"  h1: p{args.pctile}={h1_p:.4f}  absmax={h1_m:.4f}")
    print(f"  h2: p{args.pctile}={h2_p:.4f}  absmax={h2_m:.4f}")
    print(f"\n=== proposed fixed activation scales (headroom x{args.headroom}) ===")
    print(f"  f_scale  = {f_scale:.12e}   hex {struct.pack('>f',f_scale).hex()}")
    print(f"  h1_scale = {h1_scale:.12e}   hex {struct.pack('>f',h1_scale).hex()}")
    print(f"  h2_scale = {h2_scale:.12e}   hex {struct.pack('>f',h2_scale).hex()}")

    # clip-rate diagnostics: how many pixels exceed each fixed scale's range
    def cliprate(absvals, scale):
        return 100.0 * np.mean(absvals > scale * A_QMAX)
    print(f"\n=== clip rate (pixels whose max activation saturates) ===")
    print(f"  f : {cliprate(f_abs,f_scale):.3f}%   h1: {cliprate(h1_abs,h1_scale):.3f}%   h2: {cliprate(h2_abs,h2_scale):.3f}%")

    # ---- pass 2: full integer pipeline with fixed scales, error vs FP32 ----
    max_err=0.0; ss=0.0; cnt=0; worst=None
    z_quant=np.zeros_like(z_ref)
    for k,ft in enumerate(feats):
        f_int  = quant_a(ft, f_scale)
        h1_pre = int_matvec_dequant(Wq['layer1'], f_int, WEIGHT_SCALE['layer1'], f_scale, B['layer1'])
        h1     = np.maximum(h1_pre, 0.0)
        h1_int = quant_a(h1, h1_scale)
        h2_pre = int_matvec_dequant(Wq['layer2'], h1_int, WEIGHT_SCALE['layer2'], h1_scale, B['layer2'])
        h2     = np.maximum(h2_pre, 0.0)
        h2_int = quant_a(h2, h2_scale)
        y      = int_matvec_dequant(Wq['layer3'], h2_int, WEIGHT_SCALE['layer3'], h2_scale, B['layer3'])
        yr     = int_matvec_dequant(Wq['residual'], f_int, WEIGHT_SCALE['residual'], f_scale, B['residual'])
        z      = y + yr
        z_quant[k]=z
        e=np.abs(z_ref[k]-z)
        if e.max()>max_err: max_err=e.max(); worst=(k,z_ref[k].copy(),z.copy())
        ss+=np.sum(e**2); cnt+=3
    rmse=np.sqrt(ss/cnt)

    print(f"\n=== fixed-scale int pipeline vs FP32 (final z, pre-post_mlp) ===")
    print(f"  max abs err : {max_err:.6f}")
    print(f"  RMSE        : {rmse:.6f}")
    if worst:
        k,zf,zq=worst
        print(f"  worst idx {k}: fp={zf}  int={zq}")
    print()
    print("z feeds sigmoid(4z-2)+center, so sigmoid compresses error. ~0.01-0.05 max")
    print("err is likely invisible; >0.1 may show. Compare against your Q8.8 result and")
    print("your photo_out_golden tolerance. If clip rates above are >~0.5%, raise")
    print("--headroom (wider scale, less precision) or that boundary needs attention.")

    # ---- optional pass 3: apply post_mlp and diff against photo_out_golden.hex ----
    if args.photo_golden:
        def post_mlp(z, center):
            # sigmoid computed directly (LUT is a hardware-only concern)
            delta = 1.0 / (1.0 + np.exp(-z))
            delta = 4.0 * delta - 2.0
            final = center + delta
            return np.clip(final, 0.0, 1.0)

        # post_mlp on BOTH the FP32-ref z and the quantized z
        final_fp  = np.array([post_mlp(z_ref[k],   centers[k]) for k in range(len(feats))])
        final_q   = np.array([post_mlp(z_quant[k], centers[k]) for k in range(len(feats))])

        # load photo_out_golden.hex (96-bit packed BGR fp32, same layout as capture)
        gold=np.zeros((len(feats),3),np.float64)
        with open(args.photo_golden) as f:
            glines=[l.strip() for l in f if l.strip() and not l.strip().startswith('//')]
        Ntot = N*N
        for idx in range(min(Ntot, len(glines))):
            line=glines[idx].zfill(24)
            Bg=hex_f32(line[0:8]);Gg=hex_f32(line[8:16]);Rg=hex_f32(line[16:24])
            gold[idx]=[Rg,Gg,Bg]

        def rgb_err(name, approx):
            d=np.abs(approx-gold)
            print(f"  {name:24s} max={d.max():.6f}  rmse={np.sqrt((d**2).mean()):.6f}")

        print(f"\n=== post_mlp output vs photo_out_golden.hex ({N}x{N}) ===")
        rgb_err("FP32 z  -> post_mlp",  final_fp)   # sanity: should be ~0 (matches golden gen)
        rgb_err("quant z -> post_mlp",  final_q)    # the number that matters
        print("  (row1 ~0 confirms golden alignment; row2 is your quantized end-to-end error)")
        print("  Final RGB is [0,1]; a max err under ~1/255=0.004 is sub-quantization-step.")

if __name__ == '__main__':
    main()