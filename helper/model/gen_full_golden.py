#!/usr/bin/env python3
"""gen_full_golden.py — independent end-to-end golden.

  image.hex --(python bilinear warp via LUT)--> warped
            --> 29-dim features --> quantized MLP (10b W / 12b act)
            --> sigmoid LUT --> final RGB --> full_golden_q.hex

Reads NO RTL output. capture_array.hex is never touched.

Usage:
  python helper/gen_full_golden.py                       # ideal fp32 LUT
  python helper/gen_full_golden.py --lut-mode q88        # Q8.8 quantized LUT
"""

import argparse, struct
import numpy as np
import torch, torch.nn as nn
from quant_core import (QuantizationConfig, HardwareConfig, uniform_quantize,
                        dequantize, quantized_matmul_custom, requantize_direct)

# ---------------- sigmoid LUT (unchanged) ----------------
DOMAIN, LUT_N = 8.0, 4096
step = (2*DOMAIN)/(LUT_N-1)
INV_STEP = np.float32((LUT_N-1)/(2*DOMAIN))

def lut_sig(z):
    z = np.float32(z)
    if z <= -DOMAIN: return np.float32(0.0)
    if z >=  DOMAIN: return np.float32(1.0)
    idx = np.float32(np.float32(z + np.float32(DOMAIN)) * INV_STEP)
    i   = max(0, min(LUT_N-1, int(np.floor(np.float32(idx + np.float32(0.5))))))
    return np.float32(1.0/(1.0+np.exp(-(-DOMAIN + i*step))))

def f32_hex(v): return struct.pack('>f', float(v)).hex()
def hex_f32(h): return struct.unpack('>f', bytes.fromhex(h.zfill(8)))[0]

WEIGHT_BITS = 10
ACT_BITS    = 12
HW = HardwareConfig(accumulator_bits=64, use_saturation=True,
                    rounding_mode='nearest_even', requantize_outputs=True, output_bits=8)

class ParametricModel(nn.Module):
    def __init__(self, i, o, h):
        super().__init__()
        self.layer1=nn.Linear(i,h); self.layer2=nn.Linear(h,h)
        self.layer3=nn.Linear(h,o); self.residual=nn.Linear(i,o)
        self.act=nn.ReLU()

def quant_linear(x_float, Wq, bias, act_bits, hw):
    acfg = QuantizationConfig(num_bits=act_bits, symmetric=True, per_channel=False)
    x_q, _, _ = uniform_quantize(x_float, acfg, per_channel=False)
    out_q = quantized_matmul_custom(x_q, Wq, hw)
    if hw.requantize_outputs:
        out_q = requantize_direct(out_q, act_bits, hw)
    out_float = dequantize(out_q)
    if bias is not None:
        out_float = out_float + bias
    return out_float

# ---------------- geo stage in python ----------------
def read_lines(path, skip_comments=True):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            if skip_comments and line.startswith("//"): continue
            out.append(line)
    return out

def load_lut(path, W, H, mode):
    """Return (lut_x, lut_y) in pixel units.
    fp32 : 64-bit line, {y'[63:32], x'[31:0]} big-endian fp32
    q88  : 32-bit line, {y'[31:16], x'[15:0]} signed Q8.8
    """
    lines = read_lines(path)
    assert len(lines) == W*H, f"LUT has {len(lines)} lines, expected {W*H}"
    lx = np.zeros((H, W), np.float64); ly = np.zeros((H, W), np.float64)
    for idx, ln in enumerate(lines):
        y, x = idx // W, idx % W
        if mode == 'fp32':
            h = ln.zfill(16)
            ly[y, x] = hex_f32(h[0:8]); lx[y, x] = hex_f32(h[8:16])
        else:
            h = ln.zfill(8)
            yi = int(h[0:4], 16); xi = int(h[4:8], 16)
            ly[y, x] = yi / 256.0; lx[y, x] = xi / 256.0
            
    print(f"  LUT x range [{lx.min():.2f}, {lx.max():.2f}]  "
          f"y range [{ly.min():.2f}, {ly.max():.2f}]")
    return lx, ly

def load_image(path, W, H):
    lines = read_lines(path)
    assert len(lines) == W*H*3, f"image has {len(lines)} lines, expected {W*H*3}"
    return (np.array([int(v, 16) for v in lines], np.uint8)
            .reshape(H, W, 3).astype(np.float64) / 255.0)

def warp(img, lut_x, lut_y, W, H):
    x0 = np.floor(lut_x).astype(int); y0 = np.floor(lut_y).astype(int)
    dx = (lut_x - x0)[..., None];     dy = (lut_y - y0)[..., None]
    x0c = np.clip(x0, 0, W-1); x1c = np.clip(x0+1, 0, W-1)
    y0c = np.clip(y0, 0, H-1); y1c = np.clip(y0+1, 0, H-1)
    out = ((1-dx)*(1-dy)*img[y0c, x0c] + (1-dx)*dy*img[y1c, x0c] +
           dx*(1-dy)*img[y0c, x1c]     + dx*dy*img[y1c, x1c])
    return np.clip(out, 0, 1).astype(np.float32)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights',  default='#86-NN2_best.pth')
    ap.add_argument('--image',    default='data_files/image.hex')
    ap.add_argument('--lut',      default=None, help='defaults per --lut-mode')
    ap.add_argument('--lut-mode', default='fp32', choices=['fp32', 'q88'])
    ap.add_argument('--crop', type=int, default=0, help='0 = full frame')
    ap.add_argument('--out',  default='data_files')
    args = ap.parse_args()

    IN_DIM, HIDDEN, OUT_DIM = 29, 48, 3
    FULL = 256
    lut_path = args.lut or ('data_files/lut_fp32.hex' if args.lut_mode == 'fp32'
                            else 'data_files/lut_quant.hex')

    # ---- geo stage (python) ----
    print(f"loading LUT ({args.lut_mode}) ...")
    lut_x, lut_y = load_lut(lut_path, FULL, FULL, args.lut_mode)
    print("loading image ...")
    src = load_image(args.image, FULL, FULL)
    print("warping ...")
    img = warp(src, lut_x, lut_y, FULL, FULL)     # == what capture_array *should* be

    # ---- weights ----
    m = ParametricModel(IN_DIM, OUT_DIM, HIDDEN)
    sd = torch.load(args.weights, map_location='cpu')['model_state_dict']
    for n in ['layer1','layer2','layer3','residual']:
        getattr(m,n).weight.data = sd[f'models.1.model.{n}.weight']
        getattr(m,n).bias.data   = sd[f'models.1.model.{n}.bias']
    m.eval()

    wcfg = QuantizationConfig(num_bits=WEIGHT_BITS, symmetric=True, per_channel=False)
    Wq = {}
    for n in ['layer1','layer2','layer3','residual']:
        Wq[n], _, _ = uniform_quantize(getattr(m,n).weight.data, wcfg, per_channel=False)

    # ---- features ----
    xs = np.linspace(0,1,FULL,dtype=np.float32); ys = np.linspace(0,1,FULL,dtype=np.float32)
    def refl(c,N): return -c if c<0 else (2*(N-1)-c if c>=N else c)
    def px(x,y): return img[refl(y,FULL), refl(x,FULL)]
    def feat(x,y):
        f=[xs[x],ys[y]]; f.extend(img[y,x])
        for dy in(-1,0,1):
            for dx in(-1,0,1):
                if dy==0 and dx==0: continue
                f.extend(px(x+dx,y+dy))
        return np.array(f,np.float32)

    N = args.crop if args.crop > 0 else FULL
    fw = open(f"{args.out}/full_warped_q.hex",   'w')   # python-warped image, 3 fp32/px
    ff = open(f"{args.out}/full_features_q.hex", 'w')   # 29 fp32/px
    fg = open(f"{args.out}/full_golden_q.hex",   'w')   # 3 fp32/px  <-- the golden
    for y in range(N):
        if y % 32 == 0: print(f"  row {y}/{N}")
        for x in range(N):
            ft = torch.tensor(feat(x,y)).view(1, IN_DIM)
            h1 = torch.relu(quant_linear(ft, Wq['layer1'],   m.layer1.bias.data,   ACT_BITS, HW))
            h2 = torch.relu(quant_linear(h1, Wq['layer2'],   m.layer2.bias.data,   ACT_BITS, HW))
            yv =            quant_linear(h2, Wq['layer3'],   m.layer3.bias.data,   ACT_BITS, HW)
            yr =            quant_linear(ft, Wq['residual'], m.residual.bias.data, ACT_BITS, HW)
            z  = (yv + yr).view(3).detach().numpy()
            d  = np.array([lut_sig(zi) for zi in z], np.float32)
            fin = np.clip(img[y,x] + (4.0*d - 2.0), 0, 1).astype(np.float32)
            for v in img[y,x]:                    fw.write(f32_hex(v)+"\n")
            for v in ft.view(-1).detach().numpy(): ff.write(f32_hex(v)+"\n")
            for v in fin:                         fg.write(f32_hex(v)+"\n")
    fw.close(); ff.close(); fg.close()
    print(f"wrote full_golden_q.hex (3/px), full_features_q.hex (29/px), "
          f"full_warped_q.hex (3/px), {N}x{N}")

if __name__ == '__main__':
    main()