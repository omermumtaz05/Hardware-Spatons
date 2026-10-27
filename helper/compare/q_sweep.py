#!/usr/bin/env python3
"""
q_sweep.py — Find the best 16-bit fixed-point split (Qi.f) for the A12 MLP.
Sweeps fractional bits, reports weight/activation ranges + accuracy vs FP32.
16-bit signed total; integer bits = 15 - frac (plus sign). value = raw / 2^frac.
"""

import argparse, struct
import numpy as np
import torch, torch.nn as nn

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

class Q:
    """A 16-bit fixed-point format with `frac` fractional bits."""
    def __init__(self, frac, total=16):
        self.frac = frac
        self.one  = 1 << frac
        self.qmin = -(1 << (total-1))
        self.qmax =  (1 << (total-1)) - 1
        self.int_bits = total - 1 - frac        # excl sign
        self.range_max = self.qmax / self.one   # largest representable +value
    def to(self, x):
        raw = np.round(np.asarray(x, np.float64) * self.one)
        return np.clip(raw, self.qmin, self.qmax).astype(np.int64)
    def frm(self, raw):
        return raw.astype(np.float64) / self.one
    def clip_raw(self, raw):
        return np.clip(raw, self.qmin, self.qmax).astype(np.int64)

def q_matvec(q, W_raw, x_raw, b_raw):
    """out[o] = sum_i W[o,i]*x[i] + b[o] in Qi.f, wide accumulator + single requantize."""
    O, I = W_raw.shape
    out = np.zeros(O, dtype=np.int64)
    for o in range(O):
        acc = np.int64(0)                                  # wide: 2*frac fractional bits
        for i in range(I):
            acc += W_raw[o, i] * x_raw[i]                  # Qi.f * Qi.f
        acc = (acc + (1 << (q.frac - 1))) >> q.frac        # -> Qi.f, round-to-nearest
        acc += b_raw[o]
        out[o] = np.clip(acc, q.qmin, q.qmax)
    return out.astype(np.int64)

def relu_raw(x): return np.maximum(x, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='#86-NN2_best.pth')
    ap.add_argument('--capture', default='data_files/capture_array.hex')
    ap.add_argument('--crop', type=int, default=32)
    ap.add_argument('--fracs', type=int, nargs='+', default=[8,10,11,12,13])
    args = ap.parse_args()

    IN_DIM, HIDDEN, OUT_DIM = 29, 48, 3
    FULL = 256

    m = ParametricModel(IN_DIM, OUT_DIM, HIDDEN)
    sd = torch.load(args.weights, map_location='cpu')['model_state_dict']
    for n in ['layer1','layer2','layer3','residual']:
        getattr(m,n).weight.data = sd[f'models.1.model.{n}.weight']
        getattr(m,n).bias.data   = sd[f'models.1.model.{n}.bias']
    m.eval()

    # weight/bias magnitude ceiling (determines min integer bits needed)
    wmax = 0.0
    print("=== weight/bias ranges ===")
    for n in ['layer1','layer2','layer3','residual']:
        w=getattr(m,n).weight.data.numpy(); b=getattr(m,n).bias.data.numpy()
        wmax = max(wmax, np.abs(w).max(), np.abs(b).max())
        print(f"  {n:9s} |W|max={np.abs(w).max():.4f}  |b|max={np.abs(b).max():.4f}")
    print(f"  overall |w|max = {wmax:.4f}\n")

    # load capture
    img=np.zeros((FULL,FULL,3),np.float32)
    with open(args.capture) as f:
        lines=[l.strip() for l in f if l.strip() and not l.strip().startswith('//')]
    for idx,line in enumerate(lines):
        line=line.zfill(24)
        B=hex_f32(line[0:8]);G=hex_f32(line[8:16]);R=hex_f32(line[16:24])
        img[idx//FULL,idx%FULL]=[R,G,B]
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

    # FP32 reference outputs + track activation ranges (format-independent)
    feats=[]; z_ref=[]
    h1_max=h2_max=0.0
    for y in range(N):
        for x in range(N):
            ft=feat(x,y); feats.append(ft)
            with torch.no_grad():
                # tap activations for range check
                t1=torch.relu(m.layer1(torch.tensor(ft).view(1,IN_DIM)))
                t2=torch.relu(m.layer2(t1))
                zz=(m.layer3(t2)+m.residual(torch.tensor(ft).view(1,IN_DIM))).view(3).numpy()
            h1_max=max(h1_max, t1.abs().max().item())
            h2_max=max(h2_max, t2.abs().max().item())
            z_ref.append(zz)
    z_ref=np.array(z_ref)
    print(f"=== activation ranges (over {N}x{N}) ===")
    print(f"  |h1|max={h1_max:.4f}  |h2|max={h2_max:.4f}  |feat|max=1.0\n")

    print(f"=== fixed-point sweep (16-bit total) ===")
    print(f"{'frac':>4} {'int_bits':>8} {'range':>9} {'fits?':>6} {'max_err':>9} {'RMSE':>9}")
    for frac in args.fracs:
        q = Q(frac)
        # does the format's range cover weights AND activations?
        need = max(wmax, h1_max, h2_max, 1.0)
        fits = q.range_max >= need
        # quantize weights
        qw={n:q.to(getattr(m,n).weight.data.numpy()) for n in ['layer1','layer2','layer3','residual']}
        qb={n:q.to(getattr(m,n).bias.data.numpy())   for n in ['layer1','layer2','layer3','residual']}
        max_err=0.0; ss=0.0; cnt=0
        for k,ft in enumerate(feats):
            fr=q.to(ft)
            h1=relu_raw(q_matvec(q,qw['layer1'],fr,qb['layer1']))
            h2=relu_raw(q_matvec(q,qw['layer2'],h1,qb['layer2']))
            y_=q_matvec(q,qw['layer3'],h2,qb['layer3'])
            yr=q_matvec(q,qw['residual'],fr,qb['residual'])
            z=q.clip_raw(y_+yr)
            zq=q.frm(z)
            e=np.abs(z_ref[k]-zq)
            max_err=max(max_err,e.max()); ss+=np.sum(e**2); cnt+=3
        rmse=np.sqrt(ss/cnt)
        print(f"{frac:>4} {q.int_bits:>8} {q.range_max:>9.3f} {str(fits):>6} {max_err:>9.5f} {rmse:>9.5f}")

    print()
    print("Pick the LOWEST-error frac where fits=True. If the best fits=True option")
    print("still has large error, plain fixed-point won't cut it -> int-x-scale.")

if __name__ == '__main__':
    main()