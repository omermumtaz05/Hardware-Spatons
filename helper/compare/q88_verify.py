#!/usr/bin/env python3
"""
q88_verify.py — Quantize the A12 photometric MLP to Q8.8 fixed-point and measure
accuracy loss vs FP32, BEFORE committing to a hardware datapath.

Q8.8: 16-bit signed, 8 integer + 8 fractional. value = raw_int / 256.
  quantize:   raw = round(x * 256), clamped to [-32768, 32767]
  value seen: raw / 256
  multiply:   (a_raw * b_raw) >> 8   (with rounding)

This simulates what the FPGA will actually compute, so if accuracy holds here,
Q8.8 is viable; if not, we reconsider bit-widths before building hardware.
"""

import argparse, struct
import numpy as np
import torch, torch.nn as nn

Q_FRAC = 10
Q_MIN  = -(1 << 15)      # -32768
Q_MAX  =  (1 << 15) - 1  #  32767
Q_ONE  = 1 << Q_FRAC     #  256

def hex_f32(h): return struct.unpack('>f', bytes.fromhex(h))[0]

# ---- Q8.8 primitives (integer domain) ----
def to_q88(x):
    """FP array -> Q8.8 raw ints (with saturation)."""
    raw = np.round(np.asarray(x, dtype=np.float64) * Q_ONE)
    return np.clip(raw, Q_MIN, Q_MAX).astype(np.int64)

def from_q88(raw):
    """Q8.8 raw ints -> FP value."""
    return raw.astype(np.float64) / Q_ONE

def q88_mul(a_raw, b_raw):
    """Multiply two Q8.8 raws -> Q8.8 raw, round-to-nearest, saturate."""
    prod = a_raw.astype(np.int64) * b_raw.astype(np.int64)   # Q16.16
    # round-to-nearest on the >>8
    prod = (prod + (1 << (Q_FRAC - 1))) >> Q_FRAC            # back to Q8.8
    return np.clip(prod, Q_MIN, Q_MAX).astype(np.int64)

def q88_add(a_raw, b_raw):
    """Add two Q8.8 raws -> Q8.8 raw, saturate."""
    s = a_raw.astype(np.int64) + b_raw.astype(np.int64)
    return np.clip(s, Q_MIN, Q_MAX).astype(np.int64)

def q88_matvec(W_raw, x_raw, b_raw):
    """
    Q8.8 matrix-vector: out[o] = sum_i W[o,i]*x[i] + b[o], all in Q8.8.
    Accumulate products in a WIDE accumulator (Q16.16, no intermediate
    saturation) then shift+round once at the end — this mirrors a sane
    hardware MAC (wide accumulator, single requantize) rather than saturating
    every partial product.
    """
    O, I = W_raw.shape
    out = np.zeros(O, dtype=np.int64)
    for o in range(O):
        acc = np.int64(0)                       # Q16.16 wide accumulator
        for i in range(I):
            acc += W_raw[o, i] * x_raw[i]        # Q8.8 * Q8.8 = Q16.16
        acc = (acc + (1 << (Q_FRAC - 1))) >> Q_FRAC   # -> Q8.8, round
        acc += b_raw[o]                          # bias already Q8.8
        out[o] = np.clip(acc, Q_MIN, Q_MAX)
    return out.astype(np.int64)

def relu_q88(x_raw):
    return np.maximum(x_raw, 0)

# ---- FP32 model (reference) ----
class ParametricModel(nn.Module):
    def __init__(self, i, o, h):
        super().__init__()
        self.layer1=nn.Linear(i,h); self.layer2=nn.Linear(h,h)
        self.layer3=nn.Linear(h,o); self.residual=nn.Linear(i,o)
        self.act=nn.ReLU()
    def forward(self,x):
        t=self.act(self.layer1(x)); t=self.act(self.layer2(t))
        return self.layer3(t)+self.residual(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='#86-NN2_best.pth')
    ap.add_argument('--capture', default='data_files/capture_array.hex')
    ap.add_argument('--crop', type=int, default=32, help='NxN crop (0=full 256)')
    args = ap.parse_args()

    IN_DIM, HIDDEN, OUT_DIM = 29, 48, 3
    FULL = 256

    # load FP32 model
    m = ParametricModel(IN_DIM, OUT_DIM, HIDDEN)
    sd = torch.load(args.weights, map_location='cpu')['model_state_dict']
    for n in ['layer1','layer2','layer3','residual']:
        getattr(m,n).weight.data = sd[f'models.1.model.{n}.weight']
        getattr(m,n).bias.data   = sd[f'models.1.model.{n}.bias']
    m.eval()

    # ---- report weight ranges FIRST (this tells you if Q8.8 even fits) ----
    print("=== weight/bias ranges (Q8.8 covers [-128, +127.996]) ===")
    for n in ['layer1','layer2','layer3','residual']:
        w = getattr(m,n).weight.data.numpy()
        b = getattr(m,n).bias.data.numpy()
        print(f"  {n:9s} W[min={w.min():+.4f} max={w.max():+.4f}]  "
              f"b[min={b.min():+.4f} max={b.max():+.4f}]")
    print()

    # quantize all weights/biases to Q8.8 raw
    qw = {n: to_q88(getattr(m,n).weight.data.numpy()) for n in ['layer1','layer2','layer3','residual']}
    qb = {n: to_q88(getattr(m,n).bias.data.numpy())   for n in ['layer1','layer2','layer3','residual']}

    # load capture image
    img = np.zeros((FULL,FULL,3), dtype=np.float32)
    with open(args.capture) as f:
        lines=[l.strip() for l in f if l.strip() and not l.strip().startswith('//')]
    for idx,line in enumerate(lines):
        line=line.zfill(24)
        B=hex_f32(line[0:8]); G=hex_f32(line[8:16]); R=hex_f32(line[16:24])
        img[idx//FULL, idx%FULL]=[R,G,B]
    img=np.clip(img,0,1)

    xs=np.linspace(0,1,FULL,dtype=np.float32); ys=np.linspace(0,1,FULL,dtype=np.float32)
    def reflect(c,N): return -c if c<0 else (2*(N-1)-c if c>=N else c)
    def px(x,y): return img[reflect(y,FULL),reflect(x,FULL)]

    def build_feat(x,y):
        f=[xs[x],ys[y]]; f.extend(img[y,x])
        for dy in(-1,0,1):
            for dx in(-1,0,1):
                if dy==0 and dx==0: continue
                f.extend(px(x+dx,y+dy))
        return np.array(f, dtype=np.float32)

    def fp32_forward(feat):
        with torch.no_grad():
            z = m(torch.tensor(feat).view(1,IN_DIM)).view(3).numpy()
        return z

    def q88_forward(feat):
        f_raw = to_q88(feat)
        h1 = relu_q88(q88_matvec(qw['layer1'], f_raw, qb['layer1']))
        h2 = relu_q88(q88_matvec(qw['layer2'], h1,    qb['layer2']))
        y  = q88_matvec(qw['layer3'], h2, qb['layer3'])
        yr = q88_matvec(qw['residual'], f_raw, qb['residual'])
        z  = q88_add(y, yr)
        return from_q88(z)   # back to float for comparison

    N = args.crop if args.crop>0 else FULL
    max_abs_err = 0.0
    sum_sq = 0.0
    count = 0
    worst = None
    for y in range(N):
        for x in range(N):
            feat = build_feat(x,y)
            z_fp = fp32_forward(feat)
            z_q  = q88_forward(feat)
            err = np.abs(z_fp - z_q)
            e = err.max()
            if e > max_abs_err:
                max_abs_err = e; worst=(x,y,z_fp.copy(),z_q.copy())
            sum_sq += np.sum(err**2); count += 3

    rmse = np.sqrt(sum_sq/count)
    print(f"=== Q8.8 vs FP32 on {N}x{N} pixels (MLP output z, pre-post_mlp) ===")
    print(f"  max abs error : {max_abs_err:.5f}")
    print(f"  RMSE          : {rmse:.5f}")
    if worst:
        wx,wy,zf,zq = worst
        print(f"  worst pixel   : ({wx},{wy})  fp={zf}  q88={zq}")
    print()
    print("Interpretation: z feeds sigmoid(4x-2)+center. A z error of ~0.01-0.05")
    print("is likely fine (sigmoid compresses it); >0.1 may show visibly. Check the")
    print("weight ranges above — if any |w| or |b| exceeds ~127, Q8.8 saturates and")
    print("you'll need more integer bits (e.g. Q10.6) or int-x-scale instead.")

if __name__ == '__main__':
    main()