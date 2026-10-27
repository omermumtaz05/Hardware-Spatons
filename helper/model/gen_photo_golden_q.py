#!/usr/bin/env python3
"""export_a12.py — Quantize A12 MLP exactly as Javad's WeightActivationQuantizedLinear
does, export int weights + scales + intermediate goldens for RTL.

Mirrors his forward EXACTLY:
  per layer: x_q = quant(x, act_bits); out_q = matmul(x_q, W_q);
             out_q = requantize(out_q, act_bits); out = dequant(out_q) + bias
  ReLU (float) after layer1, layer2. layer3 + residual, then float add.
"""

import argparse, struct
import numpy as np
import torch, torch.nn as nn
from quant_core import (QuantizationConfig, HardwareConfig, uniform_quantize,
                        dequantize, quantized_matmul_custom, requantize_direct)
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
def hex_f32(h): return struct.unpack('>f', bytes.fromhex(h))[0]

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
    """Replicate WeightActivationQuantizedLinear.forward exactly.
    x_float: [N, in] float tensor. Wq: pre-quantized weight QuantizedTensor.
    Returns (out_float, x_q_values, out_q_values) for golden dumping."""
    wcfg_dummy = None
    acfg = QuantizationConfig(num_bits=act_bits, symmetric=True, per_channel=False)
    x_q, _, _ = uniform_quantize(x_float, acfg, per_channel=False)
    out_q = quantized_matmul_custom(x_q, Wq, hw)
    if hw.requantize_outputs:
        out_q = requantize_direct(out_q, act_bits, hw)
    out_float = dequantize(out_q)               # back to float
    if bias is not None:
        out_float = out_float + bias            # bias add in float
    return out_float, x_q.values, out_q.values

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='#86-NN2_best.pth')
    ap.add_argument('--capture', default='data_files/capture_array.hex')
    ap.add_argument('--crop', type=int, default=16)
    ap.add_argument('--out', default='data_files')
    args = ap.parse_args()

    IN_DIM, HIDDEN, OUT_DIM = 29, 48, 3
    FULL = 256

    # load FP32 A12 weights
    m = ParametricModel(IN_DIM, OUT_DIM, HIDDEN)
    sd = torch.load(args.weights, map_location='cpu')['model_state_dict']
    for n in ['layer1','layer2','layer3','residual']:
        getattr(m,n).weight.data = sd[f'models.1.model.{n}.weight']
        getattr(m,n).bias.data   = sd[f'models.1.model.{n}.bias']
    m.eval()

    # pre-quantize all weights (10-bit, symmetric, per-tensor) — his config
    wcfg = QuantizationConfig(num_bits=WEIGHT_BITS, symmetric=True, per_channel=False)
    Wq = {}; scales = {}
    for n in ['layer1','layer2','layer3','residual']:
        q, sc, _ = uniform_quantize(getattr(m,n).weight.data, wcfg, per_channel=False)
        Wq[n] = q
        scales[n] = float(q.scale[0])

    # --- load capture, build features, run quantized forward, dump goldens ---
    img=np.zeros((FULL,FULL,3),np.float32)
    with open(args.capture) as f:
        lines=[l.strip() for l in f if l.strip() and not l.strip().startswith('//')]
    for idx,line in enumerate(lines):
        line=line.zfill(24); B=hex_f32(line[0:8]);G=hex_f32(line[8:16]);R=hex_f32(line[16:24])
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
    ff = open(f"{args.out}/photo_features_q.hex", 'w')   # 29 fp32/px, flat
    fg = open(f"{args.out}/photo_golden_q.hex",   'w')   # 3 fp32/px, flat
    for y in range(N):
        for x in range(N):
            ft = torch.tensor(feat(x,y)).view(1, IN_DIM)
            # replicate his forward exactly:
            h1, _, _ = quant_linear(ft, Wq['layer1'], m.layer1.bias.data, ACT_BITS, HW)
            h1 = torch.relu(h1)
            h2, _, _ = quant_linear(h1, Wq['layer2'], m.layer2.bias.data, ACT_BITS, HW)
            h2 = torch.relu(h2)
            yv, _, _ = quant_linear(h2, Wq['layer3'], m.layer3.bias.data, ACT_BITS, HW)
            yr, _, _ = quant_linear(ft, Wq['residual'], m.residual.bias.data, ACT_BITS, HW)
            z = (yv + yr).view(3).detach().numpy()
            fv = ft.view(-1).detach().numpy()
            d   = np.array([lut_sig(zi) for zi in z], np.float32)
            fin = np.clip(img[y,x] + (4.0*d - 2.0), 0, 1).astype(np.float32)
            for v in fv:  ff.write(f32_hex(v)+"\n")
            for v in fin: fg.write(f32_hex(v)+"\n")
            if y*N + x == 9224:
                h1n = h1.view(-1).detach().numpy()
                print("GOLDEN h1:", " ".join(f32_hex(v) for v in h1n))
                print(f"GOLDEN z 9224: {f32_hex(z[0])} {f32_hex(z[1])} {f32_hex(z[2])}")
    ff.close(); fg.close()
    print(f"wrote photo_features_q.hex (29/px), photo_golden_q.hex (3/px), {N}x{N}")

    fimg = open(f"{args.out}/photo_image_q.hex", 'w')
    for y in range(N):
        for x in range(N):
            for v in img[y,x]:      # R,G,B order — match your buffer's channel order
                fimg.write(f32_hex(v)+"\n")
    fimg.close()

if __name__ == '__main__':
    main()