#!/usr/bin/env python3
"""
gen_photo_golden.py — Full photo-stage golden on the REAL geo-warped capture.
Runs Javad's complete A12 pipeline (patch extract -> MLP -> sigmoid(LUT) -> 4x-2
-> +center -> clamp) on capture_array.hex (256x256, 96-bit packed RGB).

capture_array.hex format: one line per pixel, 24 hex chars = 3 FP32:
    line[0:8]   = p2 = B  (bits [95:64])
    line[8:16]  = p1 = G  (bits [63:32])
    line[16:24] = p0 = R  (bits [31:0])

Outputs:
  data_files/photo_out_golden.hex  — expected final RGB, 96-bit packed, same layout
"""

import argparse, struct, os
import numpy as np
import torch
import torch.nn as nn

W = H = 256
CAPTURE = "data_files/capture_array.hex"


def f32_hex(v): return struct.pack('>f', float(v)).hex()
def hex_f32(h): return struct.unpack('>f', bytes.fromhex(h))[0]


class ParametricModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, out_dim)
        self.residual = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()
    def forward(self, x):
        t = self.activation(self.layer1(x))
        t = self.activation(self.layer2(t))
        t = self.layer3(t)
        return t + self.residual(x)


def reflect_idx(c, N):
    if c < 0:  return -c
    if c >= N: return 2*(N-1) - c
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='#86-NN2_best.pth')
    args = ap.parse_args()
    os.makedirs('data_files', exist_ok=True)

    IN_DIM, HIDDEN, OUT_DIM = 29, 48, 3

    model = ParametricModel(IN_DIM, OUT_DIM, HIDDEN)
    sd = torch.load(args.weights, map_location='cpu')['model_state_dict']
    model.layer1.weight.data   = sd['models.1.model.layer1.weight']
    model.layer1.bias.data     = sd['models.1.model.layer1.bias']
    model.layer2.weight.data   = sd['models.1.model.layer2.weight']
    model.layer2.bias.data     = sd['models.1.model.layer2.bias']
    model.layer3.weight.data   = sd['models.1.model.layer3.weight']
    model.layer3.bias.data     = sd['models.1.model.layer3.bias']
    model.residual.weight.data = sd['models.1.model.residual.weight']
    model.residual.bias.data   = sd['models.1.model.residual.bias']
    model.eval()

    # ---- load capture (96-bit packed: p2=B p1=G p0=R) ----
    img = np.zeros((H, W, 3), dtype=np.float32)
    with open(CAPTURE) as f:
        lines = [ln.strip() for ln in f
                 if ln.strip() and not ln.strip().startswith("//")]
    assert len(lines) == W*H, f"expected {W*H} pixels, got {len(lines)}"
    for idx, line in enumerate(lines):
        line = line.zfill(24)
        B = hex_f32(line[0:8]); G = hex_f32(line[8:16]); R = hex_f32(line[16:24])
        y, x = idx // W, idx % W
        img[y, x] = [R, G, B]
    img = np.clip(img, 0, 1)

    # sigmoid LUT (match hardware)
    DOMAIN, LUT_N = 8.0, 4096
    step = (2.0*DOMAIN)/(LUT_N-1)
    def lut_sig(z):
        if z <= -DOMAIN: return np.float32(0.0)
        if z >=  DOMAIN: return np.float32(1.0)
        idx = int(round((z+DOMAIN)/step)); idx = max(0, min(LUT_N-1, idx))
        xi = -DOMAIN + idx*step
        return np.float32(1.0/(1.0+np.exp(-xi)))

    xs = np.linspace(0, 1, W, dtype=np.float32)
    ys = np.linspace(0, 1, H, dtype=np.float32)
    def px(x, y): return img[reflect_idx(y, H), reflect_idx(x, W)]

    out = np.zeros((H, W, 3), dtype=np.float32)
    print("running full A12 on capture (256x256, be patient)...")
    for y in range(H):
        for x in range(W):
            feat = [xs[x], ys[y]]
            feat.extend(img[y, x])
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0: continue
                    feat.extend(px(x+dx, y+dy))
            ft = torch.tensor(feat, dtype=torch.float32).view(1, IN_DIM)
            with torch.no_grad():
                z = model(ft).view(3).numpy()
            delta = np.array([lut_sig(zi) for zi in z], dtype=np.float32)
            delta = 4.0*delta - 2.0
            final = np.clip(img[y, x] + delta, 0.0, 1.0)
            out[y, x] = final.astype(np.float32)
        if y % 32 == 0:
            print(f"  row {y}/{H}")

    # write golden in SAME 96-bit packed format (p2=B p1=G p0=R)
    with open('data_files/photo_out_golden.hex', 'w') as f:
        for y in range(H):
            for x in range(W):
                R, G, Bc = out[y, x]
                f.write(f"{f32_hex(Bc)}{f32_hex(G)}{f32_hex(R)}\n")
    print("wrote data_files/photo_out_golden.hex (96-bit packed)")


if __name__ == '__main__':
    main()