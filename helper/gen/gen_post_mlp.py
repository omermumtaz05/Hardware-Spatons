#!/usr/bin/env python3
"""
gen_post_mlp.py — Generate (1) a sigmoid LUT for the RTL, and (2) golden vectors
for the full post-MLP chain, matching Javad's ApproachA12.forward EXACTLY:

    delta = sigmoid(z)          # [0,1]
    delta = 4.0 * delta - 2.0   # [-2,2]
    final = center_rgb + delta  # add original center pixel RGB
    final = clamp(final, 0, 1)  # [0,1]

SIGMOID LUT DESIGN
------------------
sigmoid saturates: for z <= -8 it's ~0, for z >= +8 it's ~1. So we only need a
table over a bounded domain. We tabulate sigmoid over z in [-8, +8] with N entries,
uniformly spaced. The RTL:
  - if z <= -DOMAIN: output 0.0
  - if z >= +DOMAIN: output 1.0
  - else: index = round((z + DOMAIN) / step), output LUT[index]
Each LUT entry is FP32 (the sigmoid value), stored as 8-hex-char lines.

Outputs:
  data_files/sigmoid_lut.hex     — N FP32 sigmoid values
  data_files/post_mlp_golden.txt — "<3 z hex> <3 center_rgb hex> | <3 final hex>"
"""

import argparse
import struct
import os
import numpy as np

DOMAIN = 8.0          # tabulate sigmoid over [-DOMAIN, +DOMAIN]
LUT_N  = 4096         # number of table entries (power of 2, good resolution)


def f32_hex(v):
    return struct.pack('>f', float(v)).hex()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_lut():
    """Uniformly sample sigmoid over [-DOMAIN, +DOMAIN] with LUT_N points."""
    step = (2.0 * DOMAIN) / (LUT_N - 1)
    xs = np.array([-DOMAIN + i * step for i in range(LUT_N)], dtype=np.float64)
    vals = sigmoid(xs).astype(np.float32)
    return vals, step


def lut_sigmoid(z, lut, step):
    """Mirror the RTL's LUT lookup exactly, so the golden matches hardware."""
    if z <= -DOMAIN:
        return np.float32(0.0)
    if z >= DOMAIN:
        return np.float32(1.0)
    idx = int(round((z + DOMAIN) / step))
    idx = max(0, min(LUT_N - 1, idx))
    return lut[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--num', type=int, default=32, help='golden test vectors')
    ap.add_argument('--seed', type=int, default=2)
    ap.add_argument('--lut', default='data_files/sigmoid_lut.hex')
    ap.add_argument('--golden', default='data_files/post_mlp_golden.txt')
    args = ap.parse_args()

    os.makedirs('data_files', exist_ok=True)

    # ---- write sigmoid LUT ----
    lut, step = build_lut()
    with open(args.lut, 'w') as f:
        for v in lut:
            f.write(f32_hex(v) + "\n")
    print(f"Wrote sigmoid LUT: {args.lut} ({LUT_N} entries, domain +-{DOMAIN}, step {step:.6f})")

    # ---- generate post-MLP golden vectors ----
    # Uses the SAME lut_sigmoid the RTL will use, so hardware can match bit-closely.
    rng = np.random.default_rng(args.seed)
    lines = []
    for t in range(args.num):
        # z: raw MLP output, realistic range for this net (~[-6, +6])
        z = rng.uniform(-6, 6, size=3).astype(np.float32)
        # center_rgb: original pixel color in [0,1]
        center = rng.uniform(0, 1, size=3).astype(np.float32)

        # Javad's exact chain (using LUT sigmoid to match RTL)
        delta = np.array([lut_sigmoid(zi, lut, step) for zi in z], dtype=np.float32)
        delta = (4.0 * delta - 2.0).astype(np.float32)
        final = (center + delta).astype(np.float32)
        final = np.clip(final, 0.0, 1.0).astype(np.float32)

        z_hex = " ".join(f32_hex(v) for v in z)
        c_hex = " ".join(f32_hex(v) for v in center)
        o_hex = " ".join(f32_hex(v) for v in final)
        lines.append(f"{z_hex} {c_hex} | {o_hex}")
        if t < 5:
            print(f"  vec {t}: z={z} center={center} -> final={final}")

    with open(args.golden, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote post-MLP golden: {args.golden} ({args.num} vectors)")
    print("\nGolden line format: <z0 z1 z2> <c0 c1 c2> | <out0 out1 out2>  (all FP32 hex)")


if __name__ == '__main__':
    main()