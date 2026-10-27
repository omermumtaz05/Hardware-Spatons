#!/usr/bin/env python3
"""
gen_patch_golden.py — Golden vectors for the patch_extractor, using Javad's EXACT
feature-assembly from ApproachA12.forward:
  - reflection-pad the image
  - 3x3 patch per pixel
  - center coords normalized [0,1] via linspace (x first, then y)
  - center RGB
  - 8 neighbor RGB in raster order (skip center)
  = 29 features

Outputs:
  data_files/patch_image.hex   — a small test image, RGB per pixel, row-major,
                                  one FP32 per line (so RTL inter_buffer can load it)
  data_files/patch_golden.txt  — "<cx cy> | <29 feature hex>" per test pixel
                                  (cx, cy are integer pixel coords the RTL takes)
"""

import argparse, struct, os
import numpy as np

W = H = 8   # small image is enough to exercise edges + interior


def f32_hex(v):
    return struct.pack('>f', float(v)).hex()


def reflect_pad_index(c, N):
    """Match F.pad mode='reflect' for pad_size=1."""
    if c < 0:      return -c
    if c >= N:     return 2*(N-1) - c
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--img', default='data_files/patch_image.hex')
    ap.add_argument('--golden', default='data_files/patch_golden.txt')
    args = ap.parse_args()
    os.makedirs('data_files', exist_ok=True)

    rng = np.random.default_rng(args.seed)
    # random RGB image in [0,1], shape [H, W, 3]
    img = rng.uniform(0, 1, size=(H, W, 3)).astype(np.float32)

    # normalized coord grid (Javad: linspace(0,1,W) for x, linspace(0,1,H) for y)
    xs = np.linspace(0, 1, W, dtype=np.float32)
    ys = np.linspace(0, 1, H, dtype=np.float32)

    # ---- write image hex: row-major, RGB, one FP32 per line ----
    with open(args.img, 'w') as f:
        for y in range(H):
            for x in range(W):
                for c in range(3):
                    f.write(f32_hex(img[y, x, c]) + "\n")
    print(f"Wrote image: {args.img} ({W}x{H} RGB)")

    # ---- build golden feature vector for every pixel ----
    # test a spread: corners, edges, interior
    test_pixels = [(0,0), (W-1,0), (0,H-1), (W-1,H-1),   # corners
                   (0,3), (W-1,3), (3,0), (3,H-1),        # edges
                   (3,3), (4,2), (2,5), (5,5)]            # interior

    def get_px(x, y):
        rx = reflect_pad_index(x, W)
        ry = reflect_pad_index(y, H)
        return img[ry, rx]  # RGB

    lines = []
    for (cx, cy) in test_pixels:
        feat = []
        # center coords normalized [0,1], x then y
        feat.append(xs[cx])
        feat.append(ys[cy])
        # center RGB
        feat.extend(img[cy, cx])
        # 8 neighbors raster order skipping center: (dy,dx) for dy,dx in -1,0,1
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                feat.extend(get_px(cx+dx, cy+dy))
        assert len(feat) == 29, len(feat)
        feat_hex = " ".join(f32_hex(v) for v in feat)
        lines.append(f"{cx} {cy} | {feat_hex}")

    with open(args.golden, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote golden: {args.golden} ({len(test_pixels)} test pixels)")
    print("Golden format: <cx cy> | <29 feature FP32 hex>")


if __name__ == '__main__':
    main()