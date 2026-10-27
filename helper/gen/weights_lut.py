#!/usr/bin/env python3
"""
Precompute the A1 coordinate warp grid as a lookup table.

Since A1 is a pure coordinate→coordinate MLP (input-independent), its output
grid is identical for every image at a given resolution. This script runs the
MLP once and saves the resulting grid so C++ can skip the MLP entirely.

Usage:
  python3 tools/export_a1_lut.py --weights spatons_weights.bin --height 256 --width 256
  python3 tools/export_a1_lut.py --weights spatons_weights.bin -o a1_lut.bin

LUT binary format:
  [uint32] magic   = 0x41314C54 ("A1LT")
  [uint32] version = 1
  [uint32] H
  [uint32] W
  [float32 × H × W × 2]  grid values in [-1, 1], row-major, (x, y) per pixel
"""

import argparse
import struct
import os
import numpy as np


WEIGHTS_MAGIC = 0x5350544E  # "SPTN"
LUT_MAGIC     = 0x41314C54  # "A1LT"
LUT_VERSION   = 1

A1_HIDDEN = 64
A1_IN     = 2
A1_OUT    = 2


def load_a1_weights(weights_path):
    """Read spatons_weights.bin and extract the A1 layer weights."""
    with open(weights_path, 'rb') as f:
        magic, version, n_floats = struct.unpack('<III', f.read(12))
        if magic != WEIGHTS_MAGIC:
            raise RuntimeError(f"Invalid weights magic: 0x{magic:08X}")
        data = np.fromfile(f, dtype=np.float32, count=n_floats)

    off = 0
    def take(shape):
        nonlocal off
        n = int(np.prod(shape))
        arr = data[off:off+n].reshape(shape)
        off += n
        return arr

    W1 = take((A1_HIDDEN, A1_IN));   b1 = take((A1_HIDDEN,))
    W2 = take((A1_HIDDEN, A1_HIDDEN)); b2 = take((A1_HIDDEN,))
    W3 = take((A1_OUT, A1_HIDDEN));  b3 = take((A1_OUT,))

    return W1, b1, W2, b2, W3, b3


def build_coord_grid(H, W):
    """Normalised (x, y) coordinate grid matching ApproachA1::forward."""
    ys = np.linspace(0, 1, H, dtype=np.float32) if H > 1 else np.array([0.5], dtype=np.float32)
    xs = np.linspace(0, 1, W, dtype=np.float32) if W > 1 else np.array([0.5], dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)  # [H*W, 2]
    return coords


def forward_a1_mlp(coords, W1, b1, W2, b2, W3, b3):
    """Replicate ParametricModel forward with identity residual."""
    h1 = np.maximum(0, coords @ W1.T + b1)
    h2 = np.maximum(0, h1 @ W2.T + b2)
    out = h2 @ W3.T + b3 + coords  # identity residual (in_dim == out_dim)
    return out


def main():
    p = argparse.ArgumentParser(description='Export A1 coordinate LUT')
    p.add_argument('--weights', required=True, help='Path to spatons_weights.bin')
    p.add_argument('--height', type=int, default=256)
    p.add_argument('--width',  type=int, default=256)
    p.add_argument('-o', '--output', default=None,
                   help='Output path (default: a1_lut_HxW.bin)')
    args = p.parse_args()

    H, W = args.height, args.width
    out_path = args.output or f'a1_lut_{H}x{W}.bin'

    print(f"Loading A1 weights from: {args.weights}")
    W1, b1, W2, b2, W3, b3 = load_a1_weights(args.weights)

    print(f"Building coordinate grid: {H}×{W} ({H*W} pixels)")
    coords = build_coord_grid(H, W)

    print("Running A1 MLP forward pass...")
    transformed = forward_a1_mlp(coords, W1, b1, W2, b2, W3, b3)

    # Clamp to [0,1] then scale to [-1,1] — matches ApproachA1::forward steps 3–4
    transformed = np.clip(transformed, 0.0, 1.0)
    grid = (2.0 * transformed - 1.0).astype(np.float32)  # [H*W, 2]
    grid = grid.reshape(H, W, 2)

    print(f"Grid stats: min={grid.min():.6f}  max={grid.max():.6f}  mean={grid.mean():.6f}")

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<IIII', LUT_MAGIC, LUT_VERSION, H, W))
        grid.tofile(f)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Written: {out_path}  ({size_kb:.1f} KB)")
    print("Done!")


if __name__ == '__main__':
    main()