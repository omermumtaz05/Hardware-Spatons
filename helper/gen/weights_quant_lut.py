#!/usr/bin/env python3
"""
weights_quant_gen.py — Generate the A1 coordinate-warp LUT in Q8.8 fixed point.

A1 is a pure coordinate->coordinate MLP (input-independent), so its output grid
is identical for every image at a given resolution. This script runs the MLP
once, scales the result into pixel space [0, W-1], quantizes each coordinate to
unsigned Q8.8, and writes a ModelSim-loadable .hex file for the RTL ROM.

It also reports the Q8.8 quantization error against the fp32 reference grid, so
you can show precision loss stays under 1 LSB (1/256 ~= 0.00391).

Companion to weights_gen.py, which writes the fp32 .bin used by the C++ path.
That file is NOT replaced by this one -- different output, different consumer.

Hex line format (row-major, one entry per pixel):
  8 hex chars = {y_q88[15:0], x_q88[15:0]}, both unsigned Q8.8.

Usage:
  python3 weights_quant_gen.py --weights "#86-NN2_best.pth" -o lut.hex
  python3 weights_quant_gen.py --weights "#86-NN2_best.pth" --report
"""

import argparse
import numpy as np
import torch


A1_HIDDEN = 64
A1_IN     = 2
A1_OUT    = 2


def load_a1_weights(ckpt_path):
    """Extract the A1 layer weights from a training checkpoint (.pth)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"]
    W1 = sd["models.0.model.layer1.weight"].numpy()
    b1 = sd["models.0.model.layer1.bias"].numpy()
    W2 = sd["models.0.model.layer2.weight"].numpy()
    b2 = sd["models.0.model.layer2.bias"].numpy()
    W3 = sd["models.0.model.layer3.weight"].numpy()
    b3 = sd["models.0.model.layer3.bias"].numpy()
    return W1, b1, W2, b2, W3, b3


def build_coord_grid(H, W):
    """Normalised (x, y) coordinate grid matching ApproachA1::forward."""
    ys = np.linspace(0, 1, H, dtype=np.float32) if H > 1 else np.array([0.5], dtype=np.float32)
    xs = np.linspace(0, 1, W, dtype=np.float32) if W > 1 else np.array([0.5], dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)  # [H*W, 2]
    return coords


def forward_a1_mlp(coords, W1, b1, W2, b2, W3, b3):
    """Replicate ParametricModel forward with identity residual."""
    h1 = np.maximum(0, coords @ W1.T + b1)
    h2 = np.maximum(0, h1 @ W2.T + b2)
    out = h2 @ W3.T + b3 + coords  # identity residual (in_dim == out_dim)
    return out


def build_fp32_grid(ckpt_path, H, W):
    """Run the MLP and return the fp32 reference grid in pixel space [0, W-1]."""
    W1, b1, W2, b2, W3, b3 = load_a1_weights(ckpt_path)
    coords = build_coord_grid(H, W)
    transformed = forward_a1_mlp(coords, W1, b1, W2, b2, W3, b3)
    transformed = np.clip(transformed, 0.0, 1.0)
    grid = (transformed * (W - 1)).astype(np.float32).reshape(H, W, 2)
    return grid


def to_q88(v):
    """Clamp to [0, 255.996], scale by 256, round -> 16-bit unsigned Q8.8."""
    q = int(round(float(v) * 256.0))
    return max(0, min(0xFFFF, q))


def write_hex(grid, out_path):
    """Write Q8.8 LUT: {y_q88, x_q88} packed into 8 hex chars per line."""
    H, W = grid.shape[0], grid.shape[1]
    with open(out_path, "w") as f:
        for y in range(H):
            for x in range(W):
                xq = to_q88(grid[y, x, 0])
                yq = to_q88(grid[y, x, 1])
                f.write(f"{yq:04x}{xq:04x}\n")
    print(f"wrote {H * W} Q8.8 entries to {out_path}")


def report_error(grid, hex_path):
    """Read the .hex back, dequantize, and compare against the fp32 grid."""
    H, W = grid.shape[0], grid.shape[1]
    err = []
    with open(hex_path) as f:
        for y in range(H):
            for x in range(W):
                line = f.readline().strip()
                yq = int(line[0:4], 16)
                xq = int(line[4:8], 16)
                err.append(abs(xq / 256.0 - grid[y, x, 0]))
                err.append(abs(yq / 256.0 - grid[y, x, 1]))
    err = np.array(err)
    print(f"Q8.8 quant error: max={err.max():.5f} mean={err.mean():.5f} "
          f"(expect < {1/256:.5f})")

    print("--- spot checks (fp32 vs dequantized Q8.8) ---")
    with open(hex_path) as f:
        lines = f.read().split()
    for (x, y) in [(0, 0), (128, 128), (255, 255), (5, 3), (200, 100)]:
        line = lines[y * W + x]
        xv = int(line[4:8], 16) / 256.0
        yv = int(line[0:4], 16) / 256.0
        gx, gy = grid[y, x]
        print(f"(x={x}, y={y}) fp32 x={gx:.6f} y={gy:.6f} | "
              f"q88 x={xv:.6f} y={yv:.6f}")


def main():
    p = argparse.ArgumentParser(description="Export A1 coordinate LUT in Q8.8")
    p.add_argument("--weights", default="#86-NN2_best.pth",
                   help="Path to training checkpoint (.pth)")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("-o", "--output", default="lut.hex", help="Output .hex path")
    p.add_argument("--report", action="store_true",
                   help="Print quantization error vs fp32 reference")
    args = p.parse_args()

    grid = build_fp32_grid(args.weights, args.height, args.width)
    print(f"grid stats: min={grid.min():.4f} max={grid.max():.4f} "
          f"mean={grid.mean():.4f}")
    write_hex(grid, args.output)
    if args.report:
        report_error(grid, args.output)


if __name__ == "__main__":
    main()