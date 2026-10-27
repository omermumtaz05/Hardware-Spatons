#!/usr/bin/env python3
"""
gen_a12_golden.py — Generate golden verification vectors using JAVAD'S ACTUAL MLP.

Unlike the earlier exporter (which REBUILT the A12 architecture), this imports
Javad's real ParametricModel class from his codebase and runs it directly. That
removes the "did we reconstruct the module faithfully" risk — the golden now comes
from the exact code that produced the trained weights.

What it does:
  1. Load the real A12 (models.1) weights from the checkpoint into Javad's
     ParametricModel (complexity='complex', in=29, out=3, hidden=48).
  2. Run N inputs through it.
  3. Write "f | z" pairs to data_files/a12_golden.txt for the RTL testbench,
     where z is the RAW MLP output (before sigmoid / delta / clamp — matching
     what neural_network.sv computes).

IMPORTANT — this generates the golden for the RAW MLP output z = main + residual,
NOT the full A12 forward (which also does sigmoid, *4-2, +center, clamp). Your
neural_network.sv currently computes only up to z, so this matches that stage.
When you build sigmoid/delta/clamp, extend this to the full forward.

Usage:
  # option A: Javad's code is importable
  python gen_a12_golden.py --javad path/to/javad_code --weights "#86-NN2_best.pth"

  # option B: if you can't import his module cleanly, the fallback rebuilds
  # ParametricModel from his source inline (see REBUILD note below).
"""

import argparse
import struct
import sys
import numpy as np
import torch


def f32_hex(v):
    """FP32 value -> 8 hex chars, big-endian nibble order (matches RTL hex)."""
    return struct.pack('>f', float(v)).hex()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='#86-NN2_best.pth')
    ap.add_argument('--javad', default=None,
                    help="Directory containing Javad's model source (for import). "
                         "If omitted, uses the inline rebuild of ParametricModel.")
    ap.add_argument('-n', '--num', type=int, default=32, help='number of test vectors')
    ap.add_argument('-o', '--output', default='data_files/a12_golden.txt')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    IN_DIM, HIDDEN, OUT_DIM = 29, 48, 3

    # ---- load the real weights ----
    ckpt = torch.load(args.weights, map_location='cpu')
    sd = ckpt['model_state_dict']

    # ---- get Javad's ParametricModel ----
    model = None
    if args.javad:
        # Preferred path: import his actual class.
        sys.path.insert(0, args.javad)
        try:
            # Adjust this import to match Javad's module/file name that defines
            # ParametricModel. Common cases:
            #   from models import ParametricModel
            #   from javad import ParametricModel
            from models import ParametricModel  # noqa
            model = ParametricModel(IN_DIM, OUT_DIM, 'complex', hidden_dim=HIDDEN)
            print("Using Javad's imported ParametricModel")
        except Exception as e:
            print(f"Import failed ({e}); falling back to inline rebuild.")

    if model is None:
        # ---- REBUILD note ----
        # Inline rebuild transcribed from Javad's ParametricModel 'complex' branch:
        #   layer1: Linear(in, hidden); layer2: Linear(hidden, hidden);
        #   layer3: Linear(hidden, out); residual: Linear(in, out)
        #   forward: relu(layer1) -> relu(layer2) -> layer3 -> + residual(x)
        # This must match his forward EXACTLY. Verified against his source.
        import torch.nn as nn

        class ParametricModel(nn.Module):
            def __init__(s, in_dim, out_dim, complexity, hidden_dim=48):
                super().__init__()
                s.complexity = complexity
                s.layer1 = nn.Linear(in_dim, hidden_dim)
                s.layer2 = nn.Linear(hidden_dim, hidden_dim)
                s.layer3 = nn.Linear(hidden_dim, out_dim)
                s.residual = nn.Linear(in_dim, out_dim)
                s.activation = nn.ReLU()

            def forward(s, x):
                t = s.activation(s.layer1(x))
                t = s.activation(s.layer2(t))
                t = s.layer3(t)
                return t + s.residual(x)

        model = ParametricModel(IN_DIM, OUT_DIM, 'complex', HIDDEN)
        print("Using inline-rebuilt ParametricModel (verify against Javad's source)")

    # ---- load models.1 weights into the module ----
    model.layer1.weight.data   = sd['models.1.model.layer1.weight']
    model.layer1.bias.data     = sd['models.1.model.layer1.bias']
    model.layer2.weight.data   = sd['models.1.model.layer2.weight']
    model.layer2.bias.data     = sd['models.1.model.layer2.bias']
    model.layer3.weight.data   = sd['models.1.model.layer3.weight']
    model.layer3.bias.data     = sd['models.1.model.layer3.bias']
    model.residual.weight.data = sd['models.1.model.residual.weight']
    model.residual.bias.data   = sd['models.1.model.residual.bias']
    model.eval()

    # ---- run inputs, capture raw z ----
    rng = np.random.default_rng(args.seed)
    lines = []
    print(f"\nGenerating {args.num} golden vectors...")
    for t in range(args.num):
        f = rng.standard_normal(IN_DIM).astype(np.float32)
        with torch.no_grad():
            z = model(torch.tensor(f)).numpy()
        f_hex = " ".join(f32_hex(v) for v in f)
        z_hex = " ".join(f32_hex(v) for v in z)
        lines.append(f"{f_hex} | {z_hex}")
        if t < 5:
            print(f"  vec {t}: z = {z}")

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nWrote {args.num} vectors to {args.output}")
    print("Feed this to neural_network_tb (it already reads this format).")


if __name__ == '__main__':
    main()