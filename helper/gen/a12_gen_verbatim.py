#!/usr/bin/env python3
"""
a12_gen_verbatim.py — Generate A12 MLP golden vectors using Javad's EXACT code.

This copies his ParametricModel class VERBATIM from javad.py (lines 4616-4818),
plus the trivial BaseModel base (line 4429). No rebuild, no reconstruction —
this is literally his class definition. The 'complex' branch is the A12 MLP.

Scope: RAW MLP output z = main + residual (the ParametricModel.complex forward).
Does NOT include the ApproachA12 wrapper (patch extraction, sigmoid, *4-2,
+center, clamp) — that's the full pixel pipeline you're building separately.

Usage (from repo root):
  python helper/a12_gen_verbatim.py --weights "#86-NN2_best.pth" -n 32 -o data_files/a12_golden.txt
"""

import argparse
import struct
import os
import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# VERBATIM from javad.py line 4429 — BaseModel (trivial nn.Module base)
# ============================================================================
class BaseModel(nn.Module):
    """Base class for all transformation models"""
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        """Transform input data - to be implemented by subclasses"""
        raise NotImplementedError


# ============================================================================
# VERBATIM from javad.py lines 4616-4818 — ParametricModel
# (only the parts reached by complexity='complex' matter for A12, but the
#  whole class is copied so it's unambiguously his code)
# ============================================================================
class ParametricModel(BaseModel):
    """Parametric transformation with configurable complexity levels"""
    def __init__(self, input_dim=2, output_dim=2, complexity='simple', *args, **kwargs):
        super(ParametricModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.complexity = complexity
        hidden_dim_arg = kwargs.pop('hidden_dim', None)
        kwargs.pop('hidden_dim1', None)
        kwargs.pop('hidden_dim2', None)

        if complexity == 'complex':
            # Complex: 2-hidden-layer MLP with residual connection
            hidden_dim = hidden_dim_arg if hidden_dim_arg is not None else max(input_dim, 16)
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, hidden_dim)
            self.layer3 = nn.Linear(hidden_dim, output_dim)
            self.activation = nn.ReLU()
            if input_dim == output_dim:
                self.residual = nn.Identity()
            else:
                self.residual = nn.Linear(input_dim, output_dim)
        else:
            raise ValueError(f"This golden script only supports complexity='complex' (A12). Got {complexity}")
        # NOTE: his _initialize_identity() only touches 2->2 models; A12 is 29->3,
        # so it's a no-op here, and we overwrite all weights from the checkpoint anyway.

    def forward(self, x):
        """Apply parametric transformation — VERBATIM 'complex' branch."""
        batch_size, height, width, channels = x.shape
        x_flat = x.contiguous().view(batch_size * height * width, channels)

        # === complexity == 'complex' (his exact lines) ===
        transformed = self.activation(self.layer1(x_flat))
        transformed = self.activation(self.layer2(transformed))
        transformed = self.layer3(transformed)
        residual = self.residual(x_flat)
        transformed = transformed + residual

        transformed = transformed.contiguous().view(batch_size, height, width, self.output_dim)
        return transformed


# ============================================================================
# Golden generation
# ============================================================================
def f32_hex(v):
    return struct.pack('>f', float(v)).hex()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='#86-NN2_best.pth')
    ap.add_argument('-n', '--num', type=int, default=32)
    ap.add_argument('-o', '--output', default='data_files/a12_golden.txt')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    IN_DIM, HIDDEN, OUT_DIM = 29, 48, 3

    # build his exact model
    model = ParametricModel(IN_DIM, OUT_DIM, 'complex', hidden_dim=HIDDEN)

    # load real A12 (models.1) weights
    ckpt = torch.load(args.weights, map_location='cpu')
    sd = ckpt['model_state_dict']
    model.layer1.weight.data   = sd['models.1.model.layer1.weight']
    model.layer1.bias.data     = sd['models.1.model.layer1.bias']
    model.layer2.weight.data   = sd['models.1.model.layer2.weight']
    model.layer2.bias.data     = sd['models.1.model.layer2.bias']
    model.layer3.weight.data   = sd['models.1.model.layer3.weight']
    model.layer3.bias.data     = sd['models.1.model.layer3.bias']
    model.residual.weight.data = sd['models.1.model.residual.weight']
    model.residual.bias.data   = sd['models.1.model.residual.bias']
    model.eval()

    print("Using Javad's VERBATIM ParametricModel (complex branch)")
    print(f"Shapes: {IN_DIM} -> {HIDDEN} -> {HIDDEN} -> {OUT_DIM}, residual {IN_DIM} -> {OUT_DIM}")

    rng = np.random.default_rng(args.seed)
    lines = []
    for t in range(args.num):
        f = rng.standard_normal(IN_DIM).astype(np.float32)
        # his forward wants [B, H, W, C]; wrap the vector as a 1x1 image
        x = torch.tensor(f).view(1, 1, 1, IN_DIM)
        with torch.no_grad():
            z = model(x).view(OUT_DIM).numpy()
        lines.append(f"{' '.join(f32_hex(v) for v in f)} | {' '.join(f32_hex(v) for v in z)}")
        if t < 5:
            print(f"  vec {t}: z = {z}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nWrote {args.num} vectors to {args.output}")


if __name__ == '__main__':
    main()