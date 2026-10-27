#!/usr/bin/env python3
"""gen_mlp_golden_int.py -- regenerate a12_golden_mlp_q.txt for the INTEGER RTL.

The old golden was produced against a forward pass with NO pre-bias
requantize. nn_pipelined_int implements quant_core's requantize_direct, so the
old file is ~9.4e-3 off in z and every vector will look like a failure.

This emits the reference forward pass from quant_core -- float64 throughout,
including requantize_direct -- which is what the RTL now approximates in
integer. Expected RTL deviation from this file is ~1e-4, dominated by:
  - 18-bit reciprocal in requant_int   (<=1 LSB of 2047)
  - 24-bit M mantissa                  (~1e-6 relative)
  - B_int rounding at F=10             (~7e-6)
All far inside nn_streaming_quant_tb's tolerance (abs 5e-3 AND rel 3e-2).

Output format, matching the testbench's $fscanf loop:
    <29 fp32 hex, one per line>
    ---
    <3 fp32 hex, one per line>
repeated per vector.

Usage:
    python gen_mlp_golden_int.py --weights '#86-NN2_best.pth' \
        --capture data_files/capture_array.hex --n 64
"""

import argparse, struct
import numpy as np
import torch
import torch.nn as nn

from quant_core import (QuantizationConfig, HardwareConfig, uniform_quantize,
                        dequantize, quantized_matmul_custom, requantize_direct)

WEIGHT_BITS = 10
ACT_BITS    = 12
IN_DIM, HIDDEN, OUT_DIM = 29, 48, 3
FULL = 256
LAYERS = ['layer1', 'layer2', 'layer3', 'residual']

HW = HardwareConfig(accumulator_bits=64, use_saturation=True,
                    rounding_mode='nearest_even', requantize_outputs=True,
                    output_bits=8)


class ParametricModel(nn.Module):
    def __init__(self, i, o, h):
        super().__init__()
        self.layer1 = nn.Linear(i, h)
        self.layer2 = nn.Linear(h, h)
        self.layer3 = nn.Linear(h, o)
        self.residual = nn.Linear(i, o)


def f32_hex(v):
    return struct.pack('>f', float(v)).hex()


def hex_f32(h):
    return struct.unpack('>f', bytes.fromhex(h))[0]


def quant_linear(x_float, Wq, bias, act_bits, hw):
    """Exact replica of WeightActivationQuantizedLinear.forward."""
    acfg = QuantizationConfig(num_bits=act_bits, symmetric=True, per_channel=False)
    x_q, _, _ = uniform_quantize(x_float, acfg, per_channel=False)
    out_q = quantized_matmul_custom(x_q, Wq, hw)
    if hw.requantize_outputs:
        out_q = requantize_direct(out_q, act_bits, hw)   # <-- the step the RTL now has
    out_float = dequantize(out_q)
    if bias is not None:
        out_float = out_float + bias
    return out_float


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='#86-NN2_best.pth')
    ap.add_argument('--capture', default='data_files/capture_array.hex')
    ap.add_argument('--n', type=int, default=64, help='vectors to emit (MAX_VEC)')
    ap.add_argument('--stride', type=int, default=997,
                    help='pixel stride, so the vectors are not all neighbours')
    ap.add_argument('--out', default='data_files/a12_golden_mlp_q.txt')
    ap.add_argument('--sep', default='---')
    args = ap.parse_args()

    # ---- model ----
    m = ParametricModel(IN_DIM, OUT_DIM, HIDDEN)
    sd = torch.load(args.weights, map_location='cpu')['model_state_dict']
    for n in LAYERS:
        getattr(m, n).weight.data = sd[f'models.1.model.{n}.weight']
        getattr(m, n).bias.data = sd[f'models.1.model.{n}.bias']
    m.eval()

    wcfg = QuantizationConfig(num_bits=WEIGHT_BITS, symmetric=True, per_channel=False)
    Wq = {}
    for n in LAYERS:
        q, _, _ = uniform_quantize(getattr(m, n).weight.data, wcfg, per_channel=False)
        Wq[n] = q

    # ---- features, identical construction to gen_photo_golden_q.py ----
    img = np.zeros((FULL, FULL, 3), np.float32)
    with open(args.capture) as f:
        lines = [l.strip() for l in f
                 if l.strip() and not l.strip().startswith('//')]
    for idx, line in enumerate(lines):
        line = line.zfill(24)
        B, G, R = hex_f32(line[0:8]), hex_f32(line[8:16]), hex_f32(line[16:24])
        img[idx // FULL, idx % FULL] = [R, G, B]
    img = np.clip(img, 0, 1)

    xs = np.linspace(0, 1, FULL, dtype=np.float32)
    ys = np.linspace(0, 1, FULL, dtype=np.float32)

    def refl(c, N):
        return -c if c < 0 else (2 * (N - 1) - c if c >= N else c)

    def px(x, y):
        return img[refl(y, FULL), refl(x, FULL)]

    def feat(x, y):
        v = [xs[x], ys[y]]
        v.extend(img[y, x])
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                v.extend(px(x + dx, y + dy))
        return np.array(v, np.float32)

    # ---- emit ----
    zmin, zmax = 1e30, -1e30
    with open(args.out, 'w') as fo:
        for v in range(args.n):
            p = (v * args.stride) % (FULL * FULL)
            x, y = p % FULL, p // FULL
            ft = torch.tensor(feat(x, y)).view(1, IN_DIM)

            h1 = torch.relu(quant_linear(ft, Wq['layer1'], m.layer1.bias.data, ACT_BITS, HW))
            h2 = torch.relu(quant_linear(h1, Wq['layer2'], m.layer2.bias.data, ACT_BITS, HW))
            yv = quant_linear(h2, Wq['layer3'], m.layer3.bias.data, ACT_BITS, HW)
            yr = quant_linear(ft, Wq['residual'], m.residual.bias.data, ACT_BITS, HW)
            z = (yv + yr).view(OUT_DIM).detach().numpy()

            zmin = min(zmin, float(z.min()))
            zmax = max(zmax, float(z.max()))

            for a in ft.view(-1).detach().numpy():
                fo.write(f32_hex(a) + "\n")
            fo.write(args.sep + "\n")
            for a in z:
                fo.write(f32_hex(a) + "\n")

    print(f"wrote {args.out}: {args.n} vectors, stride {args.stride}")
    print(f"z range across the set: [{zmin:.4f}, {zmax:.4f}]")
    print()
    print("Testbench tolerance is abs > 5e-3 AND rel > 3e-2 to fail.")
    print("Expect RTL deviation ~1e-4. If you see ~9e-3 systematically, the RTL")
    print("is missing the pre-bias requantize; if you see garbage on one output")
    print("lane only, suspect LAMBDA3 / LAMBDA_RS rather than the datapath.")


if __name__ == '__main__':
    main()