#!/usr/bin/env python3
"""gen_int_consts.py -- constants for the all-integer inter-layer datapath.

Replaces the per-neuron fp32 dequant chain with:

    Y_j = (acc_j * M + round_bias) >> S  +  B_int_j

where M is ONE per-layer runtime scalar and B_int_j is a static per-neuron
integer. Emits nn_const_pkg.sv plus .hex bias ROMs, and reports the widths
and overflow margins you need before touching the RTL.

Algebra (per layer k):
    y = acc*s_act*s_w + b  =  s_w * ( acc*s_act + b/s_w )
    Y = 2^F * ( acc*s_act + b/s_w )        <-- what hardware computes
      = acc * (2^F * s_act) + B_int
    B_int = round(2^F * b / s_w)

    2^F*s_act for layer 1 / residual : from the fp32 feature abs-max
    2^F*s_act for layer 2 / layer 3  : s_w(prev) * Ymax(prev) / 2047
                                       (the previous layer's s_w -- not its own)

Y carries an implicit factor of lambda = s_w * 2^-F. It cancels at the next
layer's requantize and is paid back once, at the 3 final outputs.

Usage:
    python gen_int_consts.py --weights '#86-NN2_best.pth' \
        --capture data_files/capture_array.hex --crop 16
"""

import argparse, json, struct, sys
import numpy as np
import torch
import torch.nn as nn

from quant_core import QuantizationConfig, uniform_quantize

WEIGHT_BITS = 10
ACT_BITS    = 12
QMAX        = (1 << (ACT_BITS - 1)) - 1        # 2047
QMIN        = -(1 << (ACT_BITS - 1))           # -2048
IN_DIM, HIDDEN, OUT_DIM = 29, 48, 3
FULL = 256

LAYERS = ['layer1', 'layer2', 'layer3', 'residual']


class ParametricModel(nn.Module):
    def __init__(self, i, o, h):
        super().__init__()
        self.layer1 = nn.Linear(i, h)
        self.layer2 = nn.Linear(h, h)
        self.layer3 = nn.Linear(h, o)
        self.residual = nn.Linear(i, o)


def hex_f32(h):
    return struct.unpack('>f', bytes.fromhex(h))[0]


def f32_bits(v):
    return struct.unpack('>I', struct.pack('>f', float(v)))[0]


# ---------------------------------------------------------------- features

def load_features(capture, crop):
    """Rebuild the 29-feature vectors exactly as gen_photo_golden_q.py does."""
    img = np.zeros((FULL, FULL, 3), np.float32)
    with open(capture) as f:
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
        f = [xs[x], ys[y]]
        f.extend(img[y, x])
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                f.extend(px(x + dx, y + dy))
        return np.array(f, np.float32)

    N = crop if crop > 0 else FULL
    return np.stack([feat(x, y) for y in range(N) for x in range(N)]), img


def synth_features(n=4096, seed=0):
    """Fallback when no capture file is available. Coords in [0,1], RGB in [0,1]."""
    rng = np.random.default_rng(seed)
    f = np.zeros((n, IN_DIM), np.float32)
    f[:, 0:2] = rng.random((n, 2))
    f[:, 2:] = rng.random((n, IN_DIM - 2))
    return f, None


# ---------------------------------------------------------------- int model

def quant_vec(vals, qmax=QMAX):
    """Integer requantize: q = round(v * qmax / max|v|). Returns (q, max|v|).

    Every scale common to all elements cancels here -- this is why the wire
    between layers does not need to carry one.
    """
    m = int(np.abs(vals).max())
    if m == 0:
        return np.zeros_like(vals, dtype=np.int64), 0
    q = np.rint(vals.astype(np.float64) * qmax / m).astype(np.int64)
    return np.clip(q, QMIN, qmax), m


def run_int_layer(q_in, Wq, B_int, s2f, relu, model_requant, stats, tag):
    """One layer of the integer datapath.

    q_in  : int12 activations           s2f : 2^F * s_act (a real; fixed-pointed later)
    Wq    : int10 weights [out, in]     B_int: static integer biases
    """
    acc = q_in @ Wq.T                                  # int, dot_prod_int
    stats.setdefault(tag + '.acc', []).append(int(np.abs(acc).max()))

    if model_requant:
        # matches quant_core.requantize_direct: acc renormalised to 12 bits
        racc, maxacc = quant_vec(acc)
        M = maxacc * s2f / QMAX
        core = racc.astype(np.float64) * M
        stats.setdefault(tag + '.M', []).append(M)
        stats.setdefault(tag + '.mul', []).append(float(np.abs(core).max()))
    else:
        M = s2f
        core = acc.astype(np.float64) * M
        stats.setdefault(tag + '.M', []).append(M)
        stats.setdefault(tag + '.mul', []).append(float(np.abs(core).max()))

    Y = np.rint(core).astype(np.int64) + B_int
    if relu:
        Y = np.maximum(Y, 0)
    stats.setdefault(tag + '.Y', []).append(int(np.abs(Y).max()))
    return Y


def forward_int(x, Wq, B, s_w, F, model_requant, stats):
    """Full MLP on the integer datapath. x: fp32 feature vector [29]."""
    twoF = float(2 ** F)

    # ---- layer 1 + residual: the only fp32 input in the network ----
    xmax = float(np.abs(x).max())
    s_act = xmax / QMAX
    q0 = np.clip(np.rint(x.astype(np.float64) / s_act), QMIN, QMAX).astype(np.int64)
    stats.setdefault('s_act_L1', []).append(s_act)

    s2f_0 = twoF * s_act
    Y1 = run_int_layer(q0, Wq['layer1'], B['layer1'], s2f_0, True,
                       model_requant, stats, 'L1')
    YR = run_int_layer(q0, Wq['residual'], B['residual'], s2f_0, False,
                       model_requant, stats, 'RS')

    # ---- layer 2: s_act comes from the PREVIOUS layer's s_w and Ymax ----
    q1, Ymax1 = quant_vec(Y1)
    s2f_1 = s_w['layer1'] * Ymax1 / QMAX
    Y2 = run_int_layer(q1, Wq['layer2'], B['layer2'], s2f_1, True,
                       model_requant, stats, 'L2')

    q2, Ymax2 = quant_vec(Y2)
    s2f_2 = s_w['layer2'] * Ymax2 / QMAX
    Y3 = run_int_layer(q2, Wq['layer3'], B['layer3'], s2f_2, False,
                       model_requant, stats, 'L3')

    stats.setdefault('Ymax1', []).append(Ymax1)
    stats.setdefault('Ymax2', []).append(Ymax2)

    # ---- final combine: the two branches owe different lambdas ----
    lam3 = s_w['layer3'] / twoF
    lamR = s_w['residual'] / twoF
    z = Y3.astype(np.float64) * lam3 + YR.astype(np.float64) * lamR
    return z


def forward_float(x, Wq, m, s_w):
    """Reference: the current fp32 chain, for error comparison."""
    def lin(v, name, relu):
        vmax = float(np.abs(v).max())
        sa = vmax / QMAX
        q = np.clip(np.rint(v.astype(np.float64) / sa), QMIN, QMAX)
        acc = q @ Wq[name].T
        out = acc * sa * s_w[name] + getattr(m, name).bias.data.numpy().astype(np.float64)
        return np.maximum(out, 0) if relu else out

    h1 = lin(x.astype(np.float64), 'layer1', True)
    h2 = lin(h1, 'layer2', True)
    y3 = lin(h2, 'layer3', False)
    yr = lin(x.astype(np.float64), 'residual', False)
    return y3 + yr


# ---------------------------------------------------------------- emit

def bits_for(v):
    """Signed two's-complement width needed to hold integer v."""
    v = int(abs(v))
    return 1 if v == 0 else int(np.ceil(np.log2(v + 1))) + 1


def fixed_point(val, target_bits):
    """Represent a positive real as (mantissa, shift) with mantissa < 2^target_bits."""
    if val <= 0:
        return 0, 0
    shift = target_bits - 1 - int(np.floor(np.log2(val)))
    mant = int(round(val * (2.0 ** shift)))
    while mant >= (1 << target_bits):
        shift -= 1
        mant = int(round(val * (2.0 ** shift)))
    return mant, shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='#86-NN2_best.pth')
    ap.add_argument('--capture', default='data_files/capture_array.hex')
    ap.add_argument('--crop', type=int, default=16)
    ap.add_argument('--out', default='data_files')
    ap.add_argument('-F', type=int, default=16,
                    help='fractional bits on the bias constants')
    ap.add_argument('--mant-bits', type=int, default=18,
                    help='width of the runtime scale mantissa M (18 = one Cyclone V DSP)')
    ap.add_argument('--no-requant', action='store_true',
                    help='skip the pre-bias 12-bit requantize (diverges from quant_core)')
    args = ap.parse_args()

    F = args.F
    model_requant = not args.no_requant

    # ---- weights ----
    m = ParametricModel(IN_DIM, OUT_DIM, HIDDEN)
    sd = torch.load(args.weights, map_location='cpu')['model_state_dict']
    for n in LAYERS:
        getattr(m, n).weight.data = sd[f'models.1.model.{n}.weight']
        getattr(m, n).bias.data = sd[f'models.1.model.{n}.bias']
    m.eval()

    wcfg = QuantizationConfig(num_bits=WEIGHT_BITS, symmetric=True, per_channel=False)
    Wq, s_w = {}, {}
    for n in LAYERS:
        q, _, _ = uniform_quantize(getattr(m, n).weight.data, wcfg, per_channel=False)
        Wq[n] = q.values.astype(np.int64)
        s_w[n] = float(q.scale[0])

    print('=' * 68)
    print('weight scales (s_w) recovered from the checkpoint')
    print('=' * 68)
    for n in LAYERS:
        print(f'  {n:10s} {s_w[n]:.12e}')

    # ---- B_int ----
    B = {}
    print()
    print('=' * 68)
    print(f'integer biases  B_int = round(2^{F} * b / s_w)')
    print('=' * 68)
    for n in LAYERS:
        b = getattr(m, n).bias.data.numpy().astype(np.float64)
        B[n] = np.rint(b * (2.0 ** F) / s_w[n]).astype(np.int64)
        err = np.abs(B[n] * s_w[n] / (2.0 ** F) - b).max()
        print(f'  {n:10s} n={len(B[n]):3d}  |B|max={np.abs(B[n]).max():>12d} '
              f'({bits_for(np.abs(B[n]).max())} bits)  round err={err:.3e}')

    # ---- calibration sweep ----
    try:
        feats, _ = load_features(args.capture, args.crop)
        src = f'{args.capture} ({len(feats)} px)'
    except (FileNotFoundError, OSError):
        feats, _ = synth_features()
        src = 'SYNTHETIC (capture not found -- widths are indicative only)'

    stats = {}
    errs = []
    for x in feats:
        z_int = forward_int(x, Wq, B, s_w, F, model_requant, stats)
        z_ref = forward_float(x, Wq, m, s_w)
        errs.append(np.abs(z_int - z_ref).max())
    errs = np.array(errs)

    print()
    print('=' * 68)
    print(f'calibration: {src}')
    print(f'pre-bias 12-bit requantize: {"MODELLED" if model_requant else "SKIPPED"}')
    print('=' * 68)

    # ---- widths / overflow ----
    print()
    print(f'{"signal":<14}{"max |value|":>18}{"bits":>7}   headroom vs int32')
    print('-' * 68)
    rows = []
    for tag in ['L1', 'RS', 'L2', 'L3']:
        rows.append((f'{tag} acc', max(stats[tag + '.acc'])))
        rows.append((f'{tag} acc*M', max(stats[tag + '.mul'])))
        rows.append((f'{tag} Y', max(stats[tag + '.Y'])))
    for name, v in rows:
        b = bits_for(v)
        head = (2 ** 31 - 1) / max(abs(v), 1)
        flag = '  <-- OVERFLOW' if b > 32 else ('  <-- tight' if head < 4 else '')
        print(f'{name:<14}{v:>18.0f}{b:>7}   {head:>8.1f}x{flag}')

    worst = max(bits_for(v) for _, v in rows)
    print('-' * 68)
    print(f'widest intermediate: {worst} bits  ->  '
          f'{"int32 is sufficient" if worst <= 32 else f"NEED {worst}-bit accumulator"}')

    # ---- analytic worst case: holds for ANY input image ----
    #
    # |q| <= 2047 at every layer by construction, so
    #     |acc| <= 2047 * max_j sum_i |Wq[j,i]|
    # and the acc*M product telescopes:
    #     L1: |acc*M| <= (sum|W1|) * 2^F * XFEAT_MAX
    #     Lk: |acc*M| <= (sum|Wk|) * s_w[k-1] * Ybound[k-1]
    # No image can exceed these. If they fit, overflow is impossible.
    XFEAT_MAX = 1.0        # coords in [0,1], RGB clipped to [0,1]
    wsum = {n: int(np.abs(Wq[n]).sum(axis=1).max()) for n in LAYERS}
    bmax = {n: int(np.abs(B[n]).max()) for n in LAYERS}

    print()
    print('=' * 68)
    print('ANALYTIC worst case -- independent of which images you run')
    print('=' * 68)
    bound = {}
    mul1 = wsum['layer1'] * (2.0 ** F) * XFEAT_MAX
    bound['layer1'] = mul1 + bmax['layer1']
    mulR = wsum['residual'] * (2.0 ** F) * XFEAT_MAX
    bound['residual'] = mulR + bmax['residual']
    mul2 = wsum['layer2'] * s_w['layer1'] * bound['layer1']
    bound['layer2'] = mul2 + bmax['layer2']
    mul3 = wsum['layer3'] * s_w['layer2'] * bound['layer2']
    bound['layer3'] = mul3 + bmax['layer3']

    print(f'{"layer":<12}{"row sum |Wq|":>14}{"max |acc*M|":>18}'
          f'{"max |Y|":>18}{"bits":>7}')
    print('-' * 68)
    for n, mul in [('layer1', mul1), ('residual', mulR),
                   ('layer2', mul2), ('layer3', mul3)]:
        b = bits_for(bound[n])
        flag = '  <-- OVERFLOW' if b > 32 else ''
        print(f'{n:<12}{wsum[n]:>14}{mul:>18.0f}{bound[n]:>18.0f}{b:>7}{flag}')

    hard = max(bits_for(v) for v in bound.values())
    print('-' * 68)
    margin = (2 ** 31 - 1) / max(bound.values())
    if hard <= 32:
        print(f'PROVABLY SAFE in int32 for any input image ({margin:.1f}x margin).')
        print(f'Measured worst case was {worst} bits; bound is {hard} bits.')
    else:
        need_F = F - (hard - 32)
        print(f'BOUND EXCEEDS int32 ({hard} bits). Options:')
        print(f'  - rerun with -F {need_F}  (each bit off F halves every intermediate)')
        print(f'  - or widen BW_Y to {hard} bits in the RTL')

    # ---- runtime scalar M ----
    print()
    print('=' * 68)
    print(f'runtime scale M, fixed-pointed to {args.mant_bits} bits')
    print('=' * 68)
    Mconst = {}
    for tag, layer in [('L1', 'layer1'), ('RS', 'residual'),
                       ('L2', 'layer2'), ('L3', 'layer3')]:
        vals = np.array(stats[tag + '.M'])
        lo, hi = vals.min(), vals.max()
        mant, sh = fixed_point(hi, args.mant_bits)
        rel = abs(mant * 2.0 ** -sh - hi) / hi
        Mconst[layer] = (mant, sh)
        print(f'  {tag}  range [{lo:.6e}, {hi:.6e}]  span {hi/max(lo,1e-30):.1f}x')
        print(f'       mantissa={mant} shift={sh}  rel err at max={rel:.2e}')
    print()
    print('  M spans several orders of magnitude across pixels, so it must be')
    print('  computed at runtime as (C_fx * Ymax) >> shift, not stored flat.')

    # ---- static per-layer C constants ----
    print()
    print('=' * 68)
    print('static constants for the RTL')
    print('=' * 68)
    C2 = s_w['layer1'] / QMAX
    C3 = s_w['layer2'] / QMAX
    c2m, c2s = fixed_point(C2, 24)
    c3m, c3s = fixed_point(C3, 24)
    print(f'  C2 = s_w[layer1]/2047 = {C2:.12e}   -> {c2m} >> {c2s}')
    print(f'  C3 = s_w[layer2]/2047 = {C3:.12e}   -> {c3m} >> {c3s}')
    lam3 = s_w['layer3'] / 2.0 ** F
    lamR = s_w['residual'] / 2.0 ** F
    print(f'  lambda3   = s_w[layer3]/2^{F}   = {lam3:.12e}  fp32 0x{f32_bits(lam3):08x}')
    print(f'  lambdaRes = s_w[residual]/2^{F} = {lamR:.12e}  fp32 0x{f32_bits(lamR):08x}')
    print(f'  ratio lambda3/lambdaRes = {lam3/lamR:.6f}  '
          f'(not a power of 2 -- the two branches cannot be added as raw ints)')

    # ---- accuracy ----
    print()
    print('=' * 68)
    print('accuracy of the integer datapath vs the fp32 chain (pre-sigmoid z)')
    print('=' * 68)
    print(f'  max  abs err : {errs.max():.4e}')
    print(f'  mean abs err : {errs.mean():.4e}')
    print(f'  p99  abs err : {np.percentile(errs, 99):.4e}')
    print('  (your end-to-end target is 1.35e-2 after the sigmoid, which')
    print('   compresses this -- z error here should land well under that)')

    # ---- emit ----
    for n in LAYERS:
        path = f'{args.out}/bias_int_{n}.hex'
        with open(path, 'w') as f:
            for v in B[n]:
                f.write(f'{int(v) & 0xFFFFFFFF:08x}\n')
        print(f'\nwrote {path}  ({len(B[n])} entries)')

    pkg = f'''// generated by gen_int_consts.py -- do not edit
package nn_const_pkg;
    parameter int F           = {F};
    parameter int BW_ACT      = {ACT_BITS};
    parameter int BW_W        = {WEIGHT_BITS};
    parameter int BW_Y        = {max(32, worst)};
    parameter int MANT_BITS   = {args.mant_bits};

    parameter int C2_MANT     = {c2m};
    parameter int C2_SHIFT    = {c2s};
    parameter int C3_MANT     = {c3m};
    parameter int C3_SHIFT    = {c3s};

    parameter logic [31:0] LAMBDA3   = 32'h{f32_bits(lam3):08x};
    parameter logic [31:0] LAMBDA_RS = 32'h{f32_bits(lamR):08x};
endpackage
'''
    with open(f'{args.out}/nn_const_pkg.sv', 'w') as f:
        f.write(pkg)
    print(f'wrote {args.out}/nn_const_pkg.sv')

    with open(f'{args.out}/int_consts.json', 'w') as f:
        json.dump({'F': F, 's_w': s_w,
                   'B_int': {k: v.tolist() for k, v in B.items()},
                   'C2': C2, 'C3': C3, 'lambda3': lam3, 'lambda_res': lamR,
                   'widest_bits': worst}, f, indent=2)
    print(f'wrote {args.out}/int_consts.json')


if __name__ == '__main__':
    main()