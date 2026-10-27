#!/usr/bin/env python3
"""gen_shift_consts.py -- constants for the shift-scaled datapath.

    python gen_shift_consts.py --weights '#86-NN2_best.pth' --out data_files

Emits, into --out:
    a12_layer1_w_q.hex   layer1 weights, PER-CHANNEL quantised   (changed)
    a12_layer2_w_q.hex   layer2 weights, PER-CHANNEL quantised   (changed)
    a12_layer3_w_q.hex   layer3 weights, per-tensor              (unchanged)
    a12_residual_w_q.hex residual weights, per-tensor            (unchanged)
    km_layer1.hex        per-neuron K_MANT for layer1            (new)
    km_layer2.hex        per-neuron K_MANT for layer2            (new)
    bias_int_*.hex       integer biases                          (unchanged)
and prints the scalar parameters for nn_pipelined_int.sv.

===========================================================================
WHY PER-CHANNEL ON LAYER1 AND LAYER2

A per-tensor weight scale is set by max|w| across the whole matrix. Layer2's
max is 67x its median, so the MEDIAN weight resolves to ~2.9 of its 10 bits:

    layer     max|w|   p50|w|  max/p50   eff. bits at the median
    layer1    0.8061   0.0417     19.3       4.7
    layer2    7.5074   0.1115     67.3       2.9
    layer3    1.8148   0.2524      7.2       6.2
    residual  0.1777   0.0421      4.2       6.9

One scale per output neuron fixes that. Measured effect on the FINAL PIXEL
(8-bit LSBs, pooled over 9 test images incl. noise, checkerboard, flat):

    per-channel on          mean    p99    max   >2LSB
    none (current)         1.356  8.916 19.475  26.63%
    layer2 only            0.965  6.108 12.367  18.90%
    layer1 only            1.165  7.639 16.746  22.57%
    layer1 + layer2        0.436  2.791  8.413   3.36%   <-- take this
    + layer3 + residual    0.444  2.832  8.624   3.66%   (no further gain)

3.1x better, and it costs ZERO DSP blocks -- the per-neuron multiply is held
in ALMs by multstyle="logic" in layer_int.sv, ~2,100 ALMs of the 32,070
available. DSP blocks are the binding resource; ALMs are not.

===========================================================================
WHAT DOES NOT CHANGE

s_ref for a per-channel layer is max_j(s_j), which is exactly the old
per-tensor scale (both are max|w|/511). So:
    - the integer biases B_int are unchanged
    - layer3's scalar K_MANT/K_SHIFT are unchanged
    - LAMBDA3 / LAMBDA_RS are unchanged
Only the layer1/layer2 WEIGHT hex files and the two new K_MANT ROMs move.
"""
import argparse, math, os, struct
import numpy as np

WEIGHT_BITS, F, QF, BW_K = 10, 10, 24, 19
WMAX = (1 << (WEIGHT_BITS - 1)) - 1        # 511
KM_MAX = 1 << (BW_K - 1)                   # 2^18
LAYERS = ['layer1', 'layer2', 'layer3', 'residual']
PER_CHANNEL = {'layer1', 'layer2'}


def f32_bits(v):
    return struct.unpack('>I', struct.pack('>f', float(v)))[0]


def load_weights(path):
    try:
        import torch
        sd = torch.load(path, map_location='cpu')['model_state_dict']
        get = lambda k: sd[k].detach().cpu().numpy().astype(np.float64)
    except ImportError:
        from load_pth import load                     # torch-free fallback
        sd = load(path)['model_state_dict']
        get = lambda k: np.asarray(sd[k], dtype=np.float64)
    w = {n: get(f'models.1.model.{n}.weight') for n in LAYERS}
    b = {n: get(f'models.1.model.{n}.bias') for n in LAYERS}
    return w, b


def quantise(w, per_channel):
    """Returns (Wq, s_per_neuron, s_ref). s_ref == the old per-tensor scale."""
    if per_channel:
        am = np.abs(w).max(axis=1)
        s = np.where(am > 1e-10, am / WMAX, 1e-10)
        q = np.clip(np.round(w / s[:, None]), -(WMAX + 1), WMAX).astype(np.int64)
    else:
        am = np.abs(w).max()
        sc = am / WMAX if am > 1e-10 else 1e-10
        s = np.full(w.shape[0], sc)
        q = np.clip(np.round(w / sc), -(WMAX + 1), WMAX).astype(np.int64)
    return q, s, float(s.max())


def pick_ks(target_max):
    """Largest K_SHIFT with round(target_max * 2^KS) still under 2^(BW_K-1)."""
    ks = (BW_K - 2) - math.floor(math.log2(target_max))
    while round(target_max * 2.0 ** ks) >= KM_MAX:
        ks -= 1
    return ks


def write_hex(path, vals, nbits):
    mask = (1 << nbits) - 1
    with open(path, 'w') as f:
        for v in np.asarray(vals).ravel():
            f.write(f'{int(v) & mask:0{(nbits + 3) // 4}x}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='#86-NN2_best.pth')
    ap.add_argument('--out', default='data_files')
    ap.add_argument('--per-tensor', action='store_true',
                    help='disable per-channel on L1/L2 (reverts to the old, '
                         '3x worse, accuracy)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pc = set() if args.per_tensor else PER_CHANNEL

    w, b = load_weights(args.weights)
    Wq, S, sref = {}, {}, {}
    for n in LAYERS:
        Wq[n], S[n], sref[n] = quantise(w[n], n in pc)

    print('=' * 68)
    print(f'weight quantisation ({"PER-TENSOR (degraded)" if args.per_tensor else "per-channel on L1+L2"})')
    print('=' * 68)
    for n in LAYERS:
        mode = 'per-channel' if n in pc else 'per-tensor '
        print(f'  {n:9s} {mode}  s_ref = {sref[n]:.9e}  '
              f'(s_j spread {S[n].max()/S[n].min():.1f}x)')

    # ---- integer biases: use s_ref, so these are IDENTICAL to before ----
    B = {n: np.rint((2.0 ** F) * b[n] / sref[n]).astype(np.int64) for n in LAYERS}

    # ---- K_MANT ----
    # L1/RES  : KM_j / 2^KS = 2^(F-QF) * (s_j / s_ref)
    # L2      : KM_j / 2^KS = s_ref(layer1) * (s_j / s_ref(layer2))
    # L3      : KM   / 2^KS = s_ref(layer2)
    KS1 = 17 + QF - F
    KM1 = np.rint((2.0 ** KS1) * (2.0 ** (F - QF)) * (S['layer1'] / sref['layer1'])).astype(np.int64)
    KMR = np.rint((2.0 ** KS1) * (2.0 ** (F - QF)) * (S['residual'] / sref['residual'])).astype(np.int64)
    KS2 = pick_ks(sref['layer1'])
    KM2 = np.rint((2.0 ** KS2) * sref['layer1'] * (S['layer2'] / sref['layer2'])).astype(np.int64)
    KS3 = pick_ks(sref['layer2'])
    KM3 = int(round(sref['layer2'] * 2.0 ** KS3))

    for tag, km in (('L1', KM1), ('L2', KM2)):
        assert km.max() < KM_MAX, f'{tag} K_MANT {km.max()} exceeds BW_K={BW_K}'

    write_hex(f'{args.out}/km_layer1.hex', KM1, BW_K)
    write_hex(f'{args.out}/km_layer2.hex', KM2, BW_K)
    for n in LAYERS:
        write_hex(f'{args.out}/a12_{n}_w_q.hex', Wq[n], WEIGHT_BITS)
        write_hex(f'{args.out}/bias_int_{n}.hex', B[n], 32)

    lam3 = sref['layer3'] / 2.0 ** F
    lamR = sref['residual'] / 2.0 ** F

    print()
    print('=' * 68)
    print('paste into nn_pipelined_int.sv')
    print('=' * 68)
    print(f'    parameter int L1_KM = {int(KM1.max())}, parameter int L1_KS = {KS1},')
    print(f'    parameter int L2_KM = {int(KM2.max())}, parameter int L2_KS = {KS2},')
    print(f'    parameter int L3_KM = {KM3}, parameter int L3_KS = {KS3},')
    print(f'    parameter int RS_KM = {int(KMR.max())}, parameter int RS_KS = {KS1},')
    print()
    print(f"    parameter logic [31:0] LAMBDA3   = 32'h{f32_bits(lam3):08x};")
    print(f"    parameter logic [31:0] LAMBDA_RS = 32'h{f32_bits(lamR):08x};")
    print()
    print('  L1_KM / L2_KM above are only the ROM MAXIMA -- with')
    print('  KM_PER_CHANNEL=1 the layer reads km_layer{1,2}.hex and the scalar')
    print('  is unused. They are printed so you can sanity-check the width.')

    print()
    print('=' * 68)
    print('msh headroom  (msh = K_SHIFT - s_in - s_acc; must stay >= 0)')
    print('=' * 68)
    print('  Measured over 9 test images (natural, gradient, checkerboard,')
    print('  noise, flat black/grey/white, hard edges, very dark):')
    print('     L1 min 8    RES min 6    L2 min 10    L3 min 8')
    print('  All comfortably positive. If you change QF, F or BW_ACT, rerun')
    print('  eval_final_image.py -- it reports these directly.')


if __name__ == '__main__':
    main()