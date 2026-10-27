#!/usr/bin/env python3
"""
acc_bound.py -- static requant shifts for nn_pipelined_int_v3, from the
                weight/bias/K_MANT files alone. No simulation. No images.

USAGE
    python3 acc_bound.py [data_files_dir]

WHAT IT COMPUTES

  PART 1 -- RQ_ACC  (the squeeze on the way OUT of a layer)

      acc_j = sum_i q_in[i] * w[j][i]

      requant_int saturates, so |q_in[i]| <= 2047 ALWAYS. And every layer's
      activation input is NON-NEGATIVE (L1/RES take q0 from unsigned 0..255
      features; L2/L3 take the previous layer's ReLU output). So you cannot
      drive an input negative to flip a negative weight into a positive
      contribution:

          |acc_j| <= 2047 * max( sum(w>0) , sum(|w<0|) )

      For L1 and RES this is ACHIEVABLE -- the script emits the 8-bit patch
      that hits it exactly, so you can confirm it in the testbench.

  PART 2 -- RQ_IN  (the squeeze on the way IN to a layer)

      RQ_IN sees the PREVIOUS layer's out_vec, which comes out of APPLY:

          out_vec_j = (racc_j * K_MANT_j) >> msh + B_int_j
          msh       = K_SHIFT - s_in - s_acc          (clamped >= 0 in RTL)

      |racc| <= 2047 by the same saturate. msh is SMALLEST (so out_vec is
      LARGEST) when s_in and s_acc are largest:

          |out_vec_j| <= (2047 * K_MANT_j) >> msh_min + |B_int_j|

      This CASCADES. The chain starts at a provable number: features max out
      at 255 * 65793 = 16,777,215 (24 bits), so xshift <= 13.

          xshift <= 13  ->  L1 out_vec bound  ->  L2 RQ_IN shift
                        ->  L2 out_vec bound  ->  L3 RQ_IN shift

      Each step assumes the worst, so the cascade is looser than Part 1.
      If the result costs you bits you care about, measure instead --
      but this is free, so get it first.
"""

import sys
import os
import math

# ---- from nn_pipelined_int_v3.sv ----
QMAX    = 2047      # (1 << (BW_ACT-1)) - 1
BW_OUT  = 12        # BW_ACT
BW_W    = 10        # BW_W
BW_ACC  = 28        # BW_ACC
BW_Y    = 28        # BW_Y
FEAT_W  = 8         # FEAT_W

# (name, wfile, neurons, inputs, raw_feature_input,
#  K_SHIFT, K_MANT_scalar, km_file_or_None, bias_file, feeds)
LAYERS = [
    ("layer1",   "a12_layer1_w_q.hex",   48, 29, True,
     31, 131072, "km_layer1.hex", "bias_int_layer1.hex", "layer2"),
    ("layer2",   "a12_layer2_w_q.hex",   48, 48, False,
     27, 211738, "km_layer2.hex", "bias_int_layer2.hex", "layer3"),
    ("layer3",   "a12_layer3_w_q.hex",    3, 48, False,
     24, 246485, None, "bias_int_layer3.hex", None),
    ("residual", "a12_residual_w_q.hex",  3, 29, True,
     31, 131072, None, "bias_int_residual.hex", None),
]


def read_hex(path, bits):
    """$readmemh-compatible reader. Returns signed ints."""
    toks = []
    with open(path) as f:
        for line in f:
            line = line.split("//")[0].split("#")[0].strip()
            if line:
                toks += line.split()
    out = []
    for t in toks:
        v = int(t, 16)
        if v >= (1 << (bits - 1)):
            v -= (1 << bits)
        out.append(v)
    return out


def shift_for(bound):
    """Mirrors requant_int NORM:  s = max(0, msb(v) - (BW_OUT-2))"""
    b = int(abs(bound))
    if b <= 0:
        return 0
    return max(0, (b.bit_length() - 1) - (BW_OUT - 2))


# =====================================================================
# PART 1 -- RQ_ACC
# =====================================================================

def acc_bound(name, path, n_neurons, n_inputs, is_input_layer):
    flat = read_hex(path, BW_W)
    need = n_neurons * n_inputs
    if len(flat) < need:
        raise SystemExit(f"{path}: got {len(flat)} weights, need {need}")
    rows = [flat[k * n_inputs:(k + 1) * n_inputs] for k in range(n_neurons)]

    worst_k = worst_l1 = worst_1s = 0
    est = 0.0
    for k, row in enumerate(rows):
        pos = sum(w for w in row if w > 0)
        neg = -sum(w for w in row if w < 0)
        one = max(pos, neg)
        if one > worst_1s:
            worst_1s, worst_k = one, k
        worst_l1 = max(worst_l1, pos + neg)
        sw = sum(row)
        l2 = math.sqrt(sum(w * w for w in row))
        est = max(est, abs(1023.5 * sw) + 4 * 591.0 * l2)

    b_l1 = QMAX * worst_l1
    b_1s = QMAX * worst_1s
    s_acc = shift_for(b_1s)

    print(f"\n=== {name}  ({n_neurons} x {n_inputs}) -- RQ_ACC ===")
    print(f"  worst neuron                  : {worst_k}")
    print(f"  bound, both signs reachable   : {b_l1:>13,}   s = {shift_for(b_l1)}")
    print(f"  bound, inputs >= 0  (USE THIS): {b_1s:>13,}   s = {s_acc}"
          f"   <- saves {shift_for(b_l1) - s_acc} bit(s)")
    print(f"  heuristic (mean + 4 sd)       : {est:>13,.0f}   s = {shift_for(int(est))}")
    print(f"  slack vs heuristic            : {s_acc - shift_for(int(est))} bit(s)")

    cap = (1 << (BW_ACC - 1)) - 1
    if b_1s > cap:
        print(f"  !! bound exceeds BW_ACC={BW_ACC} capacity {cap:,}")

    patch = None
    if is_input_layer:
        row = rows[worst_k]
        pos = sum(w for w in row if w > 0)
        neg = -sum(w for w in row if w < 0)
        sign = 1 if pos >= neg else -1
        patch = [255 if (w * sign) > 0 else 0 for w in row]
        print(f"  worst-case 8-bit feature vector (neuron {worst_k}):")
        print("    " + " ".join(f"{v:3d}" for v in patch))

    return s_acc, patch


# =====================================================================
# PART 2 -- RQ_IN, via a bound on out_vec
# =====================================================================

def out_vec_bound(name, d, n_neurons, k_shift, km_scalar, km_file,
                  bias_file, s_in_max, s_acc):
    # K_MANT: per-neuron ROM if present, else the scalar
    if km_file and os.path.exists(os.path.join(d, km_file)):
        km = read_hex(os.path.join(d, km_file), 19)
        km_max, km_src = max(abs(v) for v in km[:n_neurons]), km_file
    else:
        km_max, km_src = km_scalar, "scalar K_MANT"

    bp = os.path.join(d, bias_file)
    if os.path.exists(bp):
        bias = read_hex(bp, 32)
        b_max, b_src = max(abs(v) for v in bias[:n_neurons]), bias_file
    else:
        b_max, b_src = 0, "MISSING -- assuming 0, result is OPTIMISTIC"

    msh_min = max(0, k_shift - s_in_max - s_acc)     # RTL clamps msh at 0
    prod    = QMAX * km_max
    bound   = (prod >> msh_min) + b_max
    cap     = (1 << (BW_Y - 1)) - 1
    bound_c = min(bound, cap)                        # APPLY saturates to BW_Y

    print(f"\n--- {name} out_vec bound  (sets the NEXT layer's RQ_IN) ---")
    print(f"  K_MANT max  : {km_max:>13,}   ({km_src})")
    print(f"  |bias| max  : {b_max:>13,}   ({b_src})")
    print(f"  msh_min     : {k_shift} - {s_in_max} - {s_acc} = {msh_min}")
    print(f"  bound       : (2047 x {km_max:,}) >> {msh_min} + {b_max:,}")
    print(f"              : {bound:>13,}")
    if bound > cap:
        print(f"  clamped by BW_Y={BW_Y} to {bound_c:,}  (APPLY saturates)")
    print(f"  => next layer RQ_IN  s_in = {shift_for(bound_c)}")
    return shift_for(bound_c)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "data_files"
    print(f"reading from: {os.path.abspath(d)}")
    print(f"QMAX={QMAX}  BW_ACT={BW_OUT}  BW_W={BW_W}  BW_ACC={BW_ACC}  BW_Y={BW_Y}")

    # ---------- PART 1 ----------
    print("\n" + "=" * 66)
    print("PART 1 -- RQ_ACC  (provable from weights alone)")
    print("=" * 66)

    s_acc, patches = {}, []
    meta = {}
    for (name, wf, nn, ni, isin, ks, kms, kmf, bf, feeds) in LAYERS:
        p = os.path.join(d, wf)
        meta[name] = (nn, ks, kms, kmf, bf, feeds)
        if not os.path.exists(p):
            print(f"\n!! missing {p} -- skipping {name}")
            continue
        s, patch = acc_bound(name, p, nn, ni, isin)
        s_acc[name] = s
        if patch:
            patches.append((name, patch))

    # ---------- PART 2 ----------
    print("\n" + "=" * 66)
    print("PART 2 -- RQ_IN  (cascade: xshift -> L1 out -> L2 in -> ...)")
    print("=" * 66)

    # chain root: features are 8-bit, f_fx = {v,v,v} = v * 65793
    f_max = ((1 << FEAT_W) - 1) * 0x010101
    xshift_max = shift_for(f_max)
    print(f"\n  root: max f_fx = 255 x 65793 = {f_max:,}"
          f"  ({f_max.bit_length()} bits)  ->  xshift <= {xshift_max}")
    print(f"  (L1 and RES take q0 directly -- they have NO RQ_IN)")

    s_in = {"layer1": xshift_max, "residual": xshift_max}

    for src in ("layer1", "layer2"):
        if src not in s_acc:
            print(f"\n!! {src} bound unavailable -- cannot continue cascade")
            break
        nn, ks, kms, kmf, bf, feeds = meta[src]
        s = out_vec_bound(src, d, nn, ks, kms, kmf, bf, s_in[src], s_acc[src])
        if feeds:
            s_in[feeds] = s

    # ---------- SUMMARY ----------
    print("\n" + "=" * 66)
    print("SUMMARY -- static shifts")
    print("=" * 66)
    print(f"\n  {'pass':<18}{'shift':<8}{'how'}")
    print("  " + "-" * 56)
    print(f"  {'u_rq0':<18}{'dynamic':<8}keep -- off the critical path")
    for n in ("layer1", "layer2", "layer3", "residual"):
        if n in s_acc:
            print(f"  {n + ' RQ_ACC':<18}{s_acc[n]:<8}PROVEN from weights")
    for n in ("layer2", "layer3"):
        if n in s_in:
            print(f"  {n + ' RQ_IN':<18}{s_in[n]:<8}bounded via cascade")

    print("\n  fold RQ_ACC into K_SHIFT (nn_pipelined_int_v3.sv:151-154):")
    for n, p in (("layer1", "L1_KS"), ("layer2", "L2_KS"),
                 ("layer3", "L3_KS"), ("residual", "RS_KS")):
        if n in s_acc:
            old = meta[n][1]
            print(f"    {p}: {old} -> {old - s_acc[n]}"
                  f"      ({old} - {s_acc[n]})")

    if patches:
        out = "worst_case_patches.txt"
        with open(out, "w") as f:
            for name, patch in patches:
                f.write(f"// worst-case feature vector for {name}\n")
                f.write(" ".join(f"{v:02x}" for v in patch) + "\n")
        print(f"\n  wrote {out} -- run these through the TB to confirm PART 1")

    print("""
  PART 1 is a guarantee: no image can break it.
  PART 2 is a bound, but a loose one -- it assumes msh hits its minimum
  and racc hits 2047 at the same instant. If the RQ_IN shifts cost you
  more than a bit or two, probe max_reg in the TB and use the real number.
""")


if __name__ == "__main__":
    main()