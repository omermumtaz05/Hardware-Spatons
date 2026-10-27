#!/usr/bin/env python3
"""
gen_post_mlp_luts.py -- constants + table for the all-integer post_mlp_int.

Emits:
    data_files/delta_lut.hex   4096 x 12-bit signed,  round(255*(4*sigmoid(z)-2))
    data_files/post_mlp_tb_golden.hex   with --tb
    IDX_M                      printed; paste into nn_pipelined_int_v3

Replaces data_files/sigmoid_lut.hex. The old table held sigmoid(z) as raw fp32
and post_mlp turned it into 4*s-2 at RUNTIME with an fp32_mul and an fp32_add,
per channel, per pixel -- constant folding done in hardware. This bakes it in.

REGENERATE IDX_M AND delta_lut.hex TOGETHER. They both encode INV_STEP; if only
one is rebuilt the index and the table disagree by a scale factor and the image
comes out as a washed-out mush that still passes every handshake assertion.

    python3 gen_post_mlp_luts.py            # write the table, print IDX_M
    python3 gen_post_mlp_luts.py --check    # + bit-compare vs the fp32 model
"""

import argparse
import os
import struct

import numpy as np

# ---- must match nn_pipelined_int_v3 / post_mlp_int ------------------------
LAMBDA3_BITS = 0x3668BE95     # the fp32 constant being folded away
LUT_N        = 4096           # 1 << IDXW
IDX_SH       = 24             # IDX_M is Q0.IDX_SH
ZC           = 23             # zi saturation width in nn_pipelined_int_v3
DLUT_W       = 12             # delta_lut signed width; delta is [-510, +510]
PIX_MAX      = 255
OUT_DIR      = "data_files"

INV_STEP = (LUT_N - 1) / 16.0                      # 255.9375
LAMBDA3  = struct.unpack(">f", struct.pack(">I", LAMBDA3_BITS))[0]
C        = LAMBDA3 * INV_STEP                      # idx slope, per unit of zi
IDX_M    = int(round(C * (1 << IDX_SH)))


def z_of_index(i):
    """Bin centre convention. MUST match however sigmoid_lut.hex was built:
    idx = floor((z+8)*INV_STEP + 0.5)  =>  z = i/INV_STEP - 8."""
    return i / INV_STEP - 8.0


def build_delta_table():
    i = np.arange(LUT_N)
    z = z_of_index(i)
    d = 4.0 / (1.0 + np.exp(-z)) - 2.0

    # Endpoints forced to the SATURATED values. In the fp32 version these came
    # from the sat_lo/sat_hi fp32_le compares (z <= -8 -> s=0, z >= 8 -> s=1).
    # Index clamping already sends every out-of-range z to bin 0 or LUT_N-1, so
    # putting the saturated values IN those bins reproduces the compares exactly
    # instead of approximating them. Without this, saturated pixels sit 0.34 of
    # an 8-bit LSB off -- invisible, but it breaks bit-exact TB comparison.
    d[0]  = -2.0
    d[-1] = +2.0

    # NO FRACTIONAL BITS. center_u8 is an integer, so
    #     rint(center + 255*delta) == center + rint(255*delta)   exactly,
    # i.e. rounding delta on its own at 1-LSB resolution IS rounding once at
    # the end. Fractional bits do not refine that -- they add a SECOND rounding
    # step (table quantisation) that can disagree with the first.
    # Measured over the full 4096 x 256 input space:
    #     DF=0 -> 0 mismatches,  DF=4 -> 3220,  DF=6 -> 1013,  DF=8 -> 466.
    # Every fractional bit is a regression. See the header of post_mlp_int.sv.
    q = np.rint(PIX_MAX * d).astype(np.int64)

    lo, hi = -(1 << (DLUT_W - 1)), (1 << (DLUT_W - 1)) - 1
    assert q.min() >= lo and q.max() <= hi, (
        f"delta_lut overflows signed {DLUT_W}b: [{q.min()}, {q.max()}] "
        f"-- raise DLUT_W"
    )
    return q


def write_hex(path, vals, width_bits):
    mask, nib = (1 << width_bits) - 1, (width_bits + 3) // 4
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for v in vals:
            fh.write(f"{int(v) & mask:0{nib}x}\n")


# ---- bit-accurate model of the RTL ---------------------------------------
def rtl_index(zi):
    """nn_pipelined_int_v3 stages 2 and 3."""
    zi = np.asarray(zi, dtype=np.int64)
    lim = 1 << (ZC - 1)
    zc = np.clip(zi, -lim, lim - 1)               # stage 2: saturate
    ish = (zc * IDX_M) >> IDX_SH                  # numpy >> on int64 is arithmetic
    return np.clip(ish + LUT_N // 2, 0, LUT_N - 1)


def rtl_pixel(zi, center_u8, dlut):
    """post_mlp_int: one ROM read, one add, one saturate."""
    return np.clip(np.asarray(center_u8, dtype=np.int64) + dlut[rtl_index(zi)],
                   0, PIX_MAX)


def float_pixel_from_idx(idx, center_u8):
    """Expected pixel for a GIVEN LUT index, in exact float arithmetic.

    This is the golden for post_mlp_int, whose input IS the index. Do not route
    this through a zi -- see the note above build_tb_vectors().

    Index 0 and LUT_N-1 use the SATURATED delta rather than sigmoid(z_of_index),
    matching both the fp32 sat_lo/sat_hi compares and the forced endpoints in
    build_delta_table().
    """
    idx = np.asarray(idx, dtype=np.int64)
    s = 1.0 / (1.0 + np.exp(-z_of_index(idx)))
    s = np.where(idx == 0, 0.0, np.where(idx == LUT_N - 1, 1.0, s))
    final = np.clip(np.asarray(center_u8, dtype=np.float64) / PIX_MAX + (4 * s - 2), 0.0, 1.0)
    return np.rint(final * PIX_MAX).astype(np.int64)


def float_pixel(zi, center_u8):
    """What the fp32 chain computed end to end, from zi. Used by --check."""
    z = np.asarray(zi, dtype=np.float64) * LAMBDA3
    idx = np.clip(np.floor((z + 8.0) * INV_STEP + 0.5), 0, LUT_N - 1).astype(np.int64)
    return float_pixel_from_idx(idx, center_u8)


def check(dlut):
    rng = np.random.default_rng(0)
    zi = np.concatenate([
        rng.integers(-(1 << 21), 1 << 21, 400_000),        # the useful range
        rng.integers(-(1 << 27), 1 << 27, 100_000),        # saturating tails
        np.array([0, 1, -1, 2306644, -2306644, 1 << 27, -(1 << 27)]),
    ])
    c = rng.integers(0, 256, zi.size)

    a, b = rtl_pixel(zi, c, dlut), float_pixel(zi, c)
    err = np.abs(a - b)

    print(f"\n--- RTL vs fp32 model, {zi.size} random (zi, centre) ---")
    print(f"  exact          : {100*np.mean(err == 0):6.3f} %")
    print(f"  within 1 LSB   : {100*np.mean(err <= 1):6.3f} %")
    print(f"  max error      : {err.max()} LSB")
    print(f"  mean error     : {err.mean():.5f} LSB")

    di = np.abs(rtl_index(zi) - np.clip(
        np.floor((zi.astype(np.float64) * LAMBDA3 + 8.0) * INV_STEP + 0.5),
        0, LUT_N - 1).astype(np.int64))
    print(f"  index exact    : {100*np.mean(di == 0):6.3f} %   (max off-by {di.max()})")

    # Both paths round independently at the output, so a 1 LSB spread is the
    # rounding boundary, not a scale error. Anything above that means IDX_M and
    # delta_lut.hex were generated from different constants.
    assert err.max() <= 1, "MISMATCH > 1 LSB -- IDX_M and delta_lut.hex disagree"
    print("  OK")


# ---- golden vectors for post_mlp_int_tb.sv -------------------------------
# Sweeps every LUT index against 8 interesting centres, with the three channels
# decorrelated so a swapped-channel wiring bug cannot pass.
#
# The expected values come from the FLOAT model -- not from rtl_pixel().
# Generating them from the RTL model would make the TB check the RTL against
# itself, which proves nothing.
#
# THE GOLDEN IS A FUNCTION OF THE INDEX, NOT OF SOME zi THAT MAPS TO IT.
#
# The first version of this went index -> representative zi -> float_pixel(),
# and float_pixel() re-derived the index from that zi. That round trip does NOT
# close: zi_of(i) = rint((i - LUT_N/2)/C) lands on the LEFT EDGE of bin i, and
# floating point puts it a hair below, so it re-derives bin i-1. The TB then
# drove index i while comparing against the value for i-1 and reported 3168
# spurious 1-LSB "failures", all in the steep part of the sigmoid where
# adjacent bins differ by an LSB.
#
# post_mlp_int's input IS the index -- the zi -> index mapping lives in
# nn_pipelined_int and is checked separately by --check. So the golden here
# must start at the index and never mention zi.
#
# Packed 84b per vector so the TB can $readmemh one array:
#   [83:72] idx2  [71:60] idx1  [59:48] idx0
#   [47:40] c2    [39:32] c1    [31:24] c0
#   [23:16] e2    [15:8]  e1    [7:0]   e0
TB_CENTRES = [0, 1, 63, 64, 127, 128, 254, 255]


def build_tb_vectors():
    nc = len(TB_CENTRES)
    idx = np.arange(LUT_N).repeat(nc)
    j = np.tile(np.arange(nc), LUT_N)

    i0 = idx
    i1 = (idx + 1365) % LUT_N
    i2 = (idx + 2731) % LUT_N

    cs = np.array(TB_CENTRES, dtype=np.int64)
    c0 = cs[j]
    c1 = cs[(j + 3) % nc]
    c2 = cs[(j + 5) % nc]

    e0 = float_pixel_from_idx(i0, c0)
    e1 = float_pixel_from_idx(i1, c1)
    e2 = float_pixel_from_idx(i2, c2)

    w = (
        (i2.astype(object) << 72) | (i1.astype(object) << 60) | (i0.astype(object) << 48)
        | (c2.astype(object) << 40) | (c1.astype(object) << 32) | (c0.astype(object) << 24)
        | (e2.astype(object) << 16) | (e1.astype(object) << 8) | e0.astype(object)
    )
    return w, len(w)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tb", action="store_true",
                    help="also emit post_mlp_tb_golden.hex for post_mlp_int_tb.sv")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    print(f"LAMBDA3   = 0x{LAMBDA3_BITS:08x} = {LAMBDA3!r}")
    print(f"INV_STEP  = {INV_STEP}")
    print(f"C         = LAMBDA3*INV_STEP = {C!r}")
    print(f"IDX_SH    = {IDX_SH}")
    print(f"IDX_M     = {IDX_M}          <-- paste into nn_pipelined_int_v3")
    print(f"            rel err {abs(IDX_M/(1<<IDX_SH) - C)/C:.3e}, "
          f"{IDX_M.bit_length()}b, popcount {bin(IDX_M).count('1')} "
          f"(shift-add terms before CSD)")
    print(f"|zi| that saturates the index: {int((LUT_N//2)/C)}  "
          f"(needs {int((LUT_N//2)/C).bit_length()+1}b signed; ZC = {ZC})")

    dlut = build_delta_table()
    path = os.path.join(args.out, "delta_lut.hex")
    write_hex(path, dlut, DLUT_W)
    print(f"\nwrote {path}: {LUT_N} x {DLUT_W}b signed, "
          f"range [{dlut.min()}, {dlut.max()}]")

    if args.tb:
        # EXHAUSTIVE pre-flight: every index x every centre, RTL vs float.
        # 4096*256 = 1M points, the DUT's ENTIRE input space. If this is not
        # 100% exact the TB will fail, and the fault is here, not in the RTL.
        # This is the check that would have caught the zi round-trip bug.
        ii, cc = np.meshgrid(np.arange(LUT_N), np.arange(256), indexing="ij")
        ii, cc = ii.ravel(), cc.ravel()
        rtl = np.clip(cc.astype(np.int64) + dlut[ii], 0, PIX_MAX)
        gold = float_pixel_from_idx(ii, cc)
        bad = np.flatnonzero(rtl != gold)
        print(f"\nexhaustive index x centre: {ii.size} points, {bad.size} mismatches")
        if bad.size:
            for b in bad[:5]:
                print(f"  idx={ii[b]} centre={cc[b]} rtl={rtl[b]} gold={gold[b]}")
            raise SystemExit("golden does not match the DUT's input space")

        w, n = build_tb_vectors()
        tbp = os.path.join(args.out, "post_mlp_tb_golden.hex")
        write_hex(tbp, w, 84)
        print(f"wrote {tbp}: {n} vectors, 84b packed "
              f"(set TB_N = {n} in post_mlp_int_tb.sv)")

    if args.check:
        check(dlut)