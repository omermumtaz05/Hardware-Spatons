"""
Compare DUT geometric-stage output (uint8) against two references:

  1. IDEAL   -- fp32 bilinear warp at fp32 coords.  Measures total
                quantisation cost of the int pipeline. Never expected to
                match exactly.
  2. BITEXACT-- numpy model of the actual Q8.8 integer datapath
                (csf_int -> ffn_int -> bilinear_interp_int).  This SHOULD
                match the DUT on every pixel.  Any mismatch is an RTL bug.

Capture file is now 3 bytes/line, "%02h%02h%02h" = p2 p1 p0  (ch2 ch1 ch0).
"""

import numpy as np
import struct
import os
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
W, H = 256, 256
N = W * H

LUT_FP32     = "data_files/lut_fp32.hex"      # fp32 {y,x}, 16 hex chars/line
LUT_QUANT    = "data_files/lut_quant.hex"     # Q8.8 {y,x},  8 hex chars/line (optional)
IMAGE_FILE   = "data_files/image.hex"
CAPTURE_FILE = "data_files/capture_array_uint.hex"

D_BITS = 8                    # BIT_WIDTH_D
ONE    = 1 << D_BITS          # 256
SHIFT  = 2 * D_BITS           # 16
HALF   = 1 << (SHIFT - 1)     # 32768

BAD_LSB = 2                   # what counts as "bad" vs the ideal ref


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def hex_to_f32(h):
    """8 hex chars -> float32.  struct needs exactly 4 bytes."""
    return struct.unpack('>f', bytes.fromhex(h.zfill(8)))[0]


def read_lines(path, skip_comments=False):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if skip_comments and line.startswith("//"):
                continue
            out.append(line)
    return out


def validate_capture(path, width_chars):
    """Report malformed lines before we try to parse for real."""
    bad = 0
    with open(path) as fh:
        for i, line in enumerate(fh):
            v = line.strip()
            if not v:
                print(f"  line {i}: BLANK")
                bad += 1
                continue
            try:
                bytes.fromhex(v.zfill(width_chars))
            except ValueError:
                print(f"  line {i}: BAD -> {repr(v)}")
                bad += 1
    if bad:
        print(f"  {bad} malformed line(s)")
    return bad


# ----------------------------------------------------------------------
# load source image (uint8 AND float views -- the int model needs uint8)
# ----------------------------------------------------------------------
print("loading image...")
img_lines = read_lines(IMAGE_FILE)
assert len(img_lines) == N * 3, f"image.hex: {len(img_lines)} lines, expected {N*3}"

img_u8 = (np.array([int(v, 16) for v in img_lines], dtype=np.uint8)
          .reshape(H, W, 3))
img    = img_u8.astype(np.float64) / 255.0


# ----------------------------------------------------------------------
# load fp32 LUT  (64-bit {y' high, x' low}, pixel units)
# ----------------------------------------------------------------------
print("loading fp32 LUT...")
lut_lines = read_lines(LUT_FP32)
assert len(lut_lines) == N

lut_x = np.zeros((H, W))
lut_y = np.zeros((H, W))
for idx, line in enumerate(lut_lines):
    h = line.zfill(16)
    y, x = idx // W, idx % W
    lut_y[y, x] = hex_to_f32(h[0:8])
    lut_x[y, x] = hex_to_f32(h[8:16])


# ----------------------------------------------------------------------
# load / derive the Q8.8 LUT the DUT actually uses
# ----------------------------------------------------------------------
if os.path.exists(LUT_QUANT):
    print("loading Q8.8 LUT (lut_quant.hex)...")
    q_lines = read_lines(LUT_QUANT)
    assert len(q_lines) == N
    xq = np.zeros((H, W), dtype=np.int64)
    yq = np.zeros((H, W), dtype=np.int64)
    for idx, line in enumerate(q_lines):
        v = int(line.zfill(8), 16)
        y, x = idx // W, idx % W
        yq[y, x] = (v >> 16) & 0xFFFF
        xq[y, x] =  v        & 0xFFFF
else:
    print("lut_quant.hex not found -- deriving Q8.8 from fp32 LUT (truncation)")
    xq = np.clip(np.floor(lut_x * ONE), 0, 0xFFFF).astype(np.int64)
    yq = np.clip(np.floor(lut_y * ONE), 0, 0xFFFF).astype(np.int64)


# ----------------------------------------------------------------------
# reference 1: ideal fp32 bilinear warp
# ----------------------------------------------------------------------
print("rendering ideal warp...")
x0 = np.floor(lut_x).astype(int)
y0 = np.floor(lut_y).astype(int)
dxf = (lut_x - x0)[..., None]
dyf = (lut_y - y0)[..., None]

x0c = np.clip(x0,     0, W - 1); x1c = np.clip(x0 + 1, 0, W - 1)
y0c = np.clip(y0,     0, H - 1); y1c = np.clip(y0 + 1, 0, H - 1)

ref = np.clip((1 - dxf) * (1 - dyf) * img[y0c, x0c] +
              (1 - dxf) * dyf       * img[y1c, x0c] +
              dxf       * (1 - dyf) * img[y0c, x1c] +
              dxf       * dyf       * img[y1c, x1c], 0, 1)

# hardware rounds half-UP: (acc + HALF) >> SHIFT.  np.rint is half-to-EVEN,
# which disagrees on exact .5 and manufactures phantom 1-LSB errors.
ref_u8 = np.floor(ref * 255.0 + 0.5).astype(np.int64)


# ----------------------------------------------------------------------
# reference 2: bit-exact model of the Q8.8 integer datapath
#   csf_int : x_int = xq >> 8, dx = xq & 0xFF
#   ffn_int : neighbour addresses, clamped to the image  (ASSUMPTION --
#             confirm against fetch_four_nearest_int.sv; if it wraps instead
#             of clamping, change the np.clip calls below to % W / % H)
#   bilinear: w = (ONE-dx)(ONE-dy) etc, acc = sum(w*p), (acc+HALF)>>SHIFT
# ----------------------------------------------------------------------
print("rendering bit-exact int model...")
xi = (xq >> D_BITS).astype(np.int64)
yi = (yq >> D_BITS).astype(np.int64)
dxi = (xq & (ONE - 1)).astype(np.int64)
dyi = (yq & (ONE - 1)).astype(np.int64)

xa = np.clip(xi,     0, W - 1); xb = np.clip(xi + 1, 0, W - 1)
ya = np.clip(yi,     0, H - 1); yb = np.clip(yi + 1, 0, H - 1)

mdx = ONE - dxi
mdy = ONE - dyi

w00 = (mdx * mdy)[..., None]
w10 = (dxi * mdy)[..., None]
w01 = (mdx * dyi)[..., None]
w11 = (dxi * dyi)[..., None]

P00 = img_u8[ya, xa].astype(np.int64)
P01 = img_u8[yb, xa].astype(np.int64)
P10 = img_u8[ya, xb].astype(np.int64)
P11 = img_u8[yb, xb].astype(np.int64)

acc = w00 * P00 + w10 * P10 + w01 * P01 + w11 * P11
bitexact_u8 = ((acc + HALF) >> SHIFT).astype(np.int64)
assert bitexact_u8.max() <= 255, f"int model overflowed: {bitexact_u8.max()}"


# ----------------------------------------------------------------------
# load DUT capture (uint8, 3 bytes per line)
# ----------------------------------------------------------------------
print("loading DUT capture...")
validate_capture(CAPTURE_FILE, 6)

cap = read_lines(CAPTURE_FILE, skip_comments=True)
assert len(cap) == N, f"capture: {len(cap)} lines, expected {N}"

dut_u8 = np.zeros((H, W, 3), dtype=np.int64)
for idx, line in enumerate(cap):
    line = line.strip().zfill(6)
    p2 = int(line[0:2], 16)
    p1 = int(line[2:4], 16)
    p0 = int(line[4:6], 16)
    y, x = idx // W, idx % W
    dut_u8[y, x] = (p0, p1, p2)

dut = dut_u8.astype(np.float64) / 255.0


# ----------------------------------------------------------------------
# stats
# ----------------------------------------------------------------------
def report(name, a, b):
    e    = np.abs(a - b)
    emax = e.max(axis=2)
    print(f"\n=== {name} (LSB units, 1 LSB = 1/255) ===")
    print(f" max  err : {e.max()} LSB")
    print(f" mean err : {e.mean():.4f} LSB")
    for t in (0, 1, 2, 4, 8):
        n = int((emax > t).sum())
        print(f" pixels err > {t} LSB : {n:6d} / {N}  ({100.0*n/N:5.2f}%)")
    return emax


# the one that matters: model vs DUT should be identical
e_bit = report("BITEXACT model vs DUT", bitexact_u8, dut_u8)
if e_bit.max() == 0:
    print(" ** BIT-EXACT MATCH -- RTL is correct **")
else:
    n = int((e_bit > 0).sum())
    print(f" *** {n} pixels differ from the int model -- RTL BUG ***")
    w = int(e_bit.argmax()); wy, wx = w // W, w % W
    print(f"  worst (x={wx}, y={wy}): model={bitexact_u8[wy,wx]} dut={dut_u8[wy,wx]}")
    print(f"    xq=0x{xq[wy,wx]:04x} yq=0x{yq[wy,wx]:04x} "
          f"-> xi={xi[wy,wx]} yi={yi[wy,wx]} dx={dxi[wy,wx]} dy={dyi[wy,wx]}")

# quality number: how much the int pipeline costs vs the ideal warp
e_ideal = report("IDEAL fp32 warp vs DUT", ref_u8, dut_u8)


# ----------------------------------------------------------------------
# where do the ideal-ref outliers land? (edge effects vs real error)
# ----------------------------------------------------------------------
bad = np.argwhere(e_ideal > BAD_LSB)
print(f"\n{len(bad)} pixels exceed {BAD_LSB} LSB vs ideal")

if len(bad):
    sx = lut_x[bad[:, 0], bad[:, 1]]
    sy = lut_y[bad[:, 0], bad[:, 1]]
    on_edge = (sx >= W - 2) | (sy >= H - 2) | (sx < 1) | (sy < 1)
    print(f"  {int(on_edge.sum())}/{len(bad)} map to a source edge (clamp artefacts)")

    interior = bad[~on_edge]
    if len(interior):
        y, x = interior[0]
        print(f"\n--- interior outlier probe: out({x},{y}) ---")
        print(f"  fp32 src ({lut_x[y,x]:.4f}, {lut_y[y,x]:.4f})")
        print(f"  Q8.8 src ({xq[y,x]/ONE:.4f}, {yq[y,x]/ONE:.4f})"
              f"  xi={xi[y,x]} yi={yi[y,x]} dx={dxi[y,x]} dy={dyi[y,x]}")
        print(f"  P00={img_u8[ya[y,x], xa[y,x]]}  P10={img_u8[ya[y,x], xb[y,x]]}")
        print(f"  P01={img_u8[yb[y,x], xa[y,x]]}  P11={img_u8[yb[y,x], xb[y,x]]}")
        print(f"  ideal={ref_u8[y,x]}  model={bitexact_u8[y,x]}  dut={dut_u8[y,x]}")
    else:
        print("  all outliers are edge pixels -- nothing interior to probe")


# ----------------------------------------------------------------------
# panels
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 5, figsize=(22, 5))

ax[0].imshow(img);                       ax[0].set_title("source")
ax[1].imshow(ref);                       ax[1].set_title("ideal warp (fp32 ref)")
ax[2].imshow(dut);                       ax[2].set_title("DUT (uint8)")

im3 = ax[3].imshow(e_ideal, cmap='hot'); ax[3].set_title("|ideal - DUT|  (LSB)")
fig.colorbar(im3, ax=ax[3], fraction=0.046)

im4 = ax[4].imshow(e_bit, cmap='hot', vmin=0, vmax=max(1, e_bit.max()))
ax[4].set_title("|int model - DUT|  (should be 0)")
fig.colorbar(im4, ax=ax[4], fraction=0.046)

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.savefig("data_files/compare.png", dpi=110, bbox_inches='tight')
print("\nsaved data_files/compare.png")
plt.show()