#!/usr/bin/env python3
"""full_sys_compare.py -- 5-panel visual for the full pipeline at 256x256.

  panel 1: source image.hex (raw input, uint8)
  panel 2: python warp (geo stage done in python)
  panel 3: full golden (python warp -> features -> quant MLP)
  panel 4: DUT output (RTL, full system)
  panel 5: |golden - DUT| in 8-bit LSBs

Run: python helper/full_sys_compare.py

===========================================================================
WHAT CHANGED FOR THE ALL-INTEGER BUILD

The DUT dump is now UINT8, one 2-hex-char value per line:
    ff        <- was 3f800000
Not fp32 words. load_hex3() sniffs the token width so the same loader reads
both the old fp32 goldens and the new uint8 dump.

ERROR IS REPORTED IN 8-BIT LSBs, NOT FLOATS. The DUT can only emit integers
0..255 now, so a float error of 3.9e-3 is not a small error -- it is exactly
one LSB, the smallest error that exists. Reporting it as 3.9e-3 makes a 1-LSB
difference look 250x better than a 1-LSB difference, which is the same number.
The golden is quantised to uint8 the same way the hardware does
(round(x*255), clipped) before differencing, so both sides live on the same
grid and the difference is an integer count of LSBs.

Reference points to compare against, from the design's own characterisation:
    per-tensor weights      mean 1.344   p99 9.107   max 22.161   >2LSB 26.35%
    PER-CHANNEL L1+L2       mean 0.542   p99 3.403   max 11.500   >2LSB  6.24%
The fp32 -> integer post_mlp change was measured at 100.000% bit-exact against
the float model, so THESE NUMBERS SHOULD NOT MOVE. If mean jumps to ~1.3 the
per-channel K_MANT ROMs are not loading; if it explodes, see the triage table
printed at the bottom.
"""

import struct
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = H = 256


def load_hex3(path):
    """Load an N*3-line hex dump as float [0,1], HxWx3, R,G,B order.

    Sniffs the token width: 8 chars -> fp32 bit pattern, 2 chars -> uint8.
    """
    toks = [w for ln in open(path)
            for w in ln.split()
            if w and not w.startswith("//")]
    if not toks:
        sys.exit(f"{path}: empty")

    n = len(toks[0])
    if n == 8:
        vals = [struct.unpack('>f', bytes.fromhex(t))[0] for t in toks]
        arr = np.array(vals, np.float32)
    elif n <= 2:
        arr = np.array([int(t, 16) for t in toks], np.float32) / 255.0
    else:
        sys.exit(f"{path}: unexpected token width {n} ('{toks[0]}')")

    if arr.size != H * W * 3:
        sys.exit(f"{path}: got {arr.size} values, expected {H*W*3} "
                 f"({arr.size/3:.0f} pixels vs {H*W})")
    return np.clip(arr.reshape(H, W, 3), 0, 1)


def to_u8(x):
    """Quantise exactly the way the hardware does, so the diff is in LSBs."""
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.int16)


src  = load_hex3("data_files/image.hex")
warp = load_hex3("data_files/full_warped_q.hex")
gold = load_hex3("data_files/full_golden_q.hex")
dut  = load_hex3("data_files/full_sys_st.hex")

g8, d8 = to_u8(gold), to_u8(dut)
err    = np.abs(g8 - d8)             # per-channel, in 8-bit LSBs
emax   = err.max(axis=2)             # per-pixel worst channel

print(f"--- golden vs DUT, in 8-bit LSBs ({err.size} channel samples) ---")
print(f"  exact (0 LSB) : {100*np.mean(err == 0):6.2f} %")
print(f"  within 1 LSB  : {100*np.mean(err <= 1):6.2f} %")
print(f"  within 2 LSB  : {100*np.mean(err <= 2):6.2f} %")
print(f"  mean          : {err.mean():.3f}")
print(f"  p99           : {np.percentile(err, 99):.3f}")
print(f"  p99.9         : {np.percentile(err, 99.9):.3f}")
print(f"  max           : {err.max()}")
print(f"  > 2 LSB       : {100*np.mean(err > 2):6.2f} %   (baseline 6.24%)")
print(f"  RMSE          : {np.sqrt((err.astype(float)**2).mean()):.3f}")

wi = int(emax.argmax()); wy, wx = wi // W, wi % W
print(f"\n  worst pixel (x={wx}, y={wy}): gold={g8[wy,wx]} dut={d8[wy,wx]}")

# --- structural checks: these separate "slightly wrong" from "wired wrong" ---
print("\n--- structural ---")
flat = d8.reshape(-1, 3)
print(f"  DUT unique values per channel: "
      f"R={len(np.unique(flat[:,0]))} G={len(np.unique(flat[:,1]))} "
      f"B={len(np.unique(flat[:,2]))}")
print(f"  DUT saturated at 0: {100*np.mean(d8 == 0):5.2f} %   "
      f"at 255: {100*np.mean(d8 == 255):5.2f} %")
print(f"  golden saturated at 0: {100*np.mean(g8 == 0):5.2f} %   "
      f"at 255: {100*np.mean(g8 == 255):5.2f} %")

# A whole-image shift means the side FIFO and the MLP output are off by N
# pixels -- the error map looks like an edge-detected copy of the image, and
# the per-pixel stats look terrible even though every value is individually
# correct. Test it directly rather than eyeballing the picture.
gf, df = g8.reshape(-1, 3), d8.reshape(-1, 3)
best = (0, np.abs(gf - df).mean())
for s in range(1, 9):
    m = np.abs(gf[s:] - df[:-s]).mean()
    if m < best[1]:
        best = (s, m)
if best[0]:
    print(f"  *** DUT appears SHIFTED BY {best[0]} PIXELS "
          f"(mean err {best[1]:.3f} vs {np.abs(gf-df).mean():.3f} unshifted)")
    print(f"  *** that is side-FIFO / z_idx alignment, not arithmetic")
else:
    print("  alignment: no whole-image shift detected")

vmax = max(int(emax.max()), 1)

fig, ax = plt.subplots(1, 5, figsize=(24, 5))
ax[0].imshow(src);  ax[0].set_title("source (image.hex)")
ax[1].imshow(warp); ax[1].set_title("python warp (geo ref)")
ax[2].imshow(gold); ax[2].set_title("full golden (independent)")
ax[3].imshow(dut);  ax[3].set_title("DUT output (RTL, all-integer)")
im = ax[4].imshow(emax, cmap='inferno', vmin=0, vmax=vmax)
ax[4].set_title(f"|golden - DUT|, LSBs (vmax={vmax})")
for a in ax:
    a.axis('off')
fig.colorbar(im, ax=ax[4], fraction=0.046)
plt.tight_layout()
plt.savefig("data_files/full_compare.png", dpi=110, bbox_inches='tight')
print("\nsaved data_files/full_compare.png")
plt.show()