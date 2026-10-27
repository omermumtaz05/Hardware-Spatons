#!/usr/bin/env python3
"""full_sys_compare.py — 5-panel visual for the full pipeline at 256x256.

  panel 1: source image.hex (raw input, uint8)
  panel 2: python warp (geo stage done in python)
  panel 3: full golden (python warp -> features -> quant MLP)
  panel 4: DUT output (RTL, full system)
  panel 5: |golden - DUT|

Run: python helper/full_sys_compare.py
"""

import struct
import numpy as np
import matplotlib.pyplot as plt

W = H = 256

def hex_f32(h): return struct.unpack('>f', bytes.fromhex(h))[0]

def load_flat3(path):
    vals = [hex_f32(w) for ln in open(path)
            for w in ln.split() if w and not ln.startswith("//")]
    arr = np.array(vals, np.float32).reshape(H, W, 3)   # R,G,B order
    return np.clip(arr, 0, 1)

def load_u8(path):
    """image.hex — one uint8 hex value per line, N*3 lines, R,G,B order."""
    vals = [int(ln.strip(), 16) for ln in open(path)
            if ln.strip() and not ln.strip().startswith("//")]
    return (np.array(vals, np.uint8).reshape(H, W, 3).astype(np.float32) / 255.0)

src  = load_u8("data_files/image.hex")
warp = load_flat3("data_files/full_warped_q.hex")
gold = load_flat3("data_files/full_golden_q.hex")
dut  = load_flat3("data_files/full_sys_dut_int.hex")

d    = np.abs(gold - dut)            # per-channel abs error
dmax = d.max(axis=2)                 # per-pixel worst channel
nz   = d[d > 0]

print(f"golden vs DUT: max={d.max():.3e} mean={d.mean():.3e}")
print(f"pixels diff > 1e-3: {int((dmax > 1e-3).sum())} / {W*H}")
print(f"pixels diff > 5e-2: {int((dmax > 5e-2).sum())} / {W*H}")
print(f"median = {np.median(d):.3e}")
print(f"std    = {d.std():.3e}")
print(f"p99    = {np.percentile(d, 99):.3e}")
print(f"p99.9  = {np.percentile(d, 99.9):.3e}")
print(f"nonzero: {nz.size} / {d.size} ({100*nz.size/d.size:.2f}%)")
print(f"RMSE   = {np.sqrt((d**2).mean()):.3e}")

# worst pixel, useful when something is structurally wrong
w  = int(dmax.argmax()); wy, wx = w // W, w % W
print(f"worst pixel (x={wx}, y={wy}): gold={gold[wy,wx]} dut={dut[wy,wx]}")

# autoscale the error map so it stays readable whether error is 1e-7 or 1.0
vmax = max(dmax.max(), 1e-6)

fig, ax = plt.subplots(1, 5, figsize=(24, 5))
ax[0].imshow(src);  ax[0].set_title("source (image.hex)")
ax[1].imshow(warp); ax[1].set_title("python warp (geo ref)")
ax[2].imshow(gold); ax[2].set_title("full golden (independent)")
ax[3].imshow(dut);  ax[3].set_title("DUT output (RTL)")
im = ax[4].imshow(dmax, cmap='inferno', vmin=0, vmax=vmax)
ax[4].set_title(f"|golden - DUT| (vmax={vmax:.2e})")
for a in ax: a.axis('off')
fig.colorbar(im, ax=ax[4], fraction=0.046)
plt.tight_layout()
plt.savefig("data_files/full_compare.png", dpi=110, bbox_inches='tight')
print("saved data_files/full_compare.png")
plt.show()