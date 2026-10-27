#!/usr/bin/env python3
"""
photo_compare.py — 4-panel visual for the photo pipeline at 256x256.
Reads 96-bit-packed hex (24 chars/line, [95:64]=B [63:32]=G [31:0]=R).

  panel 1: input (geo-warped capture)
  panel 2: python golden (expected corrected)
  panel 3: DUT output (RTL)
  panel 4: |golden - DUT| difference

Run: python helper/photo_compare.py
"""

import struct
import numpy as np
import matplotlib.pyplot as plt

W = H = 256

def hex_f32(h): return struct.unpack('>f', bytes.fromhex(h))[0]

def load_packed(path):
    with open(path) as f:
        lines = [ln.strip() for ln in f
                 if ln.strip() and not ln.strip().startswith("//")]
    arr = np.zeros((H, W, 3), dtype=np.float32)
    for idx, line in enumerate(lines):
        line = line.zfill(24)
        B = hex_f32(line[0:8]); G = hex_f32(line[8:16]); R = hex_f32(line[16:24])
        y, x = idx // W, idx % W
        arr[y, x] = [R, G, B]
    return np.clip(arr, 0, 1)

def load_flat3(path):
    vals = [hex_f32(w) for ln in open(path)
            for w in ln.split() if w and not ln.startswith("//")]
    arr = np.array(vals, np.float32).reshape(H, W, 3)   # R,G,B order
    return np.clip(arr, 0, 1)

inp  = load_packed("data_files/capture_array.hex")   # capture stays packed
gold = load_flat3("data_files/photo_golden_q.hex")
dut  = load_flat3("data_files/photo_chain_dut_int.hex")

diff = np.abs(gold - dut).max(axis=2)
print(f"golden vs DUT: max={diff.max():.3e} mean={diff.mean():.3e}")
print(f"pixels diff > 1e-3: {int((diff > 1e-3).sum())} / {W*H}")
print(f"pixels diff > 5e-2: {int((diff > 5e-2).sum())} / {W*H}")   # ADD
d = np.abs(gold - dut)                        # full per-channel abs error
nz = d[d > 0]                                 # nonzero errors only (mode/min are meaningful there)
print(f"max  = {d.max():.3e}")
print(f"mean = {d.mean():.3e}")
print(f"median = {np.median(d):.3e}")
print(f"std  = {d.std():.3e}")
print(f"p99  = {np.percentile(d, 99):.3e}")   # 99th percentile — 'typical worst'
print(f"p99.9= {np.percentile(d, 99.9):.3e}")
print(f"nonzero pixels: {nz.size} / {d.size} ({100*nz.size/d.size:.2f}%)")
print(f"RMSE = {np.sqrt((d**2).mean()):.3e}")

diff_vis = np.where(diff > 0.05, diff, 0)        

print("--- negative control ---")
bad = np.roll(dut, 1, axis=1)
print(f"shifted-by-1 max: {np.abs(gold-bad).max():.3e}")

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(inp);  ax[0].set_title("input (warped)"); ax[0].axis('off')
ax[1].imshow(gold); ax[1].set_title("python golden");  ax[1].axis('off')
ax[2].imshow(dut);  ax[2].set_title("DUT output");     ax[2].axis('off')
im = ax[3].imshow(d.max(axis=2), cmap='hot', vmin=0, vmax=max(d.max(), 1e-6))
ax[3].set_title(f"|golden - DUT| (vmax={d.max():.1e})")
fig.colorbar(im, ax=ax[3], fraction=0.046)
plt.tight_layout()
plt.savefig("data_files/photo_compare.png", dpi=110, bbox_inches='tight')
print("saved data_files/photo_compare.png")
plt.show()