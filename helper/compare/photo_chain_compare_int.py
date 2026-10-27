#!/usr/bin/env python3
import struct
import numpy as np
import matplotlib.pyplot as plt

W = H = 256

def hex_f32(h): return struct.unpack('>f', bytes.fromhex(h))[0]

def load_packed(path):
    lines = [ln.strip() for ln in open(path)
             if ln.strip() and not ln.strip().startswith("//")]
    arr = np.zeros((H, W, 3), np.float32)
    for idx, line in enumerate(lines):
        line = line.zfill(24)
        arr[idx // W, idx % W] = [hex_f32(line[16:24]),   # R
                                  hex_f32(line[8:16]),    # G
                                  hex_f32(line[0:8])]     # B
    return np.clip(arr, 0, 1)

def load_flat3(path):
    vals = [hex_f32(w) for ln in open(path)
            for w in ln.split() if w and not ln.startswith("//")]
    return np.clip(np.array(vals, np.float32).reshape(H, W, 3), 0, 1)

inp  = load_packed("data_files/capture_array.hex")
prev = load_flat3("data_files/photo_chain_dut_int.hex")
dut  = load_flat3("data_files/photo_chain_dut_int_v2.hex")

d = np.abs(prev - dut)
print(f"max  = {d.max():.3e}  ({d.max()*255:.2f} LSB)")
print(f"mean = {d.mean():.3e}  ({d.mean()*255:.2f} LSB)")
print(f"p99  = {np.percentile(d,99):.3e}  ({np.percentile(d,99)*255:.2f} LSB)")
print(f">2 LSB: {100*(d > 2/255).mean():.2f}%")

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(inp);  ax[0].set_title("input (warped capture)");   ax[0].axis('off')
ax[1].imshow(prev); ax[1].set_title("prev DUT (fp32 buffer)");   ax[1].axis('off')
ax[2].imshow(dut);  ax[2].set_title("new DUT (int chain)");      ax[2].axis('off')
im = ax[3].imshow(d.max(axis=2), cmap='hot', vmin=0, vmax=d.max())
ax[3].set_title(f"|prev - new|  max={d.max()*255:.1f} LSB")
fig.colorbar(im, ax=ax[3], fraction=0.046)
plt.tight_layout()
plt.savefig("data_files/photo_chain_compare_int.png", dpi=110, bbox_inches='tight')
plt.show()