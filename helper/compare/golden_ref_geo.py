"""
Compare DUT geometric-stage output against the ideal float reference.
Shows 4 panels (source, ideal ref, DUT, diff) and prints diff stats.

"""

import numpy as np
import struct
import matplotlib.pyplot as plt

W, H = 256, 256
N = W * H

LUT_FILE     = "data_files/lut_fp32.hex"
IMAGE_FILE   = "data_files/image.hex"
CAPTURE_FILE = "data_files/capture_array_int.hex"


with open(CAPTURE_FILE) as f:
    for i, line in enumerate(f):
        v = line.strip()
        if not v:
            print(f"line {i}: BLANK")
            continue
        try:
            bytes.fromhex(v.zfill(8))
        except ValueError:
            print(f"line {i}: BAD -> {repr(v)}")

def hex_to_f32(h):
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


# ---------- load LUT (64-bit {y' high, x' low}, pixel units) ----------
print("loading LUT...")
lut_lines = read_lines(LUT_FILE)
assert len(lut_lines) == N
lut_x = np.zeros((H, W))
lut_y = np.zeros((H, W))
for idx, line in enumerate(lut_lines):
    h = line.zfill(16)
    y, x = idx // W, idx % W
    lut_y[y, x] = hex_to_f32(h[0:8])
    lut_x[y, x] = hex_to_f32(h[8:16])

# ---------- load source image ----------
print("loading image...")
img_lines = read_lines(IMAGE_FILE)
assert len(img_lines) == N * 3
img = (np.array([int(v, 16) for v in img_lines], dtype=np.uint8)
       .reshape(H, W, 3).astype(np.float64) / 255.0)

# ---------- ideal float bilinear warp (reference) ----------
print("rendering ideal warp...")
x0 = np.floor(lut_x).astype(int)
y0 = np.floor(lut_y).astype(int)
dx = (lut_x - x0)[..., None]
dy = (lut_y - y0)[..., None]
x0c = np.clip(x0,     0, W - 1); x1c = np.clip(x0 + 1, 0, W - 1)
y0c = np.clip(y0,     0, H - 1); y1c = np.clip(y0 + 1, 0, H - 1)
ref = np.clip((1 - dx) * (1 - dy) * img[y0c, x0c] +
              (1 - dx) * dy       * img[y1c, x0c] +
              dx       * (1 - dy) * img[y0c, x1c] +
              dx       * dy       * img[y1c, x1c], 0, 1)

# ---------- load DUT capture ----------
print("loading DUT capture...")
cap = read_lines(CAPTURE_FILE, skip_comments=True)

assert len(cap) == N

dut = np.zeros((H, W, 3), dtype=np.float32)

for idx, line in enumerate(cap):
    line = line.strip().zfill(24)

    p2 = hex_to_f32(line[0:8])    # bits [95:64]
    p1 = hex_to_f32(line[8:16])   # bits [63:32]
    p0 = hex_to_f32(line[16:24])  # bits [31:0]

    y = idx // W
    x = idx % W

    dut[y, x, 0] = p0
    dut[y, x, 1] = p1
    dut[y, x, 2] = p2

dut = np.clip(dut, 0, 1)

# ---------- diff stats ----------
diff = np.abs(ref - dut)
dmax = diff.max(axis=2)
print("\n=== ref vs DUT ===")
print(f"max  diff: {diff.max():.3e}")
print(f"mean diff: {diff.mean():.3e}")
print(f"pixels diff > 1e-3: {int((dmax > 1e-3).sum())} / {N}")
print(f"pixels diff > 1e-6: {int((dmax > 1e-6).sum())} / {N}")
if (dmax > 1e-3).any():
    flat = diff.reshape(-1, 3).sum(axis=1)
    w = int(flat.argmax()); wy, wx = w // W, w % W
    print(f"worst pixel (x={wx}, y={wy}): ref={ref[wy,wx]} dut={dut[wy,wx]}")
else:
    print("** all pixels match within FP32 precision -- PASS **")


diff = np.abs(ref - dut).max(axis=2)
bad = np.argwhere(diff > 1e-3)
# for each bad output pixel, look up where the LUT sent it
edge_cnt = 0
for y, x in bad:
    sx, sy = lut_x[y, x], lut_y[y, x]   # your LUT's mapped source coords
    if sx >= 254 or sy >= 254 or sx < 0 or sy < 0:
        edge_cnt += 1
print(f"{edge_cnt}/{len(bad)} bad pixels map to a source edge")

# pick a bad, non-edge pixel
y, x = 140, 130   # example interior — choose one from your bad list not near edges
sx, sy = lut_x[y,x], lut_y[y,x]
xi, yi = int(np.floor(sx)), int(np.floor(sy))
print(f"out({x},{y}) -> src({sx:.3f},{sy:.3f}) floor({xi},{yi})")
print("expected P00 (xi,yi)    :", img[yi,   xi])
print("expected P01 (xi+1,yi)  :", img[yi,   xi+1])
print("expected P10 (xi,yi+1)  :", img[yi+1, xi])
print("expected P11 (xi+1,yi+1):", img[yi+1, xi+1])

# ---------- 4 panels ----------
fig, ax = plt.subplots(1, 4, figsize=(18, 5))
ax[0].imshow(img); ax[0].set_title("source (original)"); ax[0].axis('off')
ax[1].imshow(ref); ax[1].set_title("ideal warp (ref)");  ax[1].axis('off')
ax[2].imshow(dut); ax[2].set_title("DUT output");        ax[2].axis('off')
im = ax[3].imshow(dmax, cmap='hot'); ax[3].set_title("|ref - DUT|"); ax[3].axis('off')
fig.colorbar(im, ax=ax[3], fraction=0.046)
plt.tight_layout()
plt.savefig("data_files/compare.png", dpi=110, bbox_inches='tight')
print("\nsaved data_files/compare.png") 
plt.show()


