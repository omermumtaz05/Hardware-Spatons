#!/usr/bin/env python3
"""golden_vec.py -- whole-frame (vectorized) goldens. Replaces the per-pixel
torch loop in gen_full_golden.py (65,536 forward passes/image -> ~0.5 s/image).

Two references, same warp (fp32 geometric LUT, python bilinear):
  golden_q(img)   : bit-exact replica of gen_full_golden.py
                    (10b weights, 12b act, per-pixel dynamic act scale,
                     4096-entry sigmoid LUT). Verify with --selfcheck.
  golden_fp32(img): Javad's model in plain fp32, true sigmoid, no quantization.
                    This is the "software fp32" reference for the paper.

Both return float32 HxWx3 in [0,1]. Convert with to_u8() before differencing.

USAGE (from repo root)
  python helper/sweep/golden_vec.py --selfcheck
     -> regenerates image_orig and diffs against
        data_files/golden_cache/image_orig/full_golden_q.hex (must be 0 mismatches)
"""
import argparse, os, struct, sys
import numpy as np

W = H = 256
IN_DIM, HIDDEN, OUT_DIM = 29, 48, 3
WEIGHT_BITS, ACT_BITS = 10, 12
LAYERS = ["layer1", "layer2", "layer3", "residual"]

# ---------------- file io ----------------
def _lines(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.lstrip().startswith("//")]

def load_hex_u8_image(path):
    v = np.array([int(t, 16) for t in _lines(path)], np.uint8)
    assert v.size == W * H * 3, f"{path}: {v.size} values"
    return v.reshape(H, W, 3)

def load_hex_f32(path):
    a = np.frombuffer(bytes.fromhex("".join(t.zfill(8) for t in _lines(path))), ">f4")
    return a.astype(np.float32)

def load_lut_fp32(path):
    """64-bit line {y'[63:32], x'[31:0]} big-endian fp32 -> (lx, ly) float64."""
    raw = bytes.fromhex("".join(t.zfill(16) for t in _lines(path)))
    a = np.frombuffer(raw, ">f4").astype(np.float64).reshape(H, W, 2)
    return a[..., 1], a[..., 0]                       # x, y

def to_u8(x):
    """Same quantization full_sys_v3.py uses on the golden."""
    return np.clip(np.rint(np.clip(x, 0, 1) * 255.0), 0, 255).astype(np.int16)

# ---------------- weights ----------------
def load_weights_pth(path):
    import torch
    sd = torch.load(path, map_location="cpu")["model_state_dict"]
    return {n: (sd[f"models.1.model.{n}.weight"].numpy().astype(np.float32),
                sd[f"models.1.model.{n}.bias"].numpy().astype(np.float32))
            for n in LAYERS}

# ---------------- quant_core, per-row ----------------
def _absmax_scale(x64, qmax):
    am = np.maximum(np.abs(x64.min(axis=-1, keepdims=True)),
                    np.abs(x64.max(axis=-1, keepdims=True)))
    return np.where(am > 1e-10, am / np.float64(qmax), np.float64(1e-10))

def quant_weights(w):
    """uniform_quantize, per-tensor, symmetric."""
    w64 = w.astype(np.float64)
    qmax = 2 ** (WEIGHT_BITS - 1) - 1
    am = max(abs(w64.min()), abs(w64.max()))
    s = am / np.float64(qmax) if am > 1e-10 else np.float64(1e-10)
    q = np.clip(np.round(w64 / s + 0.0), -qmax - 1, qmax).astype(np.int64)
    return q, np.float64(s)

def quant_linear(x32, Wq, sw, b32):
    """quant_linear() in gen_full_golden.py, with one scale per row (= per pixel)."""
    qmax = 2 ** (ACT_BITS - 1) - 1
    x64 = x32.astype(np.float64)
    sx = _absmax_scale(x64, qmax)
    xq = np.clip(np.round(x64 / sx + 0.0), -qmax - 1, qmax).astype(np.int64)
    acc = np.matmul(xq, Wq.T)                                   # int64, exact
    fv = (acc.astype(np.float64) - 0.0) * (sx * sw)             # dequant @ sx*sw
    ns = _absmax_scale(fv, qmax)                                # requantize_direct
    rq = np.clip(np.rint(fv / ns), -qmax - 1, qmax).astype(np.int64)
    out = ((rq.astype(np.float64) - 0.0) * ns).astype(np.float32)
    return out + b32

# sigmoid LUT: built with the exact scalar code of gen_full_golden.lut_sig
_DOMAIN, _LUT_N = 8.0, 4096
_STEP = (2 * _DOMAIN) / (_LUT_N - 1)
_INV_STEP = np.float32((_LUT_N - 1) / (2 * _DOMAIN))
_SIG_TAB = np.array([np.float32(1.0 / (1.0 + np.exp(-(-_DOMAIN + i * _STEP))))
                     for i in range(_LUT_N)], np.float32)

def lut_sig(z):
    z = z.astype(np.float32)
    idx = ((z + np.float32(_DOMAIN)).astype(np.float32) * _INV_STEP).astype(np.float32)
    i = np.floor((idx + np.float32(0.5)).astype(np.float32)).astype(np.int64)
    out = _SIG_TAB[np.clip(i, 0, _LUT_N - 1)]
    out = np.where(z <= -_DOMAIN, np.float32(0.0), out)
    out = np.where(z >= _DOMAIN, np.float32(1.0), out)
    return out.astype(np.float32)

# ---------------- geo + features ----------------
def warp(img_u8, lx, ly):
    img = img_u8.astype(np.float64) / 255.0
    x0 = np.floor(lx).astype(int); y0 = np.floor(ly).astype(int)
    dx = (lx - x0)[..., None];     dy = (ly - y0)[..., None]
    x0c = np.clip(x0, 0, W-1); x1c = np.clip(x0+1, 0, W-1)
    y0c = np.clip(y0, 0, H-1); y1c = np.clip(y0+1, 0, H-1)
    out = ((1-dx)*(1-dy)*img[y0c, x0c] + (1-dx)*dy*img[y1c, x0c] +
           dx*(1-dy)*img[y0c, x1c]     + dx*dy*img[y1c, x1c])
    return np.clip(out, 0, 1).astype(np.float32)

def features(wimg):
    """29 = [x, y, centre RGB, 8 neighbours x RGB], reflect padding, dy-major."""
    xs = np.linspace(0, 1, W, dtype=np.float32); ys = np.linspace(0, 1, H, dtype=np.float32)
    p = np.pad(wimg, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    cols = [np.broadcast_to(xs[None, :, None], (H, W, 1)),
            np.broadcast_to(ys[:, None, None], (H, W, 1)), wimg]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0: continue
            cols.append(p[1+dy:1+dy+H, 1+dx:1+dx+W])
    return np.concatenate(cols, axis=2).reshape(-1, IN_DIM).astype(np.float32)

# ---------------- model ----------------
class Golden:
    def __init__(self, weights, lut_path):
        self.w = weights
        self.lx, self.ly = load_lut_fp32(lut_path)
        self.q = {n: quant_weights(self.w[n][0]) for n in LAYERS}

    def golden_q(self, img_u8, return_warp=False):
        wimg = warp(img_u8, self.lx, self.ly); ft = features(wimg)
        ql = lambda x, n: quant_linear(x, self.q[n][0], self.q[n][1], self.w[n][1])
        h1 = np.maximum(ql(ft, "layer1"), 0)
        h2 = np.maximum(ql(h1, "layer2"), 0)
        z = (ql(h2, "layer3") + ql(ft, "residual")).astype(np.float32)
        d = lut_sig(z)
        fin = np.clip(wimg.reshape(-1, 3) + (4.0 * d - 2.0), 0, 1).astype(np.float32)
        fin = fin.reshape(H, W, 3)
        return (fin, wimg, ft) if return_warp else fin

    def golden_fp32(self, img_u8):
        wimg = warp(img_u8, self.lx, self.ly); ft = features(wimg)
        lin = lambda x, n: x @ self.w[n][0].T + self.w[n][1]
        h1 = np.maximum(lin(ft, "layer1"), 0)
        h2 = np.maximum(lin(h1, "layer2"), 0)
        z = (lin(h2, "layer3") + lin(ft, "residual")).astype(np.float64)
        d = 1.0 / (1.0 + np.exp(-z))
        fin = np.clip(wimg.reshape(-1, 3) + (4.0 * d - 2.0), 0, 1)
        return fin.astype(np.float32).reshape(H, W, 3)

# ---------------- self-check ----------------
def selfcheck(g, data="data_files"):
    cache = os.path.join(data, "golden_cache", "image_orig")
    img = load_hex_u8_image(os.path.join(data, "image_orig.hex"))
    fin, wimg, ft = g.golden_q(img, return_warp=True)
    ok = True
    for name, mine in [("full_warped_q.hex", wimg.reshape(-1)),
                       ("full_features_q.hex", ft.reshape(-1)),
                       ("full_golden_q.hex", fin.reshape(-1))]:
        p = os.path.join(cache, name)
        if not os.path.exists(p):
            print(f"  {name:22s} (missing, skipped)"); continue
        ref = load_hex_f32(p)
        bad = int(np.sum(ref.view(np.uint32) != mine.view(np.uint32)))
        u8bad = int(np.sum(to_u8(ref) != to_u8(mine)))
        print(f"  {name:22s} bit mismatches {bad:7d}/{ref.size}   uint8 mismatches {u8bad}")
        ok &= (u8bad == 0)
    print("SELFCHECK", "PASS" if ok else "FAIL")
    return ok

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="#86-NN2_best.pth")
    ap.add_argument("--lut", default="data_files/lut_fp32.hex")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    g = Golden(load_weights_pth(a.weights), a.lut)
    if a.selfcheck:
        sys.exit(0 if selfcheck(g) else 1)
