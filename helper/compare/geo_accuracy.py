#!/usr/bin/env python3
"""geo_accuracy.py -- warped-frame accuracy of the three geometric-stage variants.

  ref     : FP32 LUT  + float bilinear           (geo_fp32_full; the reference)
  q88_fp  : Q8.8 LUT  + float bilinear           (geo_fp32: Q8.8 storage, FP32 datapath)
  q88_int : Q8.8 LUT  + integer bilinear          (geo_int: bit-exact model of
            bilinear_interp_int.sv -- Q0.8 weights, (acc + 2^15) >> 16, edge clamp
            from fetch_four_nearest_int.sv)

All three are compared against `ref` after rounding to uint8.

USAGE (from repo root)
  python3 helper/compare/geo_accuracy.py --images ~/corpus10k --limit 100
  python3 helper/compare/geo_accuracy.py --images results/roundtrip --glob "*_1_clean.png"
"""
import argparse, glob, os, csv
import numpy as np
from PIL import Image

W = H = 256

def lines(p):
    return [l.split("//")[0].strip() for l in open(p) if l.split("//")[0].strip()]

def load_lut_fp32(p):               # 64-bit {y[63:32], x[31:0]} fp32
    a = np.frombuffer(bytes.fromhex("".join(t.zfill(16) for t in lines(p))), ">f4")
    a = a.astype(np.float64).reshape(H, W, 2)
    return a[..., 1], a[..., 0]

def load_lut_q88(p):                # 32-bit {y_q88[31:16], x_q88[15:0]}
    v = np.array([int(t, 16) for t in lines(p)], dtype=np.int64).reshape(H, W)
    return v & 0xFFFF, (v >> 16) & 0xFFFF

def warp_float(img, lx, ly):
    img = img.astype(np.float64)
    x0 = np.floor(lx).astype(int); y0 = np.floor(ly).astype(int)
    dx = (lx - x0)[..., None];     dy = (ly - y0)[..., None]
    x0c = np.clip(x0, 0, W-1); x1c = np.clip(x0+1, 0, W-1)
    y0c = np.clip(y0, 0, H-1); y1c = np.clip(y0+1, 0, H-1)
    out = ((1-dx)*(1-dy)*img[y0c, x0c] + (1-dx)*dy*img[y1c, x0c] +
           dx*(1-dy)*img[y0c, x1c]     + dx*dy*img[y1c, x1c])
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)

def warp_int(img, xq, yq):          # mirrors csf_int + ffn_int + bilinear_interp_int
    xi, yi = xq >> 8, yq >> 8
    dx, dy = (xq & 0xFF)[..., None], (yq & 0xFF)[..., None]
    x1, y1 = np.minimum(xi + 1, W-1), np.minimum(yi + 1, H-1)
    mdx, mdy = 256 - dx, 256 - dy
    p = img.astype(np.int64)
    acc = (mdx*mdy*p[yi, xi] + mdx*dy*p[y1, xi] + dx*mdy*p[yi, x1] + dx*dy*p[y1, x1])
    return ((acc + 32768) >> 16).astype(np.uint8)

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return float("inf") if mse == 0 else 10*np.log10(255.0**2 / mse)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--glob", default="*.png")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lut-fp32", default="data_files/lut_fp32.hex")
    ap.add_argument("--lut-q88", default="data_files/lut_quant.hex")
    ap.add_argument("--out", default="results/ablation/geo_accuracy.csv")
    a = ap.parse_args()

    lst = os.path.join(a.images, "list.txt")
    files = ([os.path.join(a.images, l.strip()) for l in open(lst) if l.strip()]
             if os.path.exists(lst) and a.glob == "*.png"
             else sorted(glob.glob(os.path.join(a.images, a.glob))))
    if a.limit: files = files[:a.limit]

    lx, ly = load_lut_fp32(a.lut_fp32)
    xq, yq = load_lut_q88(a.lut_q88)
    lxq, lyq = xq / 256.0, yq / 256.0

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = []
    for f in files:
        img = np.array(Image.open(f).convert("RGB").resize((W, H)), dtype=np.uint8)
        ref = warp_float(img, lx, ly)
        qf  = warp_float(img, lxq, lyq)
        qi  = warp_int(img, xq, yq)
        rows.append(dict(name=os.path.basename(f),
                         q88_fp_psnr=psnr(ref, qf),  q88_fp_max=int(np.abs(ref.astype(int)-qf).max()),
                         q88_int_psnr=psnr(ref, qi), q88_int_max=int(np.abs(ref.astype(int)-qi).max())))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def stat(k):
        v = np.array([r[k] for r in rows], dtype=float); v = v[np.isfinite(v)]
        return (np.mean(v), np.min(v)) if len(v) else (float("inf"), float("inf"))
    print(f"{len(rows)} images  ->  {a.out}")
    for lbl, k in [("Q8.8 LUT + FP32 bilinear (geo_fp32)", "q88_fp"),
                   ("Q8.8 LUT + int  bilinear (geo_int) ", "q88_int")]:
        m, lo = stat(k + "_psnr"); mx = max(r[k + "_max"] for r in rows)
        print(f"  {lbl}: mean PSNR {m:.2f} dB, worst {lo:.2f} dB, worst pixel {mx} LSB  (vs FP32 LUT)")

if __name__ == "__main__":
    main()