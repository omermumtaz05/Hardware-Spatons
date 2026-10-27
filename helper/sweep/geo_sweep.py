#!/usr/bin/env python3
"""geo_sweep.py -- RTL accuracy of the three geometric-stage variants over N images.

For each image:
  1. write data_files/rom_{ee,eo,oe,oo}.hex (same packing as run_sweep.py)
  2. run the three prebuilt Verilator binaries (build with: bash helper/sim/build_geo.sh)
       obj_geo_fp32_full/Vtop_geo_fp32_full_tb -> data_files/capture_geo_fp32_full.hex  (FP32 LUT, FP32 math)
       obj_geo_fp32/Vtop_geo_fp32_tb           -> data_files/capture_geo_fp32.hex       (Q8.8 LUT, FP32 math)
       obj_geo_int/Vtop_geo_int_tb             -> data_files/capture_array_int.hex      (Q8.8 LUT, int math)
  3. compare each against the software FP32 warp (golden_vec.warp with lut_fp32.hex,
     validated against Javad's PyTorch grid_sample by check_geo_lut.py)

The original rom_*.hex are backed up and restored at the end.

USAGE (WSL, repo root):
  python3 helper/sweep/geo_sweep.py --corpus ~/corpus10k --limit 20
"""
import argparse, csv, os, re, shutil, subprocess, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden_vec as gv

W = H = 256
DD = "data_files"
ROMS = [f"rom_{t}.hex" for t in ("ee", "eo", "oe", "oo")]

VARIANTS = [  # name, binary, capture file, format
    ("fp32_full", "obj/geo_fp32_full/Vtop_geo_fp32_full_tb", "capture_geo_fp32_full.hex", "f32"),
    ("fp32_q88",  "obj/geo_fp32/Vtop_geo_fp32_tb",           "capture_geo_fp32.hex",      "f32"),
    ("int_q88",   "obj/geo_int/Vtop_geo_int_tb",             "capture_array_int.hex",     "u8"),
]

def write_roms(img):
    for tag, sub in (("ee", img[0::2, 0::2]), ("eo", img[0::2, 1::2]),
                     ("oe", img[1::2, 0::2]), ("oo", img[1::2, 1::2])):
        w = (sub[..., 0].astype(np.uint32) << 16) | (sub[..., 1].astype(np.uint32) << 8) | sub[..., 2]
        p = os.path.join(DD, f"rom_{tag}.hex")
        if os.path.islink(p): os.unlink(p)
        with open(p, "w") as f:
            f.write("".join(f"{v:06x}\n" for v in w.reshape(-1)))

def read_capture(path, fmt):
    rows = [l.strip() for l in open(path) if l.strip()]
    if len(rows) != W * H:
        raise RuntimeError(f"{path}: {len(rows)} pixels, expected {W*H}")
    if fmt == "u8":   # "%02h%02h%02h" = p2 p1 p0
        a = np.array([[int(r[4:6], 16), int(r[2:4], 16), int(r[0:2], 16)] for r in rows], np.uint8)
    else:             # "%08h%08h%08h" = p2 p1 p0, fp32 in [0,1]
        raw = bytes.fromhex("".join(r[16:24] + r[8:16] + r[0:8] for r in rows))
        f = np.frombuffer(raw, ">f4").astype(np.float64).reshape(-1, 3)
        a = np.clip(np.rint(f * 255.0), 0, 255).astype(np.uint8)
    return a.reshape(H, W, 3)

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--lut", default="data_files/lut_fp32.hex")
    ap.add_argument("--out", default="results/ablation/geo_sweep.csv")
    a = ap.parse_args()

    corpus = os.path.expanduser(a.corpus)
    lst = os.path.join(corpus, "list.txt")
    names = ([l.strip() for l in open(lst) if l.strip()] if os.path.exists(lst)
             else sorted(f for f in os.listdir(corpus) if f.endswith(".png")))[:a.limit]

    for _, b, _, _ in VARIANTS:
        if not os.path.exists(b):
            sys.exit(f"missing {b} -- run: bash helper/sim/build_geo.sh")

    lx, ly = gv.load_lut_fp32(a.lut)
    bak = os.path.join(DD, "rom_backup"); os.makedirs(bak, exist_ok=True)
    for r in ROMS:
        if os.path.exists(os.path.join(DD, r)):
            shutil.copy(os.path.join(DD, r), bak)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = []
    try:
        for i, n in enumerate(names, 1):
            img = np.array(Image.open(os.path.join(corpus, n)).convert("RGB").resize((W, H)), np.uint8)
            write_roms(img)
            ref = np.clip(np.rint(gv.warp(img, lx, ly).astype(np.float64) * 255.0), 0, 255).astype(np.uint8)
            row = {"name": os.path.splitext(n)[0]}
            for v, b, cap, fmt in VARIANTS:
                capp = os.path.join(DD, cap)
                if os.path.exists(capp): os.remove(capp)
                r = subprocess.run(["./" + b], capture_output=True, text=True, timeout=600)
                m = re.search(r"DONE pixels=(\d+)", r.stdout)
                if not m:
                    raise RuntimeError(f"{v} on {n}: no DONE line\n{r.stdout[-500:]}")
                out = read_capture(capp, fmt)
                d = np.abs(out.astype(int) - ref.astype(int))
                row[f"{v}_psnr"] = round(psnr(out, ref), 3)
                row[f"{v}_max"] = int(d.max())
                row[f"{v}_exact"] = round(100.0 * np.mean(d == 0), 3)
            rows.append(row)
            print(f"[{i}/{len(names)}] {row['name']}  " +
                  "  ".join(f"{v}: {row[v+'_psnr']} dB / max {row[v+'_max']}" for v, *_ in VARIANTS))
    finally:
        for r in ROMS:
            s = os.path.join(bak, r)
            if os.path.exists(s): shutil.copy(s, os.path.join(DD, r))

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f"\n{len(rows)} images -> {a.out}   (vs software FP32 warp)")
    for v, *_ in VARIANTS:
        p = np.array([r[f"{v}_psnr"] for r in rows], float); fin = p[np.isfinite(p)]
        mean = f"{fin.mean():.2f}" if len(fin) else "inf"
        worst = f"{fin.min():.2f}" if len(fin) else "inf"
        print(f"  {v:10s}: mean PSNR {mean} dB, worst {worst} dB, "
              f"worst pixel {max(r[v+'_max'] for r in rows)} LSB, "
              f"exact pixels {np.mean([r[v+'_exact'] for r in rows]):.2f}%")

if __name__ == "__main__":
    main()