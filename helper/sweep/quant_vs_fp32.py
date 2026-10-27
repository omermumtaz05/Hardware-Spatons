#!/usr/bin/env python3
"""quant_vs_fp32.py -- PSNR of the quantized software model vs the fp32 model (no RTL).

Measures the algorithm's own quantization cost directly, on the same images as the sweep.
Adds column qf_psnr to a copy of the sweep CSV and prints the summary.

USAGE (repo root, WSL)
  python3 helper/sweep/quant_vs_fp32.py --corpus ~/corpus10k --csv results/sweep.csv
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse, csv, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden_vec as gv

G = None
def init(weights, lut):
    global G
    G = gv.Golden(weights, lut)

def one(path):
    img = np.asarray(Image.open(path).convert("RGB"), np.uint8)
    q, f = gv.to_u8(G.golden_q(img)), gv.to_u8(G.golden_fp32(img))
    rmse = float(np.sqrt(((q - f).astype(np.float64) ** 2).mean()))
    return os.path.splitext(os.path.basename(path))[0], (20 * np.log10(255 / rmse) if rmse else 99.0), rmse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--csv", default="results/sweep.csv")
    ap.add_argument("--out", default="results/sweep_qf.csv")
    ap.add_argument("--weights", default="#86-NN2_best.pth")
    ap.add_argument("--lut", default="data_files/lut_fp32.hex")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.csv)) if r["status"] == "OK"]
    corpus = os.path.expanduser(a.corpus)
    paths = [os.path.join(corpus, r["name"] + ".png") for r in rows]
    with Pool(a.workers, init, (gv.load_weights_pth(a.weights), a.lut)) as p:
        res = {}
        for i, (n, ps, rm) in enumerate(p.imap_unordered(one, paths, chunksize=8), 1):
            res[n] = (ps, rm)
            if i % 500 == 0 or i == len(paths):
                print(f"  {i}/{len(paths)}", flush=True)

    for r in rows:
        r["qf_psnr"], r["qf_rmse"] = round(res[r["name"]][0], 3), round(res[r["name"]][1], 4)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"{len(rows)} images\n{'':22}{'mean':>8}{'median':>8}{'p1':>8}{'min':>8}")
    for k, lab in (("q_psnr", "RTL vs quant model"), ("qf_psnr", "quant model vs fp32"),
                   ("fp_psnr", "RTL vs fp32")):
        x = np.array([float(r[k]) for r in rows])
        print(f"{lab:22}{x.mean():8.2f}{np.median(x):8.2f}{np.percentile(x,1):8.2f}{x.min():8.2f}")
    print(f"saved {a.out}")

if __name__ == "__main__":
    main()
