#!/usr/bin/env python3
"""report.py -- aggregate results/sweep.csv -> console summary + figures.

USAGE  python3 helper/sweep/report.py results/sweep.csv
"""
import csv, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sweep import POINTS

path = sys.argv[1] if len(sys.argv) > 1 else "results/sweep.csv"
rows = list(csv.DictReader(open(path)))
ok = [r for r in rows if r["status"] == "OK"]
bad = [r for r in rows if r["status"] != "OK"]
col = lambda k, t=float: np.array([t(r[k]) for r in ok if r.get(k) not in ("", None)])

print(f"{len(rows)} rows: {len(ok)} OK, {len(bad)} failed")
for r in bad[:20]: print(f"  !! {r['name']}: {r['status']}")
if not ok: sys.exit()

cyc = col("cyc", int)
print(f"\ncycles/frame: unique {sorted(set(cyc))[:5]}  (data-independent if one value)")

print(f"\n{'':10}{'mean':>8}{'median':>8}{'p1':>8}{'min':>8}   worst-max-LSB")
for pre, lab in (("q", "vs quant"), ("fp", "vs fp32")):
    p = col(f"{pre}_psnr"); mx = col(f"{pre}_max", int)
    print(f"{lab:10}{p.mean():8.2f}{np.median(p):8.2f}{np.percentile(p,1):8.2f}{p.min():8.2f}"
          f"   {mx.max()}   (PSNR dB)")
    rm = col(f"{pre}_rmse")
    print(f"{'':10}pooled-RMSE PSNR {20*np.log10(255/np.sqrt((rm**2).mean())):.2f} dB, "
          f">2LSB mean {col(f'{pre}_gt2').mean():.2f}%")

print("\nworst 10 (vs fp32):")
for r in sorted(ok, key=lambda r: float(r["fp_psnr"]))[:10]:
    print(f"  {r['name']:30} fp {float(r['fp_psnr']):6.2f}  q {float(r['q_psnr']):6.2f}  max {r['fp_max']}")

print("\nrequant points:  S  | worst need | margin | images clamped | total clamps")
fig, ax = plt.subplots(2, 4, figsize=(18, 8)); ax = ax.ravel()
for i, (l, k) in enumerate(POINTS):
    pre = f"{l}_{k}"
    need = col(f"{pre}_need", int); cl = col(f"{pre}_clamps", int); S = col(f"{pre}_S", int)
    if need.size == 0:
        print(f"  {pre:10} (no RQ lines in sim log)"); continue
    s = int(S.max())
    print(f"  {pre:10} {s:3d} | {need.max():10d} | {s-need.max():6d} | "
          f"{int((cl>0).sum()):14d} | {int(cl.sum())}")
    b = np.arange(need.min(), need.max() + 2) - 0.5
    ax[i].hist(need, bins=b); ax[i].axvline(s + 0.5, color="r", ls="--", label=f"S={s}")
    ax[i].set_title(f"{pre}: per-image max need"); ax[i].legend()
for j, (pre, lab) in enumerate((("q", "vs quant model"), ("fp", "vs fp32"))):
    ax[6 + j].hist(col(f"{pre}_psnr"), bins=60); ax[6 + j].set_title(f"PSNR {lab} (dB)")
plt.tight_layout()
png = os.path.splitext(path)[0] + "_report.png"
plt.savefig(png, dpi=110); print(f"\nsaved {png}")
