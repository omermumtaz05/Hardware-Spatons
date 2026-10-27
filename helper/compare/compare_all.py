#!/usr/bin/env python3
"""compare_all.py -- loop the whole image suite, no sim, no torch.

Reads what is already on disk:
    source   data_files/dut_out/<name>.src.hex
    warp     data_files/golden_cache/<name>/full_warped_q.hex   (optional)
    golden   data_files/golden_cache/<name>/full_golden_q.hex
    DUT      data_files/dut_out/<name>.dut.hex

Writes:
    data_files/compare/<name>.png       5-panel per image
    data_files/compare/_errormaps.png   contact sheet of all error maps
    prints a summary table with PSNR

Run:  python helper/compare_all.py            # every image in golden_cache
      python helper/compare_all.py img_noise  # named subset
"""

import os
import struct
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = H = 256
DATA   = "data_files"
CACHE  = os.path.join(DATA, "golden_cache")
DUTDIR = os.path.join(DATA, "dut_out")
OUT    = os.path.join(DATA, "compare")


def load_hex3(path):
    """N*3-line hex dump -> float [0,1], HxWx3. Sniffs fp32 (8 char) vs uint8 (2 char)."""
    toks = [w for ln in open(path) for w in ln.split()
            if w and not w.startswith("//")]
    if not toks:
        raise ValueError(f"{path}: empty")
    n = len(toks[0])
    if n == 8:
        arr = np.array([struct.unpack('>f', bytes.fromhex(t))[0] for t in toks], np.float32)
    elif n <= 2:
        arr = np.array([int(t, 16) for t in toks], np.float32) / 255.0
    else:
        raise ValueError(f"{path}: unexpected token width {n} ('{toks[0]}')")
    if arr.size != H * W * 3:
        raise ValueError(f"{path}: {arr.size} values, expected {H*W*3}")
    return np.clip(arr.reshape(H, W, 3), 0, 1)


def to_u8(x):
    """Quantise the way the hardware does, so the diff is an integer LSB count."""
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.int16)


def shift_check(g8, d8):
    """A whole-image offset makes the error map look edge-detected even though every
    value is individually right. Test it rather than eyeballing."""
    gf, df = g8.reshape(-1, 3), d8.reshape(-1, 3)
    base = np.abs(gf - df).mean()
    best_s, best_m = 0, base
    for s in range(1, 9):
        m = np.abs(gf[s:] - df[:-s]).mean()
        if m < best_m:
            best_s, best_m = s, m
    return best_s, best_m, base


def analyse(name):
    gdir = os.path.join(CACHE, name)
    p_gold = os.path.join(gdir, "full_golden_q.hex")
    p_warp = os.path.join(gdir, "full_warped_q.hex")
    p_dut  = os.path.join(DUTDIR, f"{name}.dut.hex")
    p_src  = os.path.join(DUTDIR, f"{name}.src.hex")

    for p in (p_gold, p_dut):
        if not os.path.exists(p):
            print(f"  !! missing {p} -- skipped")
            return None

    gold = load_hex3(p_gold)
    dut  = load_hex3(p_dut)
    src  = load_hex3(p_src)  if os.path.exists(p_src)  else None
    warp = load_hex3(p_warp) if os.path.exists(p_warp) else None

    g8, d8 = to_u8(gold), to_u8(dut)
    err  = np.abs(g8 - d8)          # per-channel, in LSBs
    emax = err.max(axis=2)          # per-pixel worst channel
    rmse = float(np.sqrt((err.astype(float) ** 2).mean()))
    psnr = 20 * np.log10(255.0 / rmse) if rmse > 0 else float("inf")

    sft, sft_m, base_m = shift_check(g8, d8)
    wi = int(emax.argmax()); wy, wx = wi // W, wi % W

    r = dict(
        name=name,
        exact=100 * np.mean(err == 0),
        w1=100 * np.mean(err <= 1),
        w2=100 * np.mean(err <= 2),
        mean=float(err.mean()),
        p99=float(np.percentile(err, 99)),
        p999=float(np.percentile(err, 99.9)),
        mx=int(err.max()),
        gt2=100 * np.mean(err > 2),
        rmse=rmse,
        psnr=psnr,
        shift=sft,
        sat0_d=100 * np.mean(d8 == 0),   sat255_d=100 * np.mean(d8 == 255),
        sat0_g=100 * np.mean(g8 == 0),   sat255_g=100 * np.mean(g8 == 255),
        worst=f"(x={wx}, y={wy}) gold={list(g8[wy, wx])} dut={list(d8[wy, wx])}",
        emax=emax,
    )

    # ---- 5-panel figure ----
    panels = []
    if src  is not None: panels.append((src,  "source"))
    if warp is not None: panels.append((warp, "python warp (geo ref)"))
    panels.append((gold, "golden (fp32 reference)"))
    panels.append((dut,  "DUT (RTL, all-integer)"))

    vmax = max(int(emax.max()), 1)
    fig, ax = plt.subplots(1, len(panels) + 1, figsize=(4.8 * (len(panels) + 1), 5))
    for a, (img, title) in zip(ax, panels):
        a.imshow(img); a.set_title(title, fontsize=11)
    im = ax[-1].imshow(emax, cmap="inferno", vmin=0, vmax=vmax)
    ax[-1].set_title(f"|golden - DUT| LSBs   max={vmax}   PSNR {psnr:.1f} dB", fontsize=11)
    for a in ax:
        a.axis("off")
    fig.colorbar(im, ax=ax[-1], fraction=0.046)
    fig.suptitle(name, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"{name}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    if sft:
        print(f"  *** SHIFTED BY {sft} PIXELS (mean {sft_m:.3f} vs {base_m:.3f} unshifted)"
              f" -- side-FIFO / z_idx alignment, not arithmetic")
    print(f"  mean {r['mean']:.3f}  max {r['mx']:2d}  RMSE {rmse:.3f}  PSNR {psnr:.1f} dB")
    return r


def contact_sheet(results):
    """All error maps side by side -- the fastest way to spot STRUCTURE.
    Shared colour scale so the images are comparable to each other."""
    n = len(results)
    cols = 5
    rows = (n + cols - 1) // cols
    vmax = max(int(r["emax"].max()) for r in results)

    fig, ax = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.5 * rows))
    ax = np.atleast_1d(ax).ravel()
    for a in ax:
        a.axis("off")
    for a, r in zip(ax, results):
        im = a.imshow(r["emax"], cmap="inferno", vmin=0, vmax=vmax)
        a.set_title(f"{r['name']}\n{r['psnr']:.1f} dB  max {r['mx']}", fontsize=9)
    fig.suptitle(f"|golden - DUT| per pixel, worst channel  (shared scale 0-{vmax} LSB)",
                 fontsize=12, y=1.0)
    fig.colorbar(im, ax=ax.tolist(), fraction=0.02, pad=0.02)
    plt.savefig(os.path.join(OUT, "_errormaps.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = args if args else sorted(
        d for d in os.listdir(CACHE) if os.path.isdir(os.path.join(CACHE, d)))

    print(f"{len(names)} images\n")
    results = []
    for nm in names:
        print(f"=== {nm}")
        try:
            r = analyse(nm)
        except Exception as e:
            print(f"  !! {e}")
            continue
        if r:
            results.append(r)

    if not results:
        sys.exit("nothing to report")

    contact_sheet(results)

    results.sort(key=lambda r: -r["psnr"])
    print("\n" + "=" * 92)
    print(f"  {'image':<20}{'exact%':>8}{'<=1LSB':>8}{'<=2LSB':>8}"
          f"{'mean':>7}{'p99.9':>7}{'max':>5}{'RMSE':>7}{'PSNR dB':>9}{'shift':>7}")
    print("  " + "-" * 88)
    for r in results:
        print(f"  {r['name']:<20}{r['exact']:>8.2f}{r['w1']:>8.2f}{r['w2']:>8.2f}"
              f"{r['mean']:>7.3f}{r['p999']:>7.1f}{r['mx']:>5d}{r['rmse']:>7.3f}"
              f"{r['psnr']:>9.1f}{r['shift']:>7d}")
    print("  " + "-" * 88)

    rm = np.mean([r["rmse"] for r in results])
    worst = min(results, key=lambda r: r["psnr"])
    print(f"  mean RMSE {rm:.3f} -> {20*np.log10(255/rm):.1f} dB   |   "
          f"mean PSNR {np.mean([r['psnr'] for r in results]):.1f} dB   |   "
          f"WORST {worst['name']} {worst['psnr']:.1f} dB (max {worst['mx']} LSB)")
    print("=" * 92)
    print(f"""
  Wrote {len(results)} figures + a contact sheet to {OUT}/

  LOOK AT _errormaps.png. PSNR is an aggregate -- it cannot tell you whether the
  error is STRUCTURED, and structure is what is actually visible.

    scattered speckle          quantisation noise. expected. fine.
    outline / edges lit up     misalignment or latency mismatch
    horizontal bands           line-buffer or row-indexing problem
    one region hot             coordinate / LUT-index drift
    uniform wash               systematic bias (flat images do this normally)

  A 'shift' column above that is non-zero is a hard failure regardless of PSNR.
""")


if __name__ == "__main__":
    main()