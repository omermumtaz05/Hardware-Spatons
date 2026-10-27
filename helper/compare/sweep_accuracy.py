#!/usr/bin/env python3
"""
sweep_accuracy.py -- RTL DUT vs python golden, across a whole image suite.

For each image:
    1. gen_rom_banks.py      image -> 4 parity banks + data_files/image.hex
    2. gen_full_golden.py    -> full_golden_q.hex   (CACHED, see below)
    3. run the RTL sim       -> full_sys_st.hex
    4. full_sys_v3.py        -> parse mean / max / >2 LSB
Then prints one summary table.

None of the four scripts need modifying. They all use fixed filenames in
data_files/, and this driver just keeps them in sync one image at a time.

GOLDEN CACHING
  Step 2 runs 65,536 torch forward passes and is by far the slowest part.
  Goldens are archived to <cache>/<name>/ after the first run and restored
  on later runs, so a re-sweep after an RTL change costs only steps 1, 3, 4.
  Delete the cache dir if the weights or the LUT change.

USAGE
    python sweep_accuracy.py                       # all of data_files/test_images
    python sweep_accuracy.py img_very_dark img_noise   # named subset
    python sweep_accuracy.py --no-golden           # reuse cache only, never regenerate
"""

import os
import re
import sys
import shutil
import subprocess

HELPER   = "helper"
DATA     = "data_files"
IMGDIR   = os.path.join(DATA, "test_images")
CACHE    = os.path.join(DATA, "golden_cache")
PY  = "python"
SIM = ["wsl", "./obj_dir/Vtop_full_int_st_tb"] 
# files gen_full_golden.py produces that full_sys_v3.py consumes
GOLDEN_FILES = ["full_golden_q.hex", "full_warped_q.hex", "full_features_q.hex"]


def run(cmd, quiet=True):
    r = subprocess.run(cmd, capture_output=True, text=True, shell=False)
    if r.returncode != 0:
        print(f"    !! FAILED: {' '.join(cmd)}")
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        return None
    return r.stdout


def have_cache(name):
    d = os.path.join(CACHE, name)
    return all(os.path.exists(os.path.join(d, f)) for f in GOLDEN_FILES)


def save_cache(name):
    d = os.path.join(CACHE, name)
    os.makedirs(d, exist_ok=True)
    for f in GOLDEN_FILES:
        src = os.path.join(DATA, f)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(d, f))


def load_cache(name):
    d = os.path.join(CACHE, name)
    for f in GOLDEN_FILES:
        shutil.copyfile(os.path.join(d, f), os.path.join(DATA, f))


def parse(out):
    """Pull the numbers out of full_sys_v3.py's report."""
    def grab(pat, cast=float):
        m = re.search(pat, out)
        return cast(m.group(1)) if m else None
    return {
        "exact":  grab(r"exact \(0 LSB\)\s*:\s*([\d.]+)"),
        "w1":     grab(r"within 1 LSB\s*:\s*([\d.]+)"),
        "mean":   grab(r"mean\s*:\s*([\d.]+)"),
        "max":    grab(r"max\s*:\s*(\d+)", int),
        "gt2":    grab(r"> 2 LSB\s*:\s*([\d.]+)"),
        "rmse":   grab(r"RMSE\s*:\s*([\d.]+)"),
    }


def main():
    args     = [a for a in sys.argv[1:] if not a.startswith("--")]
    no_gold  = "--no-golden" in sys.argv

    if args:
        imgs = [os.path.join(IMGDIR, a if a.endswith(".hex") else a + ".hex")
                for a in args]
    else:
        imgs = sorted(os.path.join(IMGDIR, f) for f in os.listdir(IMGDIR)
                      if f.endswith(".hex"))
        orig = os.path.join(DATA, "image_orig.hex")
        if os.path.exists(orig):
            imgs.append(orig)

    print(f"{len(imgs)} images\n")
    results = []

    for p in imgs:
        name = os.path.splitext(os.path.basename(p))[0]
        print(f"=== {name}")

        # 1. banks (+ data_files/image.hex, kept in sync automatically)
        if run([PY, os.path.join(HELPER, "gen_rom_banks.py"), p]) is None:
            continue

        # 2. golden -- cached, because this is the slow one
        if have_cache(name):
            print("    golden: cached")
            load_cache(name)
        elif no_gold:
            print("    golden: MISSING and --no-golden set, skipping")
            continue
        else:
            print("    golden: generating (slow, 65536 forward passes)...")
            if run([PY, os.path.join(HELPER, "gen_full_golden.py"),
                    "--image", os.path.join(DATA, "image.hex")]) is None:
                continue
            save_cache(name)

        # 3. RTL
        print("    sim...")
        if run(SIM) is None:
            continue

        # 4. compare
        out = run([PY, os.path.join(HELPER, "full_sys_v3.py")])
        if out is None:
            continue
        r = parse(out)
        r["name"] = name
        results.append(r)
        print(f"    mean {r['mean']}  max {r['max']}  >2LSB {r['gt2']}%")

    # ---- summary ----
    print("\n" + "=" * 72)
    print(f"  {'image':<20}{'exact%':>8}{'<=1LSB%':>9}{'mean':>8}"
          f"{'max':>6}{'>2LSB%':>9}{'RMSE':>8}")
    print("  " + "-" * 68)
    for r in results:
        print(f"  {r['name']:<20}{r['exact']:>8.2f}{r['w1']:>9.2f}"
              f"{r['mean']:>8.3f}{r['max']:>6d}{r['gt2']:>9.2f}{r['rmse']:>8.3f}")

    if results:
        worst = max(results, key=lambda r: r["gt2"])
        print("  " + "-" * 68)
        print(f"  WORST: {worst['name']}  "
              f"mean {worst['mean']:.3f}  max {worst['max']}  "
              f">2LSB {worst['gt2']:.2f}%")
    print("=" * 72)
    print("""
  Judge on the WORST row, not the average -- one bad image is a bad design.
  1 LSB = 1/255 = 0.39% brightness. mean under ~1 LSB is below the sensor
  noise already in the source and is not visible.
""")


if __name__ == "__main__":
    main()