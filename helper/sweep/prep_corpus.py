#!/usr/bin/env python3
"""prep_corpus.py -- natural images -> N x 256x256 RGB PNGs + list file.

Center-crops to square, resizes to 256x256 (Lanczos). Seeded random pick, so
the subset is reproducible and citable ("10,000 images from Places365-val, seed 0").

USAGE
  python helper/sweep/prep_corpus.py --src ~/datasets/val_256 --out ~/corpus10k --n 10000
"""
import argparse, os, random
from multiprocessing import Pool
from PIL import Image

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def one(job):
    src, dst = job
    try:
        im = Image.open(src).convert("RGB")
        w, h = im.size; s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        if im.size != (256, 256):
            im = im.resize((256, 256), Image.Resampling.LANCZOS)
        im.save(dst)
        return None
    except Exception as e:
        return f"{src}: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(os.path.expanduser(a.src))
                   for f in fs if f.lower().endswith(EXT))
    if len(files) < a.n:
        raise SystemExit(f"only {len(files)} images in {a.src}, need {a.n}")
    random.Random(a.seed).shuffle(files)
    pick = files[:a.n]

    out = os.path.expanduser(a.out); os.makedirs(out, exist_ok=True)
    jobs = [(p, os.path.join(out, os.path.splitext(os.path.basename(p))[0] + ".png"))
            for p in pick]
    assert len({d for _, d in jobs}) == len(jobs), "duplicate basenames in source"
    with Pool() as pool:
        errs = [e for e in pool.imap_unordered(one, jobs, chunksize=32) if e]
    for e in errs: print("  !!", e)
    with open(os.path.join(out, "list.txt"), "w") as f:
        f.write("\n".join(os.path.basename(d) for _, d in jobs) + "\n")
    print(f"{len(jobs) - len(errs)} images -> {out}  (seed {a.seed}, list.txt written)")

if __name__ == "__main__":
    main()
