#!/usr/bin/env python3
"""power_image.py -- swap the image baked into the int_v4 netlist's input ROMs (no recompile).
  show  [out.png]          rebuild the image currently in db/*.hdl.mif (verify bank mapping)
  pick  <corpus>           rank images by texture in warped rows 5-10 (the power window), copy 90th pct
  write <image.png>        overwrite the 4 input-ROM MIFs with this image (originals -> *.mif.orig)
"""
import re, glob, os, sys, shutil, numpy as np
from PIL import Image
DB = "synth/top_full/db"
BANKS = [(0, 0), (0, 1), (1, 0), (1, 1)]          # ram0..3 -> ee, eo, oe, oo
RAD = {"BIN": 2, "HEX": 16, "OCT": 8, "DEC": 10, "UNS": 10}

def mifs():
    fs = [glob.glob(f"{DB}/int_v4.ram{r}_input_buffer_parity_int_*.hdl.mif") for r in range(4)]
    assert all(len(f) == 1 for f in fs), fs
    return [f[0] for f in fs]

def hdr(txt):
    h = txt.split("CONTENT")[0]
    g = lambda k: re.search(k + r"\s*=\s*(\w+)\s*;", h, re.I).group(1).upper()
    return h, int(g("WIDTH")), int(g("DEPTH")), RAD[g("ADDRESS_RADIX")], RAD[g("DATA_RADIX")]

def read(p):
    txt = open(p).read(); h, W, D, ar, dr = hdr(txt)
    d = np.zeros(D, np.uint32)
    body = txt.split("BEGIN", 1)[1].split("END", 1)[0]
    for m in re.finditer(r"(\[\s*(\w+)\s*\.\.\s*(\w+)\s*\]|(\w+))\s*:\s*([^;]+);", body):
        vals = [int(v, dr) for v in m.group(5).split()]
        if m.group(2): d[int(m.group(2), ar):int(m.group(3), ar) + 1] = vals[0]
        else:
            a = int(m.group(4), ar); d[a:a + len(vals)] = vals
    return d, h, W, D, ar, dr

def fmt(v, r, W):
    return format(v, f"0{W}b") if r == 2 else format(v, "X") if r == 16 else format(v, "o") if r == 8 else str(v)

def show(out="results/power/rom_now.png"):
    img = np.zeros((256, 256, 3), np.uint8)
    for p, (y, x) in zip(mifs(), BANKS):
        d = read(p)[0][:128 * 128].reshape(128, 128)
        img[y::2, x::2] = np.stack([(d >> 16) & 255, (d >> 8) & 255, d & 255], -1)
    os.makedirs(os.path.dirname(out), exist_ok=True); Image.fromarray(img).save(out); print("saved", out)

def pick(corpus):
    sys.path.insert(0, "helper/sweep"); import golden_vec as gv
    corpus = os.path.expanduser(corpus); lx, ly = gv.load_lut_fp32("data_files/lut_fp32.hex")
    names = sorted(f for f in os.listdir(corpus) if f.endswith(".png"))[:2000]
    sc = []
    for n in names:
        w = gv.warp(np.asarray(Image.open(os.path.join(corpus, n)).convert("RGB"), np.uint8), lx, ly)[5:11]
        sc.append((np.abs(np.diff(w, axis=1)).mean() + np.abs(np.diff(w, axis=0)).mean(), n))
    sc.sort(); g, n = sc[int(0.9 * len(sc))]
    print(f"pick {n}  score {g:.4f}  (median {sc[len(sc)//2][0]:.4f}, max {sc[-1][0]:.4f})")
    shutil.copy(os.path.join(corpus, n), "results/power/power_input_natural.png")

def write(png):
    img = np.asarray(Image.open(png).convert("RGB"), np.uint32); assert img.shape == (256, 256, 3)
    for p, (y, x) in zip(mifs(), BANKS):
        if not os.path.exists(p + ".orig"): shutil.copy(p, p + ".orig")
        _, h, W, D, ar, dr = read(p + ".orig"); assert D == 128 * 128, f"DEPTH {D}, stop"
        s = img[y::2, x::2].reshape(-1, 3); v = (s[:, 0] << 16) | (s[:, 1] << 8) | s[:, 2]
        with open(p, "w") as f:
            f.write(h + "CONTENT BEGIN\n")
            f.writelines(f"\t{fmt(i, ar, 0)} : {fmt(int(x), dr, W)};\n" for i, x in enumerate(v))
            f.write("END;\n")
    print("MIFs written; originals kept as *.mif.orig")

if __name__ == "__main__":
    {"show": show, "pick": pick, "write": write}[sys.argv[1]](*sys.argv[2:])