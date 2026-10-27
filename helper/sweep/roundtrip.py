#!/usr/bin/env python3
"""roundtrip.py -- clean -> RTL pre-distortion -> AR display model -> what the viewer sees.

Per image, three paths through the AR model (ar_distort.distort, #86 config):
  uncorrected : clean            -> AR
  fp32        : clean -> fp32 SW -> AR
  rtl         : clean -> RTL     -> AR      (RTL output from run_sweep.py --keep-dut)

Metrics (inside the AR valid mask = non-black region of AR(white)):
  rtl_vs_fp32 : PSNR(AR(rtl), AR(fp32))  -- does the hardware look like the software through the glasses
  *_vs_clean  : PSNR vs clean, needs Javad's expected-output definition (see TARGET below)

Output: <name>_1_clean.png, <name>_2_rtl_predist.png, <name>_3_rtl_seen.png per image

USAGE (repo root, WSL)
  python3 helper/sweep/run_sweep.py --corpus ~/corpus_rt --out results/rt.csv --keep-dut
  python3 helper/sweep/roundtrip.py --corpus ~/corpus_rt --dut results/dut
"""
import argparse, os, sys
import numpy as np, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden_vec as gv, ar_distort as A

T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0
U8 = lambda t: (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)

def psnr(a, b, m):
    mse = (((a - b) ** 2) * m).sum().item() / (m.sum().item() * a.shape[0])
    return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="dir of clean 256x256 PNGs")
    ap.add_argument("--dut", default="results/dut", help="RTL outputs from run_sweep --keep-dut")
    ap.add_argument("--weights", default="#86-NN2_best.pth")
    ap.add_argument("--lut", default="data_files/lut_fp32.hex")
    ap.add_argument("--out", default="results/roundtrip")
    a = ap.parse_args()

    ar = A.config_from_checkpoint(a.weights)              # exact pipelines stored in the checkpoint
    g = gv.Golden(gv.load_weights_pth(a.weights), a.lut)
    mask = (A.distort(torch.ones(3, 256, 256), ar).mean(0, keepdim=True) > 0.01).float()
    os.makedirs(a.out, exist_ok=True)

    names = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.expanduser(a.corpus))
                   if f.endswith(".png"))
    rows = []
    for n in names:
        dp = os.path.join(a.dut, n + ".png")
        if not os.path.exists(dp):
            print(f"  skip {n}: no RTL output at {dp}"); continue
        clean = np.asarray(Image.open(os.path.join(os.path.expanduser(a.corpus), n + ".png")).convert("RGB"))
        rtl = np.asarray(Image.open(dp).convert("RGB"))
        fp = (g.golden_fp32(clean) * 255 + 0.5).astype(np.uint8)

        s_unc, s_rtl, s_fp = (A.distort(T(x), ar) for x in (clean, rtl, fp))
        r = dict(name=n, rtl_vs_fp32=psnr(s_rtl, s_fp, mask), unc_vs_fp32=psnr(s_unc, s_fp, mask))
        rows.append(r)
        print(f"  {n:28} seen RTL vs seen fp32 {r['rtl_vs_fp32']:6.2f} dB   "
              f"(uncorrected vs fp32 {r['unc_vs_fp32']:6.2f} dB)")
        for tag, im in (("1_clean", clean), ("2_rtl_predist", rtl), ("3_rtl_seen", U8(s_rtl))):
            Image.fromarray(im).save(os.path.join(a.out, f"{n}_{tag}.png"))
    print(f"saved 3 PNGs per image -> {a.out}/")

if __name__ == "__main__":
    main()
