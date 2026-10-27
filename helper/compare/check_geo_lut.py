#!/usr/bin/env python3
"""check_geo_lut.py -- independent check that the geometric LUTs match Javad's model.

Builds the reference ONLY from the checkpoint + Javad's ApproachA1 forward (javad.py):
    t   = clamp( layer3(relu(layer2(relu(layer1(c))))) + c , 0, 1)   # identity residual
    out = F.grid_sample(img, 2t-1, bilinear, padding_mode='border', align_corners=True)
No code from gen_* / weights_quant_lut / golden_vec is used to build the reference.

Checks:
  1. lut_fp32.hex  vs  t*(W-1) from the checkpoint      (should be ~1e-5 px, fp32 rounding)
  2. lut_quant.hex vs  t*(W-1)                           (should be <= 1/512 px, Q8.8 rounding)
  3. golden_vec.warp(img, lut_fp32)  vs  Javad's grid_sample on a real image
     -- validates the FP32 reference used by your 10k sweep             (should be ~1e-6)

USAGE (WSL, repo root, needs torch):
  python3 helper/compare/check_geo_lut.py --image results/roundtrip/Places365_val_00004111_1_clean.png
"""
import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

W = H = 256

def lines(p):
    return [l.split("//")[0].strip() for l in open(p) if l.split("//")[0].strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="#86-NN2_best.pth")
    ap.add_argument("--lut-fp32", default="data_files/lut_fp32.hex")
    ap.add_argument("--lut-q88", default="data_files/lut_quant.hex")
    ap.add_argument("--image", required=True)
    a = ap.parse_args()

    # ---- reference coordinates straight from the checkpoint (Javad's forward) ----
    sd = torch.load(a.weights, map_location="cpu")["model_state_dict"]
    L = lambda n: (sd[f"models.0.model.{n}.weight"].double(), sd[f"models.0.model.{n}.bias"].double())
    (W1, b1), (W2, b2), (W3, b3) = L("layer1"), L("layer2"), L("layer3")
    gy, gx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
    c = torch.stack([gx, gy], -1).reshape(-1, 2).double()
    h = torch.relu(c @ W1.T + b1); h = torch.relu(h @ W2.T + b2)
    t = torch.clamp(h @ W3.T + b3 + c, 0.0, 1.0)                 # [H*W, 2] = (x, y)
    ref_px = (t * (W - 1)).reshape(H, W, 2).numpy()            # pixel coordinates

    # ---- 1. fp32 LUT ----
    raw = bytes.fromhex("".join(s.zfill(16) for s in lines(a.lut_fp32)))
    f = np.frombuffer(raw, ">f4").astype(np.float64).reshape(H, W, 2)   # {y, x}
    lut_px = np.stack([f[..., 1], f[..., 0]], -1)
    e1 = np.abs(lut_px - ref_px).max()
    print(f"1. lut_fp32.hex  vs checkpoint : max |err| = {e1:.2e} px   "
          f"{'OK' if e1 < 1e-3 else 'MISMATCH'}")

    # ---- 2. Q8.8 LUT ----
    v = np.array([int(s, 16) for s in lines(a.lut_q88)], dtype=np.int64).reshape(H, W)
    q_px = np.stack([(v & 0xFFFF) / 256.0, ((v >> 16) & 0xFFFF) / 256.0], -1)
    e2 = np.abs(q_px - ref_px).max()
    print(f"2. lut_quant.hex vs checkpoint : max |err| = {e2:.2e} px   "
          f"{'OK (<= 1/512 + eps)' if e2 <= 1/512 + 1e-6 else 'MISMATCH'}")

    # ---- 3. sweep golden warp vs Javad's grid_sample ----
    img = np.array(Image.open(a.image).convert("RGB").resize((W, H)), dtype=np.uint8)
    x = torch.from_numpy(img).double().permute(2, 0, 1)[None] / 255.0
    grid = (2.0 * t - 1.0).reshape(1, H, W, 2)
    jav = F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                        align_corners=True)[0].permute(1, 2, 0).numpy()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sweep"))
    import golden_vec as gv
    lx, ly = gv.load_lut_fp32(a.lut_fp32)
    mine = gv.warp(img, lx, ly).astype(np.float64)
    e3 = np.abs(mine - jav).max()
    print(f"3. golden_vec.warp vs Javad grid_sample : max |err| = {e3:.2e} (0..1 scale) = "
          f"{e3*255:.3f} LSB   {'OK' if e3*255 < 0.5 else 'MISMATCH'}")

if __name__ == "__main__":
    main()