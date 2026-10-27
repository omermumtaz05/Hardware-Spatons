#!/usr/bin/env python3
"""
gen_test_images.py -- the 9-image suite in image.hex format.

Emits 256x256x3 uint8 images as one 2-digit hex byte per line, row-major,
RGB interleaved -- the layout gen_full_golden.py's load_image() expects:

    lines = W*H*3,  reshape(H, W, 3)

USAGE
    python3 gen_test_images.py [outdir] [real1.png real2.jpg ...]

  Writes  <outdir>/img_<name>.hex  for each.
  Any extra args are treated as real photo paths and converted (needs Pillow).

WHY THESE NINE
  The requant shift is set by the abs-max of each vector, so what stresses it
  is DYNAMIC RANGE, not visual interest.

    flat_black   shift goes to its floor -- worst case for static precision
    very_dark    same, but not degenerate. THE most informative image here.
    flat_grey    mid-scale, correlated inputs
    flat_white   drives features to 255, ceiling case
    gradient     smooth, every brightness represented
    checkerboard maximal local contrast, adjacent-pixel features fight
    hard_edges   large flat regions with sharp boundaries
    noise        LEAST stressful -- random signs cancel. Included as a control.
    natural      real photo, if you pass one

  Random images are near-useless on their own: uncorrelated inputs make the
  48-term dot products cancel, so accumulators stay near sqrt(48) instead of
  48 and you measure a max that is far too low.
"""

import sys
import os
import numpy as np

W = H = 256


def write_hex(path, img):
    """img: (H, W, 3) uint8 -> one hex byte per line, row-major RGB."""
    flat = img.reshape(-1)
    with open(path, "w") as f:
        f.write("\n".join(f"{v:02x}" for v in flat))
        f.write("\n")
    print(f"  wrote {path}  ({flat.size} lines)")


def flat(v):
    return np.full((H, W, 3), v, np.uint8)


def gradient():
    x = np.linspace(0, 255, W, dtype=np.float64)
    y = np.linspace(0, 255, H, dtype=np.float64)
    g = ((x[None, :] + y[:, None]) / 2).astype(np.uint8)
    img = np.zeros((H, W, 3), np.uint8)
    img[..., 0] = g
    img[..., 1] = g[:, ::-1]
    img[..., 2] = 255 - g
    return img


def checkerboard(cell=1):
    yy, xx = np.mgrid[0:H, 0:W]
    m = (((yy // cell) + (xx // cell)) % 2).astype(np.uint8) * 255
    return np.dstack([m, m, m])


def hard_edges():
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :64] = 0
    img[:, 64:128] = 255
    img[:, 128:192] = 32
    img[:, 192:] = 223
    img[:64, :] = np.roll(img[:64, :], 96, axis=1)
    img[128:192, :] = 255 - img[128:192, :]
    return img


def noise(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (H, W, 3), dtype=np.uint8)


def very_dark(seed=1):
    """Low mean, low variance -- but NOT all zeros. This is the one that
    pushes the dynamic requant shift to its minimum without degenerating."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 12, (H, W, 3)).astype(np.float64)
    yy = np.linspace(0, 6, H)[:, None, None]
    return np.clip(base + yy, 0, 255).astype(np.uint8)


def bright_texture(seed=2):
    """High mean with structure -- correlated inputs near the ceiling."""
    rng = np.random.default_rng(seed)
    base = rng.integers(230, 256, (H, W, 3)).astype(np.float64)
    yy, xx = np.mgrid[0:H, 0:W]
    ripple = 12 * np.sin(xx / 7.0) * np.cos(yy / 9.0)
    return np.clip(base + ripple[..., None], 0, 255).astype(np.uint8)


SYNTH = [
    ("flat_black",     lambda: flat(0)),
    ("very_dark",      very_dark),
    ("flat_grey",      lambda: flat(128)),
    ("flat_white",     lambda: flat(255)),
    ("gradient",       gradient),
    ("checkerboard",   checkerboard),
    ("hard_edges",     hard_edges),
    ("bright_texture", bright_texture),
    ("noise",          noise),
]


def convert_real(path, outdir):
    try:
        from PIL import Image
    except ImportError:
        print(f"  !! Pillow not installed, skipping {path}"
              f"  (pip install pillow)")
        return
    im = Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS)
    name = os.path.splitext(os.path.basename(path))[0]
    write_hex(os.path.join(outdir, f"img_{name}.hex"), np.array(im, np.uint8))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "data_files/test_images"
    reals  = sys.argv[2:]
    os.makedirs(outdir, exist_ok=True)

    print(f"writing to {os.path.abspath(outdir)}")
    for name, fn in SYNTH:
        write_hex(os.path.join(outdir, f"img_{name}.hex"), fn())
    for p in reals:
        convert_real(p, outdir)

    print(f"""
DONE. To sweep them for clipping -- NO GOLDEN NEEDED:

    for f in {outdir}/img_*.hex; do
        cp "$f" data_files/image.hex
        ./obj_dir/Vtop_full_int_st_tb | grep clamps
        echo "  ^^ $f"
    done

Every line must read clamps=0. If one fires, raise that S_IN/S_ACC by the
gap shown in its `needed` field and adjust the matching K_SHIFT by the same
amount in the opposite direction.

Watch `very_dark` and `flat_black` hardest -- those drive the dynamic shift
to its floor, which is where static loses the most precision. `noise` is the
control: if it is your worst case, something is wrong with the measurement.
""")


if __name__ == "__main__":
    main()