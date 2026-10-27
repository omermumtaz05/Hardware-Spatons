#!/usr/bin/env python3
"""
gen_rom_banks.py -- image.hex -> the four parity ROM banks the geo stage reads.

    rom_ee.hex   even row, even col
    rom_eo.hex   even row, odd  col
    rom_oe.hex   odd  row, even col
    rom_oo.hex   odd  row, odd  col

Each bank is (W/2)*(H/2) words of 24-bit RRGGBB -- one word per pixel, so
fetch_four_nearest_int gets all three channels of all four bilinear
neighbours in a single cycle.

USAGE
    python3 gen_rom_banks.py <source_image.hex> [data_dir]

  Default data_dir is data_files/.

WHY THIS ALSO COPIES image.hex
  The DUT reads ONLY the four banks. gen_full_golden.py / full_sys_v3.py read
  ONLY data_files/image.hex. If those ever disagree, the golden and the DUT
  are looking at different pictures and every error number becomes garbage --
  which looks exactly like a precision regression and is miserable to trace.
  So this script writes both, from one source, in one step. Do not update
  either by hand.
"""

import sys
import os
import shutil

IMG_W = IMG_H = 256

NAMES = {0: "rom_ee.hex", 1: "rom_eo.hex", 2: "rom_oe.hex", 3: "rom_oo.hex"}


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    src      = sys.argv[1]
    data_dir = sys.argv[2] if len(sys.argv) > 2 else "data_files"
    os.makedirs(data_dir, exist_ok=True)

    lines = [l.strip() for l in open(src) if l.strip()]
    want  = IMG_W * IMG_H * 3
    if len(lines) != want:
        raise SystemExit(f"{src}: {len(lines)} lines, expected {want}")

    Q = (IMG_W // 2) * (IMG_H // 2)          # 16384 words per bank
    banks = {b: ["000000"] * Q for b in range(4)}

    for y in range(IMG_H):
        for x in range(IMG_W):
            b    = (y & 1) * 2 + (x & 1)                  # parity -> bank
            half = (y >> 1) * (IMG_W // 2) + (x >> 1)     # slot within bank
            s    = (y * IMG_W + x) * 3
            banks[b][half] = lines[s] + lines[s + 1] + lines[s + 2]   # RRGGBB

    # ---- reversibility check: unpack and compare against the source ----
    recon = ["00"] * want
    for y in range(IMG_H):
        for x in range(IMG_W):
            b    = (y & 1) * 2 + (x & 1)
            half = (y >> 1) * (IMG_W // 2) + (x >> 1)
            w    = banks[b][half]
            s    = (y * IMG_W + x) * 3
            recon[s], recon[s + 1], recon[s + 2] = w[0:2], w[2:4], w[4:6]
    assert recon == lines, "MISMATCH -- packing is wrong, do NOT use these banks"

    for b, fn in NAMES.items():
        p = os.path.join(data_dir, fn)
        with open(p, "w") as f:
            f.write("\n".join(banks[b]) + "\n")

    dst = os.path.join(data_dir, "image.hex")
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copyfile(src, dst)

    print(f"{os.path.basename(src)} -> 4 banks x {Q} words (24-bit RRGGBB) "
          f"+ image.hex   [reversibility OK]")


if __name__ == "__main__":
    main()