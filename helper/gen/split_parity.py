# helper/split_parity.py  <in.hex>
import sys
W = H = 256; HW = W//2
toks = [t for ln in open(sys.argv[1]) for t in ln.split()]
banks = {k: ["000000"]*(HW*HW) for k in ("ee","eo","oe","oo")}
for i in range(W*H):
    y, x = divmod(i, W)
    r, g, b = toks[3*i:3*i+3]
    key = ("ee","eo","oe","oo")[(y & 1)*2 + (x & 1)]
    banks[key][(y>>1)*HW + (x>>1)] = f"{r}{g}{b}"
for k, v in banks.items():
    open(f"data_files/rom_{k}.hex","w").write("\n".join(v) + "\n")