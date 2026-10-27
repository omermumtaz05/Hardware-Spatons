#!/usr/bin/env python3
"""Golden vectors for fp32_recip_fixed: '<x_hex> <recip_hex>' per line.

Reference is numpy float32 division, i.e. correctly-rounded 1/x in fp32.
Covers mantissa edges (all-zeros, all-ones, +-1 ulp around them), exponent
sweep, the values act_quant actually sees (max_abs of an activation vector),
and a big random spray.
"""
import struct
import numpy as np

f2h = lambda v: struct.unpack('<I', struct.pack('<f', np.float32(v)))[0]
h2f = lambda h: np.float32(struct.unpack('<f', struct.pack('<I', h))[0])

vals = set()

# mantissa corners at a few exponents
for e in (110, 120, 126, 127, 128, 130, 140):
    for m in (0, 1, 2, 0x400000, 0x7FFFFE, 0x7FFFFF, 0x3FFFFF, 0x400001):
        vals.add((e << 23) | m)

# full exponent sweep at three mantissas
for e in range(1, 255):
    for m in (0, 0x2AAAAA, 0x7FFFFF):
        vals.add((e << 23) | m)

# values act_quant realistically sees as abs_max
for v in (1.0, 2.0, 0.5, 3.0, 1/3, 2047.0, 1/2047.0, 6.0221e23, 1.4e-8,
          0.99999994, 1.00000012, 127.5, 1e-30, 1e30):
    vals.add(f2h(v))

rng = np.random.default_rng(20260724)
for v in rng.uniform(1e-6, 1e6, 20000):
    vals.add(f2h(v))
for v in np.exp(rng.uniform(-60, 60, 20000)):
    vals.add(f2h(v))

with open('recip_vec.txt', 'w') as fh:
    n = 0
    for h in sorted(vals):
        x = h2f(h)
        if not np.isfinite(x) or x == 0:
            continue
        r = np.float32(1.0) / x
        if not np.isfinite(r) or r == 0:
            continue                      # skip over/underflow, tested separately
        if (f2h(r) >> 23) & 0xFF == 0:
            continue                      # denormal result: RTL flushes to zero
        fh.write(f"{h:08x} {f2h(r):08x}\n")
        n += 1
print(f"wrote {n} vectors to recip_vec.txt")