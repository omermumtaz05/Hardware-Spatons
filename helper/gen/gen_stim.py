#!/usr/bin/env python3
"""
gen_stim.py -- worst-case stimulus for the requant calibration run.

Emits 8-bit feature vectors (IN_DIM=29) chosen to DRIVE THE ACCUMULATORS
TO THEIR CEILING, which is what you need for measuring max|out_vec|.

NOT random. Random pixel values produce the SMALLEST accumulators -- random
signs cancel, so the sum stays near sqrt(48) instead of 48. A random image
makes you measure a small max, pick a small shift, and overflow on the first
real photo.

WHAT IT GENERATES
  48 x 2   layer1  sign-aligned patterns, + and - direction, one per neuron
   3 x 2   residual  same
   3       flat fields: all 0, all 128, all 255
   4       ramps and split fields
  ------
  ~113 vectors

  Each layer1 pattern maximises ONE neuron's accumulator:
      f[i] = 255 where w[j][i] > 0   (+ direction)
      f[i] = 0   elsewhere
  so feeding all 48 probes every corner of layer 1's output space, which is
  exactly what layer 2's RQ_IN sees.

USAGE
    python3 gen_stim.py [data_files_dir] [out_file]

  Default out_file is stim_worst.hex, one vector per line, 29 two-digit
  hex bytes separated by spaces. Adapt the writer at the bottom if your
  testbench wants a different layout.
"""

import sys
import os

IN_DIM = 29
BW_W   = 10

SOURCES = [
    ("layer1",   "a12_layer1_w_q.hex",   48, IN_DIM),
    ("residual", "a12_residual_w_q.hex",  3, IN_DIM),
]


def read_hex(path, bits):
    toks = []
    with open(path) as f:
        for line in f:
            line = line.split("//")[0].split("#")[0].strip()
            if line:
                toks += line.split()
    out = []
    for t in toks:
        v = int(t, 16)
        if v >= (1 << (bits - 1)):
            v -= (1 << bits)
        out.append(v)
    return out


def main():
    d   = sys.argv[1] if len(sys.argv) > 1 else "data_files"
    out = sys.argv[2] if len(sys.argv) > 2 else "stim_worst.hex"

    vecs = []   # (label, [29 bytes])

    for name, wf, n_neurons, n_inputs in SOURCES:
        p = os.path.join(d, wf)
        if not os.path.exists(p):
            print(f"!! missing {p} -- skipping {name}")
            continue
        flat = read_hex(p, BW_W)
        need = n_neurons * n_inputs
        if len(flat) < need:
            raise SystemExit(f"{p}: got {len(flat)}, need {need}")

        for k in range(n_neurons):
            row = flat[k * n_inputs:(k + 1) * n_inputs]
            # + direction: drive every positive-weight input to 255
            vecs.append((f"{name}_n{k}_pos",
                         [255 if w > 0 else 0 for w in row]))
            # - direction: drive every negative-weight input to 255
            vecs.append((f"{name}_n{k}_neg",
                         [255 if w < 0 else 0 for w in row]))

    # flat fields -- correlated inputs, no cancellation
    for v in (0, 128, 255):
        vecs.append((f"flat_{v}", [v] * IN_DIM))

    # ramps and split fields
    vecs.append(("ramp_up",   [min(255, i * 9) for i in range(IN_DIM)]))
    vecs.append(("ramp_down", [max(0, 255 - i * 9) for i in range(IN_DIM)]))
    vecs.append(("split_lo",  [255] * (IN_DIM // 2) + [0] * (IN_DIM - IN_DIM // 2)))
    vecs.append(("split_hi",  [0] * (IN_DIM // 2) + [255] * (IN_DIM - IN_DIM // 2)))

    with open(out, "w") as f:
        for label, v in vecs:
            f.write("// " + label + "\n")
            f.write(" ".join(f"{b:02x}" for b in v) + "\n")

    print(f"wrote {out}: {len(vecs)} vectors x {IN_DIM} features")
    print("\nfeed these through the TB with the golden comparison DISABLED --")
    print("there is no reference output for them, and you only want max_reg.")
    print("\nexpect the measured max to RISE versus a normal image run. if it")
    print("does not move at all, check that the TB is actually consuming the")
    print("file rather than falling back to its default stimulus.")


if __name__ == "__main__":
    main()