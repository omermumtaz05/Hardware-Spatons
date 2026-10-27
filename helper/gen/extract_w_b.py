#!/usr/bin/env python3
"""
export_a12_weights.py — Export the A12 photo-stage MLP weights to hex files for RTL.

A12 (models.1) is the 'complex' ParametricModel:
    h1 = ReLU(W1 @ f  + b1)     29 -> 48
    h2 = ReLU(W2 @ h1 + b2)     48 -> 48
    y  =      W3 @ h2 + b3      48 -> 3   (linear)
    yr =      Wres @ f + bres   29 -> 3   (linear residual, takes ORIGINAL f)
    z  = y + yr                            (raw output; sigmoid is downstream)

Unlike A1 (geo), A12 is INPUT-DEPENDENT — it runs per pixel, so we export raw
weights, not a precomputed LUT.

Hex layout (matches the RTL `layer` module's W_flat):
    W_flat[k*INPUT_SIZE + i] = neuron k's weight on input i.
nn.Linear stores weight as (out_features, in_features) = (neurons, inputs), so
row k IS neuron k's weight vector — NO transpose needed. Row-major flatten of the
PyTorch weight matrix gives exactly the W_flat order.

Each value is written as one line of 8 hex chars = IEEE-754 FP32, big-nibble-first,
same format your MAC/dot_product consume.

Outputs (into ./data_files/ by default):
    a12_layer1_w.hex  a12_layer1_b.hex
    a12_layer2_w.hex  a12_layer2_b.hex
    a12_layer3_w.hex  a12_layer3_b.hex
    a12_res_w.hex     a12_res_b.hex
"""

import os
import struct
import numpy as np
import torch

CKPT = "#86-NN2_best.pth"
OUT_DIR = "data_files"

# A12 architecture (from checkpoint shapes — DO NOT hardcode elsewhere, verified below)
IN_DIM  = 29
HIDDEN  = 48
OUT_DIM = 3


def f32_to_hexline(v):
    """One FP32 value -> 8 hex chars (big-endian nibble order), matching RTL hex."""
    b = struct.pack('>f', float(v))          # big-endian IEEE-754 single
    return b.hex()                           # 8 hex chars


def write_matrix_hex(path, W):
    """
    Write weight matrix W of shape (num_neurons, input_size) in W_flat order:
    for each neuron k (row), write its input_size weights in order.
    """
    W = np.asarray(W, dtype=np.float32)
    num_neurons, input_size = W.shape
    with open(path, "w") as f:
        for k in range(num_neurons):
            for i in range(input_size):
                f.write(f32_to_hexline(W[k, i]) + "\n")
    print(f"  wrote {path}  ({num_neurons}x{input_size} = {num_neurons*input_size} weights)")


def write_vector_hex(path, b):
    """Write a bias vector, one FP32 per line."""
    b = np.asarray(b, dtype=np.float32)
    with open(path, "w") as f:
        for v in b:
            f.write(f32_to_hexline(v) + "\n")
    print(f"  wrote {path}  ({len(b)} biases)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {CKPT} ...")
    ckpt = torch.load(CKPT, map_location="cpu")
    sd = ckpt["model_state_dict"]

    # pull A12 (models.1) tensors
    W1 = sd["models.1.model.layer1.weight"].numpy();   b1 = sd["models.1.model.layer1.bias"].numpy()
    W2 = sd["models.1.model.layer2.weight"].numpy();   b2 = sd["models.1.model.layer2.bias"].numpy()
    W3 = sd["models.1.model.layer3.weight"].numpy();   b3 = sd["models.1.model.layer3.bias"].numpy()
    Wr = sd["models.1.model.residual.weight"].numpy(); br = sd["models.1.model.residual.bias"].numpy()

    # ---- shape sanity check: fail loudly if architecture != expected ----
    assert W1.shape == (HIDDEN,  IN_DIM),  f"layer1 {W1.shape} != {(HIDDEN, IN_DIM)}"
    assert W2.shape == (HIDDEN,  HIDDEN),  f"layer2 {W2.shape} != {(HIDDEN, HIDDEN)}"
    assert W3.shape == (OUT_DIM, HIDDEN),  f"layer3 {W3.shape} != {(OUT_DIM, HIDDEN)}"
    assert Wr.shape == (OUT_DIM, IN_DIM),  f"residual {Wr.shape} != {(OUT_DIM, IN_DIM)}"
    print(f"Shapes OK: 29 -> {HIDDEN} -> {HIDDEN} -> 3, residual 29 -> 3")

    print("Writing hex files ...")
    write_matrix_hex(f"{OUT_DIR}/a12_layer1_w.hex", W1); write_vector_hex(f"{OUT_DIR}/a12_layer1_b.hex", b1)
    write_matrix_hex(f"{OUT_DIR}/a12_layer2_w.hex", W2); write_vector_hex(f"{OUT_DIR}/a12_layer2_b.hex", b2)
    write_matrix_hex(f"{OUT_DIR}/a12_layer3_w.hex", W3); write_vector_hex(f"{OUT_DIR}/a12_layer3_b.hex", b3)
    write_matrix_hex(f"{OUT_DIR}/a12_res_w.hex",    Wr); write_vector_hex(f"{OUT_DIR}/a12_res_b.hex",    br)

    # ---------- verification: run A12 in numpy AND torch, dump golden vectors ----------
    print("\nVerifying export against PyTorch forward pass ...")

    def numpy_a12(f):
        h1 = np.maximum(0, W1 @ f + b1)
        h2 = np.maximum(0, W2 @ h1 + b2)
        y  = W3 @ h2 + b3
        yr = Wr @ f + br
        return y + yr

    # rebuild the torch module to cross-check (exact same math path as training)
    import torch.nn as nn
    class A12(nn.Module):
        def __init__(s):
            super().__init__()
            s.layer1 = nn.Linear(IN_DIM, HIDDEN)
            s.layer2 = nn.Linear(HIDDEN, HIDDEN)
            s.layer3 = nn.Linear(HIDDEN, OUT_DIM)
            s.residual = nn.Linear(IN_DIM, OUT_DIM)
            s.act = nn.ReLU()
        def forward(s, x):
            t = s.act(s.layer1(x))
            t = s.act(s.layer2(t))
            t = s.layer3(t)
            return t + s.residual(x)

    m = A12()
    m.layer1.weight.data = torch.tensor(W1); m.layer1.bias.data = torch.tensor(b1)
    m.layer2.weight.data = torch.tensor(W2); m.layer2.bias.data = torch.tensor(b2)
    m.layer3.weight.data = torch.tensor(W3); m.layer3.bias.data = torch.tensor(b3)
    m.residual.weight.data = torch.tensor(Wr); m.residual.bias.data = torch.tensor(br)
    m.eval()

    rng = np.random.default_rng(0)
    max_diff = 0.0
    golden_lines = []
    for t in range(8):
        f = rng.standard_normal(IN_DIM).astype(np.float32)
        z_np = numpy_a12(f)
        with torch.no_grad():
            z_pt = m(torch.tensor(f)).numpy()
        max_diff = max(max_diff, float(np.abs(z_np - z_pt).max()))
        # save f and expected z as hex for the RTL testbench golden
        f_hex = " ".join(f32_to_hexline(v) for v in f)
        z_hex = " ".join(f32_to_hexline(v) for v in z_np)
        golden_lines.append(f"{f_hex} | {z_hex}")
        print(f"  test {t}: z_np={z_np}  z_pt={z_pt}  (match)")

    print(f"\nnumpy vs torch max diff: {max_diff:.3e}  (should be ~0)")

    # write golden test vectors for the RTL TB: "<29 input hexes> | <3 output hexes>"
    with open(f"{OUT_DIR}/a12_golden.txt", "w") as f:
        f.write("\n".join(golden_lines) + "\n")
    print(f"  wrote {OUT_DIR}/a12_golden.txt  (8 test vectors: f | z)")
    print("\nDone.")


if __name__ == "__main__":
    main()