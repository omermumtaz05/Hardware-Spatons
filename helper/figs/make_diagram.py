#!/usr/bin/env python3
"""
Generates the diagram set for the image-distortion-corrector presentation.

Standalone SVGs: self-contained styling, no external fonts or CSS variables,
so they render identically in PowerPoint, Keynote, Google Slides, Inkscape
and a browser. Edit the numbers here and re-run rather than editing the SVGs.

    python3 make_diagrams.py [outdir]
"""

import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- palette
INK    = "#1a1a1a"
MUTED  = "#6b6b6b"
FAINT  = "#9a9a9a"
BORDER = "#c4c4c4"
SURF   = "#f6f6f4"

RAMPS = {
    "blue":  ("#dce7f5", "#5f88c0"),
    "teal":  ("#d5ecea", "#4e9d98"),
    "amber": ("#f8e7cd", "#c9944a"),
    "red":   ("#f8dcdc", "#c47474"),
    "green": ("#ddefdd", "#67a367"),
    "grey":  (SURF,      BORDER),
}

STYLE = f"""
  <style>
    .t   {{ font: 500 14px 'Helvetica Neue', Helvetica, Arial, sans-serif; fill: {INK}; }}
    .tr  {{ font: 400 14px 'Helvetica Neue', Helvetica, Arial, sans-serif; fill: {INK}; }}
    .ts  {{ font: 400 12px 'Helvetica Neue', Helvetica, Arial, sans-serif; fill: {MUTED}; }}
    .tsi {{ font: 400 12px 'Helvetica Neue', Helvetica, Arial, sans-serif; fill: {INK}; }}
    .tm  {{ font: 400 12px 'SF Mono', Menlo, Consolas, monospace; fill: {MUTED}; }}
    .tmi {{ font: 400 12px 'SF Mono', Menlo, Consolas, monospace; fill: {INK}; }}
    .ttl {{ font: 500 16px 'Helvetica Neue', Helvetica, Arial, sans-serif; fill: {INK}; }}
    .arr {{ stroke: {MUTED}; stroke-width: 1; fill: none; }}
    .arrf{{ stroke: {FAINT}; stroke-width: 1; fill: none; stroke-dasharray: 4 3; }}
    .box {{ fill: {SURF}; stroke: {BORDER}; stroke-width: 1; }}
    .dash{{ fill: none; stroke: {FAINT}; stroke-width: 1; stroke-dasharray: 5 4; }}
    .reg {{ stroke: {FAINT}; stroke-width: 1; stroke-dasharray: 3 3; }}
  </style>
"""

DEFS = """
  <defs>
    <marker id="a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6"
            orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5"
            stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
"""


def svg(name, w, h, body, title, desc):
    doc = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img">\n'
        f'  <title>{title}</title>\n  <desc>{desc}</desc>\n'
        f'{STYLE}{DEFS}\n{body}\n</svg>\n'
    )
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print("wrote", name)


def box(x, y, w, h, title, sub=None, ramp="grey", rx=4, mono=False):
    fill, stroke = RAMPS[ramp]
    cx = x + w / 2
    cls = "tmi" if mono else "t"
    if sub:
        ty, sy = y + h / 2 - 3, y + h / 2 + 15
    else:
        ty, sy = y + h / 2 + 5, None
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
         f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>\n'
         f'<text class="{cls}" x="{cx}" y="{ty}" text-anchor="middle">{title}</text>\n')
    if sub:
        s += f'<text class="ts" x="{cx}" y="{sy}" text-anchor="middle">{sub}</text>\n'
    return s


def arrow(x1, y1, x2, y2, faint=False):
    c = "arrf" if faint else "arr"
    return f'<line class="{c}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" marker-end="url(#a)"/>\n'


def path(d, faint=False):
    c = "arrf" if faint else "arr"
    return f'<path class="{c}" d="{d}" marker-end="url(#a)"/>\n'


def label(x, y, s, cls="ts", anchor="start"):
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{s}</text>\n'


def legend(x, y, items):
    s = ""
    for i, (ramp, text) in enumerate(items):
        fill, stroke = RAMPS[ramp]
        bx = x + i * 170
        s += (f'<rect x="{bx}" y="{y - 10}" width="12" height="12" rx="2" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>\n')
        s += label(bx + 18, y, text)
    return s


# ==================================================================== 01
def d01_top_level():
    b = label(40, 40, "top_full &#8212; system dataflow", "ttl")
    rows = [
        (64,  "top_geo",            "1 pixel per cycle, stallable",   "blue"),
        (154, "inter_line_buffer",  "4 lines, 3-line window",         "grey"),
        (244, "patch_extractor",    "3&#215;3 window &#8594; 29 features", "grey"),
        (334, "photo_stage_int",    "A12 MLP, 90 cycles per pixel",   "teal"),
        (424, "output_buffer",      "final RGB at y&#183;W + x",      "grey"),
    ]
    for y, t, s, r in rows:
        b += box(220, y, 280, 56, t, s, r)
    for y in (120, 210, 300, 390):
        b += arrow(360, y, 360, y + 30)
    b += path("M220 182 H160 V92 H216")
    b += label(150, 141, "stall, per line", "ts", "end")
    b += label(520, 141, "1 pixel / cycle", "tsi")
    b += label(520, 315, "1 pixel / 90 cycles", "tsi")
    b += label(520, 333, "this gap is why the")
    b += label(520, 349, "buffer exists")
    b += legend(220, 508, [("blue", "geometric stage"), ("teal", "photometric stage")])
    svg("01_top_level.svg", 720, 540, b,
        "top_full system dataflow",
        "Five blocks in a vertical chain with a per-line stall feedback path "
        "and the one-pixel-per-cycle versus one-per-ninety-cycles rate mismatch.")


# ==================================================================== 02
def d02_geo_stage():
    b = label(40, 40, "Geometric stage &#8212; two registered cycles", "ttl")
    # cycle boundaries
    for x in (250, 500):
        b += f'<line class="reg" x1="{x}" y1="64" x2="{x}" y2="330"/>\n'
    b += label(145, 80, "cycle 0", "ts", "middle")
    b += label(375, 80, "cycle 1", "ts", "middle")
    b += label(625, 80, "cycle 2", "ts", "middle")

    b += box(40, 96, 190, 52, "coord_gen", "x, y sweep", "blue")
    b += box(268, 96, 200, 52, "LUT_quant", "registered read", "amber")
    b += box(268, 166, 200, 52, "coordsplit_format", "x_int, dx", "blue")
    b += box(268, 236, 200, 52, "fetch_four_near", "A00 &#8230; A11", "blue")
    b += box(518, 96, 140, 52, "input_buffer", "4 parity banks", "amber")
    b += box(518, 166, 140, 52, "bilinear_interp", "combinational", "red")

    b += arrow(230, 122, 264, 122)
    b += path("M368 148 V162")
    b += path("M368 218 V232")
    b += path("M468 262 H492 V122 H514")
    b += path("M588 148 V162")
    b += arrow(658, 192, 690, 192)
    b += label(694, 196, "P_out[3]", "tmi")

    # delay-matched control chain
    b += f'<rect class="dash" x="40" y="316" width="700" height="86" rx="4"/>\n'
    b += label(52, 336, "delay-matched control &#8212; must track the two register stages", "ts")
    b += box(60, 348, 150, 38, "x_ind, y_ind", None, "grey", mono=True)
    b += box(268, 348, 150, 38, "_1", None, "grey", mono=True)
    b += box(518, 348, 150, 38, "_2 &#8594; out_x/out_y", None, "grey", mono=True)
    b += arrow(210, 367, 264, 367)
    b += arrow(418, 367, 514, 367)
    b += label(40, 428, "valid_1 / valid_2 and row_last carry the same two-stage delay.", "ts")
    b += label(40, 446, "stall gates every register above; the stage freezes as a unit.", "ts")
    b += legend(40, 480, [("amber", "registered"), ("red", "critical path"), ("blue", "combinational")])
    svg("02_geo_stage.svg", 800, 510, b,
        "Geometric stage pipeline",
        "Datapath across three cycle columns with two register boundaries, plus the "
        "delay-matched coordinate and valid chain beneath it.")


# ==================================================================== 03
def d03_parity_banks():
    b = label(40, 40, "Parity banking &#8212; four corners, four banks, one cycle", "ttl")
    b += label(40, 64, "bank = {y[0], x[0]}   addr = (y&#187;1)&#183;128 + (x&#187;1)", "tm")

    cell = 34
    ox, oy = 48, 92
    names = {(0, 0): "EE", (0, 1): "EO", (1, 0): "OE", (1, 1): "OO"}
    ramp_of = {"EE": "blue", "EO": "teal", "OE": "amber", "OO": "green"}
    for gy in range(6):
        for gx in range(6):
            nm = names[(gy % 2, gx % 2)]
            fill, stroke = RAMPS[ramp_of[nm]]
            b += (f'<rect x="{ox + gx*cell}" y="{oy + gy*cell}" width="{cell}" height="{cell}" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="0.75"/>\n')
    # highlight one quad, placed at the grid's right edge so the connectors
    # leave into open space instead of crossing the grid
    qx, qy = ox + 4 * cell, oy + 2 * cell
    gr = ox + 6 * cell                      # grid right edge
    b += (f'<rect x="{qx}" y="{qy}" width="{2*cell}" height="{2*cell}" fill="none" '
          f'stroke="{INK}" stroke-width="2.5" rx="2"/>\n')
    b += label(ox + 5 * cell, oy + 6 * cell + 22, "one bilinear quad", "tsi", "middle")

    # bank boxes
    bx = 330
    for i, nm in enumerate(["EE", "EO", "OE", "OO"]):
        y = 96 + i * 60
        b += box(bx, y, 150, 46, f"ROM_{nm}", "128 &#215; 128 &#215; 24b", ramp_of[nm])
    # corner -> bank, elbowed at four distinct x so nothing overlaps
    for (ey, stage, by) in [(170, 268, 119), (184, 284, 179),
                            (204, 300, 239), (218, 316, 299)]:
        b += path(f"M{gr} {ey} H{stage} V{by} H{bx - 6}")

    b += box(530, 156, 180, 106, "", None, "grey")
    b += label(620, 186, "output crossbar", "t", "middle")
    b += label(620, 208, "registered parity", "ts", "middle")
    b += label(620, 226, "routes each bank back", "ts", "middle")
    b += label(620, 244, "to its corner", "ts", "middle")
    for i in range(4):
        b += arrow(480, 119 + i * 60, 526, 200)

    b += label(40, 366, "A bilinear quad is (x,y), (x+1,y), (x,y+1), (x+1,y+1) &#8212; so the four corners", "tsi")
    b += label(40, 386, "always carry four distinct parities. Exactly one lands in each bank, all four", "tsi")
    b += label(40, 406, "read in parallel, and total storage is unchanged. No replication.", "tsi")
    svg("03_parity_banks.svg", 760, 440, b,
        "Parity banking of the source image",
        "A checkerboard of pixel parities showing that any two-by-two bilinear quad "
        "has one corner in each of the four memory banks, allowing four parallel reads.")


# ==================================================================== 04
def d04_line_window():
    b = label(40, 40, "Inter-line buffer &#8212; circular, counted in lines", "ttl")
    lw, lh, step = 260, 40, 50
    for i in range(4):
        y = 80 + i * step
        ramp = "teal" if i < 3 else "grey"
        sub = "spare &#8212; lets geo write ahead" if i == 3 else None
        b += box(40, y, lw, lh, f"line slot {i}", sub, ramp)
    b += (f'<rect x="28" y="68" width="{lw + 24}" height="{2*step + lh + 12}" fill="none" '
          f'stroke="{INK}" stroke-width="2" stroke-dasharray="6 4" rx="6"/>\n')
    b += label(28, 300, "WINDOW = 3 live lines; rd_line_sel picks one of them", "tsi")

    b += path("M300 96 H460")
    b += label(320, 88, "wr_line_valid", "tm")
    b += path("M460 250 H306")
    b += label(320, 242, "rd_line_consume", "tm")

    # 3x3 read sequence
    b += box(470, 76, 300, 190, "", None, "grey")
    b += label(620, 102, "patch_extractor", "t", "middle")
    b += label(620, 120, "9 reads, one per cycle", "ts", "middle")
    c = 32
    gx, gy = 572, 134
    for r in range(3):
        for cc in range(3):
            n = r * 3 + cc
            fill, stroke = RAMPS["blue"] if n == 4 else RAMPS["grey"]
            b += (f'<rect x="{gx + cc*c}" y="{gy + r*c}" width="{c}" height="{c}" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="0.75"/>\n')
            b += label(gx + cc * c + c / 2, gy + r * c + c / 2 + 4, str(n), "ts", "middle")
    b += label(620, 254, "centre &#8594; f[2..4], tapped by the side FIFO", "ts", "middle")

    b += label(40, 340, "Edges reflect without repeating: column &#8722;1 maps to 1, column W maps to W&#8722;2.", "tsi")
    b += label(40, 360, "First and last row need only 2 committed lines, so the extractor cannot deadlock", "tsi")
    b += label(40, 380, "waiting for a line that will never arrive.", "tsi")
    svg("04_line_window.svg", 810, 414, b,
        "Inter-line buffer and patch extractor",
        "Four line slots with a three-line sliding window, and the nine-step raster "
        "read sequence that builds the three-by-three neighbourhood.")


# ==================================================================== 05
def d05_mlp():
    b = label(40, 40, "Photometric stage &#8212; A12 MLP with residual branch", "ttl")
    b += box(40, 92, 110, 50, "f[29]", "features", "blue")

    b += box(200, 74, 120, 44, "layer 1", "29&#8594;48 ReLU", "teal")
    b += box(340, 74, 120, 44, "layer 2", "48&#8594;48 ReLU", "teal")
    b += box(480, 74, 120, 44, "layer 3", "48&#8594;3 linear", "teal")
    b += label(400, 62, "89 of the 90-cycle II lives here", "ts", "middle")

    b += box(200, 160, 120, 44, "residual", "29&#8594;3 linear", "amber")
    b += box(360, 160, 120, 44, "fifo_residual", "depth 4, FWFT", "amber")

    b += arrow(150, 108, 196, 100)
    b += path("M95 142 V182 H196")
    b += arrow(320, 96, 336, 96)
    b += arrow(460, 96, 476, 96)
    b += arrow(320, 182, 356, 182)

    b += box(640, 110, 150, 80, "", None, "grey")
    b += label(715, 134, "combine", "t", "middle")
    b += label(715, 154, "1  integer add", "tm", "middle")
    b += label(715, 170, "2  int &#8594; fp32", "tm", "middle")
    b += label(715, 186, "3  &#215; lambda3", "tm", "middle")
    b += path("M600 96 H620 V140 H636")
    b += path("M480 182 H620 V160 H636")

    b += box(640, 240, 150, 44, "post_mlp", "4&#963;(z)&#8722;2, clamp", "red")
    b += arrow(715, 190, 715, 236)
    b += box(400, 240, 190, 44, "side FIFO", "centre RGB = f[2..4]", "grey")
    b += path("M95 142 V262 H396")
    b += arrow(590, 262, 636, 262)
    b += arrow(790, 262, 830, 262)
    b += label(834, 266, "final_rgb", "tmi")

    b += label(40, 330, "L1 and residual accept each pixel together, since they share f. The residual does not", "tsi")
    b += label(40, 350, "hold until L3 &#8212; it pushes into the FIFO and frees itself, so the input side can take", "tsi")
    b += label(40, 370, "pixel N+1 while N&#8722;2 &#8230; N are still in flight. In-order path + FIFO order means y and", "tsi")
    b += label(40, 390, "y_r are always the same pixel.", "tsi")
    b += legend(40, 424, [("teal", "main path"), ("amber", "residual"), ("red", "combinational")])
    svg("05_mlp.svg", 900, 456, b,
        "A12 MLP topology and pipelining",
        "Main path of three layers alongside a linear residual branch buffered in a FIFO, "
        "recombined in three registered stages and passed to the post-MLP chain.")


# ==================================================================== 06
def d06_layer_fsm():
    b = label(40, 40, "layer_int &#8212; FSM and where the cycles go", "ttl")
    states = [
        ("ACCEPT",  "latch input",        "grey",  1),
        ("RQ_IN",   "requant scan 1",     "blue",  14),
        ("COMPUTE", "neuron MACs",        "teal",  50),
        ("RQ_ACC",  "requant scan 2",     "blue",  14),
        ("SCALE",   "msh = K&#8722;s&#8331;&#8722;s&#8336;", "amber", 1),
        ("APPLY",   "&#215;K, +B, ReLU",  "amber", 6),
        ("HOLD",    "await out_ready",    "grey",  1),
    ]
    x = 40
    for name, sub, ramp, cyc in states:
        w = 112
        b += box(x, 84, w, 52, name, sub, ramp)
        b += label(x + w / 2, 152, f"{cyc} cyc", "ts", "middle")
        if x + w < 830:
            b += arrow(x + w, 110, x + w + 12, 110)
        x += w + 12
    b += path("M886 136 V172 H46 V142", faint=True)
    b += label(460, 188, "out_ready &#8594; next pixel", "ts", "middle")

    b += label(40, 234, "One requant_int instance serves both scans &#8212; same operation, sequential in the FSM.", "tsi")
    b += label(40, 254, "All neurons start on one pulse and share the activation mux, so no neuron can stall.", "tsi")
    b += label(40, 274, "Counts shown for the 48&#8594;48 layer at NUM_FOLDS = 1, RQ_FOLDS = 4, AP_FOLDS = 8.", "ts")
    svg("06_layer_fsm.svg", 900, 306, b,
        "layer_int finite state machine",
        "Seven states from ACCEPT through HOLD with per-state cycle costs, showing "
        "the neuron MAC sweep dominating the layer latency.")


# ==================================================================== 07
def d07_m_chain():
    b = label(40, 40, "Why the DSP budget closed &#8212; the M chain collapses", "ttl")
    b += label(40, 72, "before &#8212; runtime scalar M", "t")
    b += label(490, 72, "after &#8212; constant and a shift", "t")
    b += f'<line class="reg" x1="466" y1="60" x2="466" y2="352"/>\n'

    before = [
        ("maxacc &#215; X",               "32&#215;32 multiply",  "red"),
        ("normalise(G, 24b)",             "priority encoder",     "red"),
        ("gmant &#215; K_MANT",           "24&#215;18 multiply",  "red"),
        ("m_fx &#187; (K &#8722; gexp)",  "barrel shift",         "amber"),
        ("(racc &#215; m_fx) &#187; MSH", "+ bias",               "amber"),
    ]
    for i, (t, s, r) in enumerate(before):
        y = 92 + i * 52
        b += box(40, y, 250, 38, t, None, r)
        b += label(300, y + 24, s, "ts")
        if i < 4:
            b += arrow(165, y + 38, 165, y + 48)

    after = [
        ("s_in, s_acc are exponents",           "requant normalises by 2&#8319;"),
        ("msh = K_SHIFT &#8722; s_in &#8722; s_acc", "one subtract, registered"),
        ("(racc &#215; K_MANT) &#187; msh",     "constant multiply, in ALMs"),
    ]
    for i, (t, s) in enumerate(after):
        y = 92 + i * 76
        b += box(490, y, 300, 38, t, None, "green")
        b += label(492, y + 54, s, "ts")
        if i < 2:
            b += arrow(640, y + 60, 640, y + 68)

    b += (f'<rect x="40" y="368" width="750" height="76" rx="4" fill="{SURF}" '
          f'stroke="{BORDER}" stroke-width="1"/>\n')
    b += label(56, 394, "M = 2^(s_in + s_acc) &#215; K_MANT / 2^K_SHIFT", "tmi")
    b += label(56, 418, "Both multiplies, the normaliser and the m_fx register are deleted. For layer 1 and the", "ts")
    b += label(56, 434, "residual K_MANT is a power of two, so the multiply collapses to a wire.", "ts")

    b += label(40, 476, "DSP blocks: 146 reported &#8594; ~51 after. Initiation interval: 151 &#8594; 90 cycles per pixel.", "tsi")
    b += label(40, 498, "Enabled by one change &#8212; requant_int normalising by a power of two, not the exact abs-max.", "ts")
    svg("07_m_chain.svg", 840, 528, b,
        "Collapse of the layer scale chain",
        "Side by side comparison of the old runtime scalar chain with two multipliers "
        "and a normaliser against the new constant-and-shift form.")


# ==================================================================== 08
def d08_critical_path():
    b = label(40, 40, "The critical path &#8212; 241 ns against a 20 ns budget", "ttl")
    b += label(40, 64, "top_geo|dx_dd[16]  &#8594;  inter_line_buffer|mem|porta_datain_reg0", "tm")

    scale = 2.55       # px per ns
    x0, y0 = 40, 100
    ops = [("fp32_sub", "1 &#8722; dx"), ("fp32_mul", "weight"), ("fp32_mul", "pixel"),
           ("fp32_add", "pair sum"), ("fp32_add", "final")]
    x = x0
    for name, sub in ops:
        w = 48.25 * scale
        b += box(x, y0, w - 4, 54, name, sub, "red")
        x += w
    total = 241.262 * scale
    b += (f'<line class="arr" x1="{x0}" y1="{y0 + 70}" x2="{x0 + total}" y2="{y0 + 70}" '
          f'marker-end="url(#a)"/>\n')
    b += label(x0 + total / 2, y0 + 90, "241.262 ns of combinational logic", "tsi", "middle")

    # clock budget bar
    cy = y0 + 120
    b += (f'<rect x="{x0}" y="{cy}" width="{20*scale}" height="30" rx="3" '
          f'fill="{RAMPS["green"][0]}" stroke="{RAMPS["green"][1]}" stroke-width="1"/>\n')
    b += label(x0 + 20 * scale + 12, cy + 20, "20 ns clock period (50 MHz constraint)", "tsi")
    b += (f'<rect x="{x0}" y="{cy + 44}" width="{total}" height="30" rx="3" '
          f'fill="none" stroke="{RAMPS["red"][1]}" stroke-width="1.5" stroke-dasharray="5 4"/>\n')
    b += label(x0 + 8, cy + 64, "actual &#8212; slack &#8722;221.4 ns, Fmax 4.14 MHz", "tsi")

    b += label(40, cy + 118, "Each fp32 op is align-shift &#8594; add &#8594; leading-zero-detect &#8594; normalise &#8594; round:", "tsi")
    b += label(40, cy + 138, "four sequential barrel shifters in soft logic, about 48 ns each. dx is worse than dy", "tsi")
    b += label(40, cy + 158, "because dy only enters at the weight multiply, making its path four deep, not five.", "tsi")

    b += (f'<rect x="40" y="{cy + 180}" width="{700}" height="76" rx="4" fill="{SURF}" '
          f'stroke="{BORDER}" stroke-width="1"/>\n')
    b += label(56, cy + 204, "The fix is nearly free.", "t")
    b += label(56, cy + 224, "II is 90 cycles and geo runs at 1 pixel/cycle, so geo is stalled 89 cycles in 90.", "ts")
    b += label(56, cy + 242, "N pipeline stages cost N cycles against 5,898,240 per frame &#8212; and zero throughput.", "ts")
    svg("08_critical_path.svg", 780, cy + 296, b,
        "Critical path against the clock budget",
        "Five chained floating-point operations totalling 241 nanoseconds drawn to scale "
        "beside the 20 nanosecond clock period they must fit inside.")


# ==================================================================== 09
def d09_results():
    b = label(40, 40, "Results &#8212; 5CEFA7F31I7, fitted", "ttl")

    # utilisation bars
    b += label(40, 76, "Utilisation", "t")
    util = [("ALMs", 37892, 56480), ("RAM blocks", 520, 686),
            ("Memory bits", 4164688, 7024640), ("DSP", 127, 156), ("Pins", 101, 480)]
    bw = 260
    for i, (name, used, avail) in enumerate(util):
        y = 96 + i * 34
        frac = used / avail
        b += label(40, y + 14, name, "ts")
        b += (f'<rect x="140" y="{y}" width="{bw}" height="18" rx="3" fill="#ffffff" '
              f'stroke="{BORDER}" stroke-width="1"/>\n')
        fill, stroke = RAMPS["amber"] if frac > 0.7 else RAMPS["blue"]
        b += (f'<rect x="140" y="{y}" width="{bw*frac:.1f}" height="18" rx="3" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>\n')
        b += label(410, y + 14, f"{frac*100:.0f}%", "tsi")

    # DSP before / after
    b += label(470, 76, "DSP blocks in the MLP", "t")
    px = 1.1         # px per block
    for i, (name, val, ramp) in enumerate([("reported", 146, "red"), ("after", 51, "green")]):
        y = 96 + i * 34
        b += label(470, y + 14, name, "ts")
        fill, stroke = RAMPS[ramp]
        b += (f'<rect x="540" y="{y}" width="{val*px:.1f}" height="18" rx="3" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>\n')
        b += label(540 + val * px - 8, y + 14, str(val), "tsi", "end")
    # device limits as reference rules
    for lim, txt in [(87, "DE1-SoC 87"), (156, "A7 156")]:
        x = 540 + lim * px
        b += (f'<line x1="{x:.1f}" y1="88" x2="{x:.1f}" y2="176" stroke="{FAINT}" '
              f'stroke-width="1" stroke-dasharray="4 3"/>\n')
        b += label(x, 190, txt, "ts", "middle")

    b += label(470, 214, "what came out", "t")
    removed = [("requant reciprocal", 20), ("layer 32&#215;32", 24),
               ("layer 24&#215;18", 4), ("apply &#8594; ALMs", 16)]
    for i, (name, val) in enumerate(removed):
        y = 236 + i * 20
        b += label(470, y, name, "ts")
        b += label(700, y, f"&#8722;{val}", "tsi", "end")
    b += (f'<line x1="470" y1="{236 + 4*20 - 12}" x2="700" y2="{236 + 4*20 - 12}" '
          f'stroke="{BORDER}" stroke-width="1"/>\n')
    b += label(470, 236 + 4 * 20 + 4, "total removed", "tsi")
    b += label(700, 236 + 4 * 20 + 4, "&#8722;64", "tsi", "end")

    # FPS
    b += label(40, 310, "Frame rate is Fmax-bound, not II-bound", "t")
    fps = [("4.14 MHz  (today)", 0.70, "red"), ("20 MHz  (pipelined)", 3.4, "amber"),
           ("50 MHz", 8.5, "blue"), ("100 MHz", 17.0, "green")]
    for i, (name, v, ramp) in enumerate(fps):
        y = 330 + i * 30
        b += label(40, y + 14, name, "ts")
        fill, stroke = RAMPS[ramp]
        b += (f'<rect x="200" y="{y}" width="{v*22:.1f}" height="18" rx="3" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>\n')
        b += label(200 + v * 22 + 8, y + 14, f"{v} fps", "tsi")

    b += label(40, 470, "FPS = f_clk / (65536 &#215; 90). II = 90 is a property of layer 2, not of the device.", "ts")
    b += label(40, 490, "Past 17 fps the only lever left is NUM_FOLDS on layer 2 &#8212; affordable on the A7 in a way", "ts")
    b += label(40, 508, "it was not on the DE1-SoC.", "ts")
    svg("09_results.svg", 820, 540, b,
        "Fitted results on the Cyclone V A7",
        "Resource utilisation bars, a waterfall showing DSP blocks removed by the "
        "requantisation rewrite, and frame rate as a function of achievable clock.")


if __name__ == "__main__":
    d01_top_level()
    d02_geo_stage()
    d03_parity_banks()
    d04_line_window()
    d05_mlp()
    d06_layer_fsm()
    d07_m_chain()
    d08_critical_path()
    d09_results()
    print("\ndone ->", OUT)