import sys
def chk(path, width, signed=False):
    v = [int(l.split('//')[0].strip(), 16) for l in open(path) if l.strip()]
    hexw = max(len(l.split('//')[0].strip()) for l in open(path) if l.strip())*4
    lo, hi = (-(1 << (width-1)), (1 << (width-1))-1) if signed else (0, (1 << width)-1)
    bad = [x for x in v if not (lo <= (x - (1<<hexw) if signed and x >= (1<<(hexw-1)) else x) <= hi)]
    print(f"{path}: {len(v)} vals, {hexw}-bit literals -> {width}-bit target, {len(bad)} REAL overflows")

chk('data_files/km_layer1.hex', 19)
chk('data_files/km_layer2.hex', 19)
chk('data_files/a12_layer3_w_q.hex', 10, signed=True)
chk('data_files/a12_residual_w_q.hex', 10, signed=True)

for p in ['data_files/a12_layer3_w_q.hex', 'data_files/a12_residual_w_q.hex']:
    v = [int(l.split('//')[0].strip(), 16) for l in open(p) if l.strip()]
    print(f"{p}: n={len(v)}  min=0x{min(v):03x}  max=0x{max(v):03x}")
    print(f"   ambiguous middle band (0x400-0xBFF): "
          f"{sum(1 for x in v if 0x400 <= x <= 0xBFF)}")