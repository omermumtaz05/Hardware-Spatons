#!/usr/bin/env python3
"""run_sweep.py -- N-image RTL accuracy sweep, parallel, resumable.

Per image (inside one worker):
  1. PNG -> 4 parity ROM banks (same packing as gen_rom_banks.py) in the worker's data_files/
  2. run the prebuilt Verilator binary with cwd = worker dir (TB paths are relative)
  3. parse DONE/TIMEOUT + RQ_IN/RQ_ACC need/clamp lines
  4. read data_files/full_sys_st.hex, diff vs golden_q (quantized model) and golden_fp32
  5. one CSV row

No recompiles: ROMs load via $readmemh at runtime. Each worker has its own dir, so
the fixed filenames never collide. Re-running skips images already OK in the CSV.

USAGE (WSL, from repo root, after building obj_dir/Vtop_full_int_st_tb with the VCD block commented out)
  python3 helper/sweep/run_sweep.py --corpus ~/corpus10k --limit 20      # smoke test
  python3 helper/sweep/run_sweep.py --corpus ~/corpus10k                  # full run
  python3 helper/sweep/report.py results/sweep.csv
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")          # one core per worker, no oversubscription

import argparse, csv, re, shutil, subprocess, sys, time
from multiprocessing import Pool, current_process
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden_vec as gv

# files the sweep writes per image -- never shared between workers
PER_IMAGE = {"rom_ee.hex", "rom_eo.hex", "rom_oe.hex", "rom_oo.hex",
             "full_sys_st.hex", "image.hex"}
POINTS = [("l1", "RQ_ACC"), ("l2", "RQ_IN"), ("l2", "RQ_ACC"),
          ("l3", "RQ_IN"), ("l3", "RQ_ACC"), ("res", "RQ_ACC")]
RQ_RE = re.compile(r"\.u_(\w+)\s+(RQ_IN|RQ_ACC)\s+S_(?:IN|ACC)=(\d+)\s+need\s+(-?\d+)\.\.(-?\d+).*?clamps=(\d+)")
DONE_RE = re.compile(r"DONE out_count=(\d+)/(\d+)\s+cyc=(\d+)(?:\s+sink_stall=(\d+)\s+idle=(\d+))?")

def fields():
    f = ["name", "status", "cyc", "sink_stall", "sim_s"]
    for r in ("q", "fp"):
        f += [f"{r}_psnr", f"{r}_rmse", f"{r}_mean", f"{r}_max", f"{r}_gt2", f"{r}_exact"]
    for l, k in POINTS:
        f += [f"{l}_{k}_S", f"{l}_{k}_need", f"{l}_{k}_clamps"]
    return f + ["sat0_dut", "sat0_gold"]

# ---------------- worker ----------------
G = None; WD = None; CFG = None

def init(cfg, weights):
    global G, WD, CFG
    CFG = cfg
    G = gv.Golden(weights, cfg["lut"])
    k = current_process()._identity[0] if current_process()._identity else 0
    WD = os.path.join(cfg["work"], f"w{k:02d}")
    dd = os.path.join(WD, "data_files"); os.makedirs(dd, exist_ok=True)
    for f in os.listdir(cfg["const"]):
        dst = os.path.join(dd, f)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(cfg["const"], f), dst)

def write_roms(img, dd):
    for tag, sub in (("ee", img[0::2, 0::2]), ("eo", img[0::2, 1::2]),
                     ("oe", img[1::2, 0::2]), ("oo", img[1::2, 1::2])):
        w = (sub[..., 0].astype(np.uint32) << 16) | (sub[..., 1].astype(np.uint32) << 8) | sub[..., 2]
        p = os.path.join(dd, f"rom_{tag}.hex")
        if os.path.islink(p): os.unlink(p)          # never write through a link
        with open(p, "w") as f:
            f.write("".join(f"{v:06x}\n" for v in w.reshape(-1)))

def metrics(ref8, dut8, pre):
    e = np.abs(ref8 - dut8); rmse = float(np.sqrt((e.astype(np.float64) ** 2).mean()))
    return {f"{pre}_psnr": round(20 * np.log10(255 / rmse), 3) if rmse > 0 else 99.0,
            f"{pre}_rmse": round(rmse, 4), f"{pre}_mean": round(float(e.mean()), 4),
            f"{pre}_max": int(e.max()), f"{pre}_gt2": round(100 * float((e > 2).mean()), 3),
            f"{pre}_exact": round(100 * float((e == 0).mean()), 3)}

def run_one(job):
    name, path = job
    row = {"name": name, "status": "OK"}
    try:
        img = np.asarray(Image.open(path).convert("RGB"), np.uint8)
        assert img.shape == (256, 256, 3), f"shape {img.shape}"
        dd = os.path.join(WD, "data_files")
        write_roms(img, dd)
        out = os.path.join(dd, CFG["hex"])
        if os.path.lexists(out): os.unlink(out)

        t = time.time()
        r = subprocess.run([CFG["bin"]], cwd=WD, capture_output=True, text=True,
                           timeout=CFG["timeout"])
        row["sim_s"] = round(time.time() - t, 2)
        log = r.stdout + r.stderr

        m = DONE_RE.search(log)
        if not m:
            row["status"] = "TIMEOUT" if "TIMEOUT" in log else f"NODONE rc={r.returncode}"
            return row
        row["cyc"] = int(m.group(3)); row["sink_stall"] = m.group(4)
        for mod, kind, s, lo, hi, cl in RQ_RE.findall(log):
            row[f"{mod}_{kind}_S"] = int(s); row[f"{mod}_{kind}_need"] = int(hi)
            row[f"{mod}_{kind}_clamps"] = int(cl)

        dut = np.array(open(out).read().split(), dtype=object)
        assert dut.size == 256 * 256 * 3, f"dut has {dut.size} values"
        dut8 = np.array([int(t, 16) for t in dut], np.int16).reshape(256, 256, 3)

        gq = gv.to_u8(G.golden_q(img)); gf = gv.to_u8(G.golden_fp32(img))
        row.update(metrics(gq, dut8, "q")); row.update(metrics(gf, dut8, "fp"))
        row["sat0_dut"] = round(100 * float((dut8 == 0).mean()), 3)
        row["sat0_gold"] = round(100 * float((gq == 0).mean()), 3)
        if CFG["keep_dut"]:
            Image.fromarray(dut8.astype(np.uint8)).save(os.path.join(CFG["dut_dir"], name + ".png"))
    except subprocess.TimeoutExpired:
        row["status"] = "HANG"
    except Exception as e:
        row["status"] = f"ERROR {type(e).__name__}: {e}"[:200]
    return row

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="dir of 256x256 PNGs (+ list.txt)")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--bin", default="obj_dir/Vtop_full_int_st_tb")
    ap.add_argument("--weights", default="#86-NN2_best.pth")
    ap.add_argument("--lut", default="data_files/lut_fp32.hex", help="golden warp LUT")
    ap.add_argument("--work", default="~/sweep_work", help="put on Linux FS, not /mnt/c")
    ap.add_argument("--out", default="results/sweep.csv")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=300, help="s per sim")
    ap.add_argument("--keep-dut", action="store_true", help="save DUT PNGs (~100 KB each)")
    ap.add_argument("--hex", default="full_sys_st.hex", help="output file the TB writes in data_files/")
    a = ap.parse_args()

    repo = os.path.abspath(a.repo); j = lambda p: p if os.path.isabs(p) else os.path.join(repo, p)
    corpus = os.path.expanduser(a.corpus); work = os.path.expanduser(a.work)
    binp = j(a.bin)
    if not os.access(binp, os.X_OK): raise SystemExit(f"no binary at {binp}")

    lst = os.path.join(corpus, "list.txt")
    names = ([l.strip() for l in open(lst) if l.strip()] if os.path.exists(lst)
             else sorted(f for f in os.listdir(corpus) if f.endswith(".png")))
    if a.limit: names = names[:a.limit]
    jobs = [(os.path.splitext(n)[0], os.path.join(corpus, n)) for n in names]

    # resume
    out = j(a.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            done = {r["name"] for r in csv.DictReader(f) if r["status"] == "OK"}
    jobs = [x for x in jobs if x[0] not in done]
    print(f"{len(names)} images, {len(done)} already OK, {len(jobs)} to run, {a.workers} workers")
    if not jobs: return

    # constants: one copy on the Linux FS, symlinked into every worker
    const = os.path.join(work, "const"); os.makedirs(const, exist_ok=True)
    src = os.path.join(repo, "data_files")
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if (os.path.isfile(p) and f not in PER_IMAGE and not f.startswith("full_")
                and not f.endswith((".png", ".log"))):
            shutil.copy2(p, os.path.join(const, f))
    dut_dir = os.path.join(os.path.dirname(out), "dut"); os.makedirs(dut_dir, exist_ok=True)

    weights = gv.load_weights_pth(j(a.weights))
    cfg = dict(lut=j(a.lut), work=work, const=const, bin=binp, timeout=a.timeout,
           keep_dut=a.keep_dut, dut_dir=dut_dir, hex=a.hex)

    new = not os.path.exists(out) or os.path.getsize(out) == 0
    t0 = time.time(); n = 0; bad = 0
    with open(out, "a", newline="") as fh, Pool(a.workers, init, (cfg, weights)) as pool:
        w = csv.DictWriter(fh, fieldnames=fields(), extrasaction="ignore")
        if new: w.writeheader()
        for row in pool.imap_unordered(run_one, jobs):
            w.writerow(row); fh.flush(); n += 1
            if row["status"] != "OK":
                bad += 1; print(f"  !! {row['name']}: {row['status']}")
            if n % 25 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"  {n}/{len(jobs)}  {el/60:.1f} min  ETA {el/n*(len(jobs)-n)/60:.1f} min  bad={bad}",
                      flush=True)
    print(f"done -> {out}")

if __name__ == "__main__":
    main()
