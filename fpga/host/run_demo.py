#!/usr/bin/env python3
"""
CIMple-FPGA demo: one encoder-only attention head (paper Sec. IV-C) on a
chosen backend, checked against the integer reference and float attention.

  python fpga/host/run_demo.py                               # ISA simulator
  python fpga/host/run_demo.py --backend devmem --base 0xA4000000   # Linux on the Versal APU
  python fpga/host/run_demo.py --backend uio --uio /dev/uio0
  python fpga/host/run_demo.py --backend xsdb --base 0xA4000000 [--url tcp:host:3121]

The base address is wherever the Vivado block design places cpl_top's
AXI4-Lite slave (see fpga/vivado); it is not fixed by the RTL.
Needs numpy and cimple_rtl (the golden model) on the path, e.g. the sw venv.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_repo_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
if os.path.isdir(_repo_src):
    sys.path.insert(1, os.path.abspath(_repo_src))

from cpl import compiler as C  # noqa: E402
from cpl.driver import Device, DevMemBackend, SimBackend, XsdbBackend  # noqa: E402


def make_backend(a):
    if a.backend == "sim":
        return SimBackend(gbuf_bytes=a.gbuf_kib * 1024)
    if a.backend == "devmem":
        return DevMemBackend(base_addr=int(a.base, 0))
    if a.backend == "uio":
        return DevMemBackend(uio=a.uio)
    if a.backend == "xsdb":
        return XsdbBackend(int(a.base, 0), xsdb=a.xsdb, url=a.url, target=a.target)
    raise ValueError(a.backend)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["sim", "devmem", "uio", "xsdb"], default="sim")
    ap.add_argument("--base", default="0xA4000000", help="physical base of the AXI4-Lite slave")
    ap.add_argument("--uio", default="/dev/uio0")
    ap.add_argument("--xsdb", default="xsdb", help="xsdb / xsct executable")
    ap.add_argument("--url", default=None, help="hw_server url, e.g. tcp:localhost:3121")
    ap.add_argument("--target", default="*A72*#0", help="xsdb target name pattern")
    ap.add_argument("--gbuf-kib", type=int, default=512, help="sim GBUF size")
    ap.add_argument("--n", type=int, default=64, help="tokens")
    ap.add_argument("--dm", type=int, default=64, help="model dim")
    ap.add_argument("--d", type=int, default=64, help="head dim (<= 64)")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--asm", action="store_true", help="print the program")
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    X = rng.standard_normal((a.n, a.dm))
    Ws = [rng.standard_normal((a.dm, a.d)) / np.sqrt(a.dm) for _ in range(3)]
    cal = C.calibrate(X, *Ws, causal=a.causal)
    b = C.encoder_attention_head(cal.xq, cal.wq, cal.wk, cal.wv, cal.scales, causal=a.causal)
    segs = b.segments()
    print(f"encoder head N={a.n} d_model={a.dm} d={a.d} causal={a.causal}: "
          f"{len(b.program)} instructions in {len(segs)} program(s), GBUF {b.layout.top} B")
    if a.asm:
        from cpl import isa
        for s in segs:
            print(isa.disassemble(s))

    dev = Device(make_backend(a))
    dev.check_id()
    cfg = dev.cfg()
    if b.layout.top > cfg["gbuf_bytes"]:
        print(f"layout needs {b.layout.top} B > device GBUF {cfg['gbuf_bytes']} B")
        return 2
    t0 = time.monotonic()
    outs, tot = dev.execute(b)
    dt = time.monotonic() - t0

    ok = True
    for k in b.layout.outputs:
        same = np.array_equal(outs[k], b.ref[k])
        ok &= same
        print(f"  {k:3s} vs integer reference: {'exact' if same else 'MISMATCH'}"
              + ("" if same else f" ({int(np.sum(outs[k] != b.ref[k]))} elements differ)"))
    fl = C.float_attention(X, *Ws, causal=a.causal)
    print(f"  cosine(device output, float attention) = "
          f"{C.cosine(outs['O'] * cal.scales.out_scale, fl):.4f}")
    print(f"counters: cycles={tot['cycles']} instret={tot['instret']} passes={tot['passes']} "
          f"runs={tot['runs']}  (host wall time {dt:.3f} s)")
    if tot["cycles"]:
        print(f"  at 200 MHz: {tot['cycles'] / 200:.1f} us")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
