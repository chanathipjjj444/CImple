"""cocotb test: cpl_quant, the 32b -> 8b requantiser (ARCH.md §2, §9).

Bit-exact against cimple_model.quantise (RND_FIX = 0) or a true
round-half-away-from-zero reference (RND_FIX = 1), for shift 0..31 and
scale 0..65535, random values plus edges and exact ties.

    uv run make -C fpga/tb MODULE=test_cpl_quant TOPLEVEL=cpl_quant
    uv run make -C fpga/tb MODULE=test_cpl_quant TOPLEVEL=cpl_quant PARAMS="-GRND_FIX=1"
    uv run make -C fpga/tb MODULE=test_cpl_quant TOPLEVEL=cpl_quant PARAMS="-GLAT=4"

The parameters are read back from the DUT (falling back to the PARAMS
string), so the same test checks every build. LAT = 0 is checked
combinationally, LAT > 0 by streaming one vector per cycle and comparing
LAT cycles later.
"""
import os
import re

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, Timer

from cimple_rtl.model import cimple_model as gm

IN_W = 32


def param(dut, name, default):
    try:
        return int(getattr(dut, name).value)
    except Exception:
        m = re.search(rf"-G{name}=(\d+)", os.environ.get("PARAMS", ""))
        return int(m.group(1)) if m else default


def quantise_fix(acc, scale, shift):
    """True round half away from zero, saturating to int8."""
    prod = int(acc) * int(scale)
    if shift == 0:
        r = prod
    elif prod >= 0:
        r = (prod + (1 << (shift - 1))) >> shift
    else:
        r = -((-prod + (1 << (shift - 1))) >> shift)
    return max(-128, min(127, r))


def vectors(rng):
    lo, hi = -(1 << (IN_W - 1)), (1 << (IN_W - 1)) - 1
    vec = []
    edges_a = [lo, lo + 1, -(1 << 22), -129, -128, -3, -2, -1, 0, 1, 2, 3, 127, 128,
               (1 << 22) - 1, hi - 1, hi]
    edges_s = [0, 1, 2, 3, 255, 256, 32767, 32768, 65534, 65535]
    for a in edges_a:
        for s in edges_s:
            for sh in (0, 1, 7, 15, 16, 30, 31):
                vec.append((a, s, sh))
    # exact ties and their neighbours: |a*s| = (2k+1) * 2^(sh-1) (+-1)
    for _ in range(1500):
        sh = int(rng.integers(1, 32))
        k = int(rng.integers(0, 300))
        s = 1 << int(rng.integers(0, 16)) if rng.random() < 0.5 else 1
        target = (2 * k + 1) << (sh - 1)
        for d in (-1, 0, 1):
            a = (target + d) // s if s > 1 else target + d
            for sign in (1, -1):
                aa = max(lo, min(hi, sign * a))
                vec.append((aa, s, sh))
    # random, log-uniform magnitudes so every shift sees in-range results
    for _ in range(6000):
        sh = int(rng.integers(0, 32))
        s = int(rng.integers(0, 1 << 16))
        mag = int(rng.integers(0, 32))
        a = int(rng.integers(-(1 << mag), (1 << mag) + 1))
        vec.append((max(lo, min(hi, a)), s, sh))
    return vec


@cocotb.test()
async def test_quant_vectors(dut):
    rnd_fix = param(dut, "RND_FIX", 0)
    lat = param(dut, "LAT", 0)
    ref = quantise_fix if rnd_fix else gm.quantise
    rng = np.random.default_rng(2604)
    vec = vectors(rng)
    dut._log.info(f"RND_FIX={rnd_fix} LAT={lat}: {len(vec)} vectors")

    def drive(a, s, sh):
        dut.acc_i.value = a & 0xFFFFFFFF
        dut.scale_i.value = s
        dut.shift_i.value = sh

    def check(i, got):
        a, s, sh = vec[i]
        exp = ref(a, s, sh)
        assert got == exp, f"acc={a} scale={s} shift={sh}: RTL {got} != ref {exp}"

    if lat == 0:
        for i, (a, s, sh) in enumerate(vec):
            drive(a, s, sh)
            await Timer(1, unit="ns")
            check(i, dut.q_o.value.to_signed())
    else:
        cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
        for n in range(len(vec) + lat):
            await FallingEdge(dut.clk_i)
            if n >= lat:   # the vector driven lat cycles ago
                check(n - lat, dut.q_o.value.to_signed())
            if n < len(vec):
                drive(*vec[n])
    dut._log.info("all vectors bit-exact")


@cocotb.test()
async def test_rounding_examples(dut):
    """Directed: the known -1 LSB bias of the golden model on negatives (§9)."""
    rnd_fix = param(dut, "RND_FIX", 0)
    if param(dut, "LAT", 0) != 0:
        return   # covered by test_quant_vectors
    cases = [  # (acc, scale, shift, gm result, true round-half-away)
        (3, 1, 1, 2, 2),      #  1.5  ->  2
        (-3, 1, 1, -2, -2),   # -1.5  -> -2 (tie: both agree)
        (-1, 1, 2, -1, 0),    # -0.25 -> gm -1, true 0
        (-5, 1, 2, -2, -1),   # -1.25 -> gm -2, true -1
        (-6, 1, 2, -2, -2),   # -1.5  -> -2
        (5, 1, 2, 1, 1),
        (1 << 30, 65535, 0, 127, 127),
        (-(1 << 31), 65535, 31, -128, -128),
    ]
    for a, s, sh, e_gm, e_fix in cases:
        assert gm.quantise(a, s, sh) == e_gm
        assert quantise_fix(a, s, sh) == e_fix
        dut.acc_i.value = a & 0xFFFFFFFF
        dut.scale_i.value = s
        dut.shift_i.value = sh
        await Timer(1, unit="ns")
        exp = e_fix if rnd_fix else e_gm
        got = dut.q_o.value.to_signed()
        assert got == exp, f"acc={a} scale={s} shift={sh}: RTL {got} != {exp}"
