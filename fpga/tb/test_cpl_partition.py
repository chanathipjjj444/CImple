"""cocotb test: one cpl_partition (VP1202 CIM partition) against the golden model.

Bit-exact against cimple_model.partition_mac for INT8 activations and against
a plain numpy dot for UQ8 activations (act_unsigned_i = 1). Run for both
PIPE_TREE values (0, 1, 2 = default):

    uv run make -C fpga/tb MODULE=test_cpl_partition TOPLEVEL=cpl_partition
    uv run make -C fpga/tb MODULE=test_cpl_partition TOPLEVEL=cpl_partition \
        PARAMS="-GPIPE_TREE=0"

(the Makefile derives a separate build directory from PARAMS).
"""
import os
import re

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from cimple_rtl.model import cimple_model as m

WR_BYTES = 16
N_WGRP = m.N_ROWS // WR_BYTES


# ---------------------------------------------------------------- helpers --
def pipe_tree(dut) -> int:
    """PIPE_TREE of the elaborated DUT.

    Read from the parameter handle when the simulator exposes it, else from
    make's PARAMS. If both exist and disagree the build is stale (the Makefile
    reuses sim_build/<sim>_<top> across PARAMS values): fail loudly.
    """
    mt = re.search(r"-GPIPE_TREE=(\d+)", os.environ.get("PARAMS", ""))
    env = int(mt.group(1)) if mt else None
    try:
        hdl = int(dut.PIPE_TREE.value)
    except Exception:  # noqa: BLE001 -- parameter not exposed by the simulator
        hdl = None
    if hdl is not None and env is not None and hdl != env:
        raise RuntimeError(f"stale build: PARAMS says PIPE_TREE={env}, DUT has {hdl}; "
                           "pass a separate SIM_BUILD=... per PARAMS value")
    return hdl if hdl is not None else (env if env is not None else 2)


def mac_lat(dut) -> int:
    return m.DATA_W + pipe_tree(dut)


def ref_trace(w, a, unsigned):
    """Bit-serial reference (accumulator after each of the 8 updates)."""
    w = m.s8(w).astype(np.int64)
    raw = (np.asarray(a).astype(np.int64) % 256) if unsigned else m.u8(a)
    acc, tr = 0, []
    for k in range(m.DATA_W - 1, -1, -1):
        comb = int(np.sum(w * ((raw >> k) & 1)))
        if k == m.DATA_W - 1:
            acc = comb if unsigned else -comb
        else:
            acc = 2 * acc + comb
        tr.append(acc)
    return tr


def expected(w, a, unsigned):
    if unsigned:
        exp = int(np.dot(m.s8(w), np.asarray(a, dtype=np.int64)))
    else:
        exp, _ = m.partition_mac(w, a)
    assert -(1 << (m.MAC_W - 1)) <= exp < (1 << (m.MAC_W - 1))
    assert ref_trace(w, a, unsigned)[-1] == exp
    return exp


def bitplanes(acts):
    """acts (64,) -> [x_bits for k = 7 .. 0] as 64b ints (row r -> bit r)."""
    raw = np.asarray(acts).astype(np.int64) % 256
    out = []
    for k in range(m.DATA_W - 1, -1, -1):
        bits = 0
        for r in range(m.N_ROWS):
            bits |= int((int(raw[r]) >> k) & 1) << r
        out.append(bits)
    return out


def group_word(w, g):
    """128b write word for row group g: byte i = row 16*g + i."""
    word = 0
    for i in range(WR_BYTES):
        word |= int(m.u8(w[WR_BYTES * g + i])) << (8 * i)
    return word


async def reset(dut):
    dut.rst_ni.value = 0
    for sig in ("we_i", "wbank_i", "wgrp_i", "wdata_i", "bank_sel_i",
                "x_bits_i", "mac_en_i", "mac_first_i", "act_unsigned_i"):
        getattr(dut, sig).value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


async def start(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)


async def write_groups(dut, w, bank, groups=range(N_WGRP)):
    dut.wbank_i.value = bank
    for g in groups:
        dut.we_i.value = 1
        dut.wgrp_i.value = g
        dut.wdata_i.value = group_word(w, g)
        await RisingEdge(dut.clk_i)
    dut.we_i.value = 0
    dut.wdata_i.value = (1 << 128) - 1          # garbage while we_i = 0


async def drive_pass(dut, acts, bank, unsigned=False, stall_rng=None):
    """Drive the 8 bit-serial cycles (optionally with random stall cycles)."""
    dut.bank_sel_i.value = bank
    dut.act_unsigned_i.value = int(unsigned)
    for i, bits in enumerate(bitplanes(acts)):
        while stall_rng is not None and stall_rng.random() < 0.4:
            dut.mac_en_i.value = 0
            dut.mac_first_i.value = int(stall_rng.integers(2))  # ignored when idle
            dut.x_bits_i.value = int(stall_rng.integers(0, 1 << 63))
            await RisingEdge(dut.clk_i)
        dut.x_bits_i.value = bits
        dut.mac_en_i.value = 1
        dut.mac_first_i.value = 1 if i == 0 else 0
        await RisingEdge(dut.clk_i)
    dut.mac_en_i.value = 0
    dut.mac_first_i.value = 0


async def read_acc(dut):
    await Timer(1, unit="ns")
    return dut.acc_o.value.to_signed()


async def run_mac(dut, acts, bank, unsigned=False, stall_rng=None):
    await drive_pass(dut, acts, bank, unsigned, stall_rng)
    for _ in range(pipe_tree(dut)):
        await RisingEdge(dut.clk_i)
    return await read_acc(dut)


def rand_w(rng):
    return rng.integers(-128, 128, m.N_ROWS, dtype=np.int64)


def rand_a(rng, unsigned):
    return rng.integers(0, 256, m.N_ROWS, dtype=np.int64) if unsigned \
        else rng.integers(-128, 128, m.N_ROWS, dtype=np.int64)


# ------------------------------------------------------------------ tests --
@cocotb.test()
async def test_random_signed(dut):
    """INT8 x INT8, both banks, vs cimple_model.partition_mac."""
    await start(dut)
    dut._log.info("PIPE_TREE=%d MAC_LAT=%d", pipe_tree(dut), mac_lat(dut))
    rng = np.random.default_rng(1234)
    for trial in range(40):
        bank = trial & 1
        w, a = rand_w(rng), rand_a(rng, False)
        await write_groups(dut, w, bank)
        got = await run_mac(dut, a, bank)
        exp = expected(w, a, False)
        assert got == exp, f"trial {trial}: RTL {got} != model {exp}"


@cocotb.test()
async def test_random_unsigned(dut):
    """INT8 weights x UQ8 activations vs numpy dot."""
    await start(dut)
    rng = np.random.default_rng(99)
    for trial in range(40):
        bank = (trial >> 1) & 1
        w, a = rand_w(rng), rand_a(rng, True)
        await write_groups(dut, w, bank)
        got = await run_mac(dut, a, bank, unsigned=True)
        exp = expected(w, a, True)
        assert got == exp, f"trial {trial}: RTL {got} != numpy {exp}"


@cocotb.test()
async def test_corners(dut):
    """Signed / unsigned extremes, where a nibble-split MAC usually breaks."""
    await start(dut)
    full = lambda v: np.full(m.N_ROWS, v, dtype=np.int64)  # noqa: E731
    alt = np.array([(-128 if r & 1 else 127) for r in range(m.N_ROWS)])
    cases = [  # (w, a, unsigned)
        (full(-128), full(-128), False),   # max positive  (+2^20)
        (full(-128), full(127), False),    # max negative
        (full(127), full(127), False),
        (full(127), full(-128), False),
        (full(0), full(-128), False),
        (full(-1), full(-1), False),       # all-ones x all-ones
        (full(-1), full(1), False),
        (alt, alt, False),
        (full(-128), full(255), True),     # unsigned most negative (-2,088,960)
        (full(127), full(255), True),      # unsigned most positive
        (full(-1), full(255), True),       # all-ones
        (full(-128), full(0), True),
        (full(127), full(128), True),      # MSB only: the +comb first cycle
        (full(-128), full(128), True),
        (alt, full(255), True),
    ]
    for i, (w, a, uns) in enumerate(cases):
        for bank in (0, 1):
            await write_groups(dut, w, bank)
            got = await run_mac(dut, a, bank, unsigned=uns)
            exp = expected(w, a, uns)
            assert got == exp, f"corner {i} bank {bank}: RTL {got} != {exp}"


@cocotb.test()
async def test_bank_independence(dut):
    """Writing bank 1 during a MAC on bank 0 must not disturb it (and v.v.)."""
    await start(dut)
    rng = np.random.default_rng(7)
    w = [rand_w(rng), rand_w(rng)]
    await write_groups(dut, w[0], 0)
    await write_groups(dut, w[1], 1)
    for trial in range(8):
        active = trial & 1
        new = rand_w(rng)
        a = rand_a(rng, bool(trial & 2))
        uns = bool(trial & 2)
        # concurrently rewrite the inactive bank, one group per MAC cycle
        dut.bank_sel_i.value = active
        dut.act_unsigned_i.value = int(uns)
        dut.wbank_i.value = 1 - active
        for i, bits in enumerate(bitplanes(a)):
            dut.x_bits_i.value = bits
            dut.mac_en_i.value = 1
            dut.mac_first_i.value = int(i == 0)
            dut.we_i.value = int(i < N_WGRP)
            dut.wgrp_i.value = i % N_WGRP
            dut.wdata_i.value = group_word(new, i % N_WGRP)
            await RisingEdge(dut.clk_i)
        dut.we_i.value = 0
        dut.mac_en_i.value = 0
        dut.mac_first_i.value = 0
        for _ in range(pipe_tree(dut)):
            await RisingEdge(dut.clk_i)
        got = await read_acc(dut)
        exp = expected(w[active], a, uns)
        assert got == exp, f"trial {trial}: active bank {active}: {got} != {exp}"
        w[1 - active] = new
        # the freshly written bank computes with its new weights
        got = await run_mac(dut, a, 1 - active, unsigned=uns)
        exp = expected(new, a, uns)
        assert got == exp, f"trial {trial}: rewritten bank: {got} != {exp}"


@cocotb.test()
async def test_partial_group_reload(dut):
    """128b group writes: reloading one row group touches exactly 16 rows."""
    await start(dut)
    rng = np.random.default_rng(55)
    w = rand_w(rng)
    await write_groups(dut, w, 0)
    for trial in range(12):
        g = int(rng.integers(N_WGRP))
        new = rand_w(rng)
        await write_groups(dut, new, 0, groups=[g])
        w[WR_BYTES * g: WR_BYTES * (g + 1)] = new[WR_BYTES * g: WR_BYTES * (g + 1)]
        a = rand_a(rng, False)
        got = await run_mac(dut, a, 0)
        assert got == expected(w, a, False), f"trial {trial} group {g}"
    # one-hot rows locate every byte lane: w = e_r * (r+1), x = 1
    await write_groups(dut, np.zeros(m.N_ROWS, dtype=np.int64), 1)
    for r in range(m.N_ROWS):
        w1 = np.zeros(m.N_ROWS, dtype=np.int64)
        w1[r] = r + 1
        await write_groups(dut, w1, 1, groups=[r // WR_BYTES])
        got = await run_mac(dut, np.ones(m.N_ROWS, dtype=np.int64), 1)
        assert got == r + 1, f"row {r}: {got}"
        w1[r] = 0
        await write_groups(dut, w1, 1, groups=[r // WR_BYTES])


@cocotb.test()
async def test_latency_hold_stall(dut):
    """MAC_LAT exactness, hold until the next pass, back-to-back, stalls."""
    await start(dut)
    pt, lat = pipe_tree(dut), mac_lat(dut)
    rng = np.random.default_rng(2024)
    w = rand_w(rng)
    await write_groups(dut, w, 0)
    prev = None
    for trial in range(10):
        uns = bool(trial & 1)
        a = rand_a(rng, uns)
        tr = ref_trace(w, a, uns)
        exp = expected(w, a, uns)
        planes = bitplanes(a)
        dut.bank_sel_i.value = 0
        dut.act_unsigned_i.value = int(uns)
        # cycle c0 .. c0+lat: present 8 planes, then idle; sample every cycle
        seen = []
        for c in range(lat + 1):
            if c < m.DATA_W:
                dut.x_bits_i.value = planes[c]
                dut.mac_en_i.value = 1
                dut.mac_first_i.value = int(c == 0)
            else:
                dut.mac_en_i.value = 0
                dut.mac_first_i.value = 0
            seen.append(await read_acc(dut))   # value visible in cycle c0 + c
            await RisingEdge(dut.clk_i)
        seen.append(await read_acc(dut))       # cycle c0 + lat + 1
        # before the pass touches the accumulator the previous result holds
        if prev is not None:
            for c in range(1 + pt):
                assert seen[c] == prev, f"trial {trial}: old result lost in cycle c0+{c}"
        # the accumulator steps through the trace, one update per cycle
        for j in range(m.DATA_W):
            assert seen[1 + pt + j] == tr[j], \
                f"trial {trial}: cycle c0+{1 + pt + j}: {seen[1 + pt + j]} != {tr[j]}"
        assert seen[lat] == exp and seen[lat + 1] == exp
        for _ in range(int(rng.integers(0, 4))):
            await RisingEdge(dut.clk_i)
        assert await read_acc(dut) == exp, "result did not hold"
        prev = exp
    # back-to-back passes (no idle cycle between them): pass i starts in
    # cycle 8i, its result must be visible in cycle 8i + MAC_LAT
    passes = [(rand_a(rng, bool(i & 1)), bool(i & 1)) for i in range(6)]
    seen = []
    for i, (a, uns) in enumerate(passes + [(None, False)]):
        for c in range(m.DATA_W):
            if a is not None:
                dut.x_bits_i.value = bitplanes(a)[c]
                dut.act_unsigned_i.value = int(uns)
                dut.mac_en_i.value = 1
                dut.mac_first_i.value = int(c == 0)
            else:
                dut.mac_en_i.value = 0
                dut.mac_first_i.value = 0
            seen.append(await read_acc(dut))
            await RisingEdge(dut.clk_i)
    for i, (a, uns) in enumerate(passes):
        got = seen[m.DATA_W * i + lat]
        assert got == expected(w, a, uns), f"back-to-back pass {i}: {got}"
    # random stall cycles (mac_en_i = 0, garbage on x_bits / mac_first) inside a pass
    for trial in range(10):
        uns = bool(trial & 1)
        a = rand_a(rng, uns)
        got = await run_mac(dut, a, 0, uns, stall_rng=rng)
        assert got == expected(w, a, uns), f"stall trial {trial}: {got}"


@cocotb.test()
async def test_reset_flushes_pipeline(dut):
    """Synchronous reset flushes the PIPE_TREE control stages.

    The accumulator itself has no reset (the mac_first update overwrites
    it; ARCH.md fixes no reset value), so the check is that no pending
    pipeline update reaches acc_o after the reset edge, and that a pass
    after reset is exact.
    """
    await start(dut)
    rng = np.random.default_rng(3)
    w = rand_w(rng)
    await write_groups(dut, w, 0)
    a = rand_a(rng, False)
    # stop right after the last bit: up to PIPE_TREE updates still pending
    await drive_pass(dut, a, 0)
    dut.rst_ni.value = 0
    await RisingEdge(dut.clk_i)                  # synchronous reset edge
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)
    held = await read_acc(dut)
    for _ in range(4):
        await RisingEdge(dut.clk_i)
        assert await read_acc(dut) == held, "pipeline update leaked through reset"
    # weights (no reset) survive and still compute
    assert await run_mac(dut, a, 0) == expected(w, a, False)
