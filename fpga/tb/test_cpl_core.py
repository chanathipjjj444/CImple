"""cocotb test: cpl_core (32 partitions x 64 rows x 2 banks + 32->4 output mux).

Bit-exact against cimple_model.core_mac (INT8 activations) and numpy W @ x
(UQ8 activations, act_unsigned_i = 1). Run for every PIPE_TREE value (0, 1,
2 = default):

    uv run make -C fpga/tb MODULE=test_cpl_core TOPLEVEL=cpl_core
    uv run make -C fpga/tb MODULE=test_cpl_core TOPLEVEL=cpl_core PARAMS="-GPIPE_TREE=0"

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
    """PIPE_TREE of the DUT (handle, else make PARAMS; mismatch = stale build)."""
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


def bitplanes(acts):
    raw = np.asarray(acts).astype(np.int64) % 256
    out = []
    for k in range(m.DATA_W - 1, -1, -1):
        bits = 0
        for r in range(m.N_ROWS):
            bits |= int((int(raw[r]) >> k) & 1) << r
        out.append(bits)
    return out


def group_word(wcol, g):
    word = 0
    for i in range(WR_BYTES):
        word |= int(m.u8(wcol[WR_BYTES * g + i])) << (8 * i)
    return word


def expected(W, a, unsigned):
    """(N_PART,) results; W (N_PART, N_ROWS) INT8."""
    if unsigned:
        exp = m.s8(W) @ np.asarray(a, dtype=np.int64)
    else:
        exp = m.core_mac(W, a)
        assert np.array_equal(exp, m.s8(W) @ m.s8(a))
    assert np.all(np.abs(exp) < (1 << (m.MAC_W - 1)))
    return exp


def ref_trace(W, a, unsigned):
    """(8, N_PART): accumulator after each bit-serial update."""
    W = m.s8(W).astype(np.int64)
    raw = (np.asarray(a).astype(np.int64) % 256)
    acc = np.zeros(m.N_PART, dtype=np.int64)
    tr = []
    for k in range(m.DATA_W - 1, -1, -1):
        comb = W @ ((raw >> k) & 1)
        acc = (comb if unsigned else -comb) if k == m.DATA_W - 1 else 2 * acc + comb
        tr.append(acc.copy())
    return np.array(tr)


async def reset(dut):
    dut.rst_ni.value = 0
    for sig in ("we_i", "wbank_i", "wpart_i", "wgrp_i", "wdata_i", "bank_sel_i",
                "x_bits_i", "mac_en_i", "mac_first_i", "act_unsigned_i",
                "out_phase_i"):
        getattr(dut, sig).value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


async def start(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)
    dut._log.info("PIPE_TREE=%d MAC_LAT=%d", pipe_tree(dut), mac_lat(dut))


def write_seq(W, parts=range(m.N_PART), groups=range(N_WGRP)):
    return [(p, g, group_word(W[p], g)) for p in parts for g in groups]


async def write(dut, bank, seq):
    dut.wbank_i.value = bank
    for p, g, word in seq:
        dut.we_i.value = 1
        dut.wpart_i.value = p
        dut.wgrp_i.value = g
        dut.wdata_i.value = word
        await RisingEdge(dut.clk_i)
    dut.we_i.value = 0
    dut.wdata_i.value = (1 << 128) - 1      # garbage while we_i = 0


async def drive_pass(dut, acts, bank, unsigned=False):
    dut.bank_sel_i.value = bank
    dut.act_unsigned_i.value = int(unsigned)
    for i, bits in enumerate(bitplanes(acts)):
        dut.x_bits_i.value = bits
        dut.mac_en_i.value = 1
        dut.mac_first_i.value = int(i == 0)
        await RisingEdge(dut.clk_i)
    dut.mac_en_i.value = 0
    dut.mac_first_i.value = 0


def lanes(dut):
    return [dut.cim_out_o[l].value.to_signed() for l in range(m.N_LANES)]


async def sweep(dut):
    """Read all 32 results by sweeping out_phase_i (combinational, no clock)."""
    out = [None] * m.N_PART
    for k in range(m.MUX_PHASES):
        dut.out_phase_i.value = k
        await Timer(1, unit="ns")
        for l, v in enumerate(lanes(dut)):
            out[l * m.MUX_PHASES + k] = v
    return np.array(out, dtype=np.int64)


async def drain(dut):
    """Clocked drain: phase k in the k-th cycle, 4 lanes each."""
    out = [None] * m.N_PART
    for k in range(m.MUX_PHASES):
        dut.out_phase_i.value = k
        await Timer(1, unit="ns")
        for l, v in enumerate(lanes(dut)):
            out[l * m.MUX_PHASES + k] = v
        await RisingEdge(dut.clk_i)
    return np.array(out, dtype=np.int64)


async def run_pass(dut, acts, bank, unsigned=False):
    await drive_pass(dut, acts, bank, unsigned)
    for _ in range(pipe_tree(dut)):
        await RisingEdge(dut.clk_i)
    return await drain(dut)


def rand_W(rng):
    return rng.integers(-128, 128, (m.N_PART, m.N_ROWS), dtype=np.int64)


def rand_a(rng, unsigned):
    return rng.integers(0, 256, m.N_ROWS, dtype=np.int64) if unsigned \
        else rng.integers(-128, 128, m.N_ROWS, dtype=np.int64)


def check(got, exp, what):
    bad = np.flatnonzero(got != exp)
    assert bad.size == 0, f"{what}: partitions {bad[:6]} RTL {got[bad[:6]]} != {exp[bad[:6]]}"


# ------------------------------------------------------------------ tests --
@cocotb.test()
async def test_matvec_signed(dut):
    """64x32 INT8 mat-vec in 8 cycles on both banks vs cimple_model.core_mac."""
    await start(dut)
    rng = np.random.default_rng(31337)
    W = [rand_W(rng), rand_W(rng)]
    for b in (0, 1):
        await write(dut, b, write_seq(W[b]))
    for trial in range(8):
        b = trial & 1
        a = rand_a(rng, False)
        check(await run_pass(dut, a, b), expected(W[b], a, False), f"trial {trial} bank {b}")


@cocotb.test()
async def test_matvec_unsigned(dut):
    """INT8 weights x UQ8 activations vs numpy W @ x."""
    await start(dut)
    rng = np.random.default_rng(4242)
    W = [rand_W(rng), rand_W(rng)]
    for b in (0, 1):
        await write(dut, b, write_seq(W[b]))
    for trial in range(8):
        b = (trial >> 1) & 1
        uns = bool(trial & 1)             # interleave signed / unsigned passes
        a = rand_a(rng, uns)
        check(await run_pass(dut, a, b, uns), expected(W[b], a, uns),
              f"trial {trial} bank {b} unsigned {uns}")


@cocotb.test()
async def test_corners(dut):
    """Extremes on every partition, incl. the unsigned range limits."""
    await start(dut)
    full = lambda v: np.full(m.N_ROWS, v, dtype=np.int64)  # noqa: E731
    Wc = np.stack([full([-128, 127, -1, 0][p % 4]) for p in range(m.N_PART)])
    Wc[5::8] = np.where(np.arange(m.N_ROWS) & 1, -128, 127)
    await write(dut, 0, write_seq(Wc))
    await write(dut, 1, write_seq(-1 - Wc))      # bitwise complement in bank 1
    cases = [(full(-128), False), (full(127), False), (full(-1), False),
             (full(0), False), (full(255), True), (full(128), True),
             (full(0), True), (full(127), True)]
    for i, (a, uns) in enumerate(cases):
        for b, W in ((0, Wc), (1, -1 - Wc)):
            check(await run_pass(dut, a, b, uns), expected(W, a, uns), f"corner {i} bank {b}")


@cocotb.test()
async def test_out_phase_mapping(dut):
    """Lane l, phase k = partition 8l+k, for every (l, k)."""
    await start(dut)
    W = np.zeros((m.N_PART, m.N_ROWS), dtype=np.int64)
    W[:, 0] = np.arange(m.N_PART) - 100          # acc[p] = p - 100, all distinct
    W[:, 1] = 3
    await write(dut, 0, write_seq(W))
    a = np.zeros(m.N_ROWS, dtype=np.int64)
    a[0], a[1] = 1, 0
    exp = np.arange(m.N_PART) - 100
    await drive_pass(dut, a, 0)
    for _ in range(pipe_tree(dut)):
        await RisingEdge(dut.clk_i)
    for k in [3, 0, 7, 5, 1, 6, 2, 4]:          # any order, combinational
        dut.out_phase_i.value = k
        await Timer(1, unit="ns")
        got = lanes(dut)
        want = [int(exp[l * m.MUX_PHASES + k]) for l in range(m.N_LANES)]
        assert got == want, f"phase {k}: lanes {got} != partitions {want}"
    check(await drain(dut), exp, "clocked drain")
    check(await sweep(dut), exp, "sweep")


@cocotb.test()
async def test_bank_independence(dut):
    """Stream bank-1 writes during bank-0 passes (and v.v.); neither disturbs."""
    await start(dut)
    rng = np.random.default_rng(77)
    W = [rand_W(rng), rand_W(rng)]
    await write(dut, 0, write_seq(W[0]))
    await write(dut, 1, write_seq(W[1]))
    for active in (0, 1, 0):
        new = rand_W(rng)
        seq = write_seq(new)
        rng.shuffle(seq)                          # arbitrary partition / group order
        dut.wbank_i.value = 1 - active
        lat = mac_lat(dut)
        si = 0
        while si < len(seq):
            a = rand_a(rng, bool(rng.integers(2)))
            uns = bool(np.all(a >= 0) and np.any(a > 127))
            planes = bitplanes(a)
            dut.bank_sel_i.value = active
            dut.act_unsigned_i.value = int(uns)
            for c in range(lat):                  # MAC cycles + pipeline, writes every cycle
                dut.mac_en_i.value = int(c < m.DATA_W)
                dut.mac_first_i.value = int(c == 0)
                if c < m.DATA_W:
                    dut.x_bits_i.value = planes[c]
                if si < len(seq):
                    p, g, word = seq[si]
                    dut.we_i.value = 1
                    dut.wpart_i.value, dut.wgrp_i.value, dut.wdata_i.value = p, g, word
                    si += 1
                else:
                    dut.we_i.value = 0
                await RisingEdge(dut.clk_i)
            dut.we_i.value = 0
            check(await sweep(dut), expected(W[active], a, uns),
                  f"active bank {active} during writes ({si}/{len(seq)})")
        W[1 - active] = new
        a = rand_a(rng, False)
        check(await run_pass(dut, a, 1 - active), expected(new, a, False),
              f"rewritten bank {1 - active}")


@cocotb.test()
async def test_partial_reload(dut):
    """Single (partition, group) 128b writes change exactly 16 weights."""
    await start(dut)
    rng = np.random.default_rng(909)
    W = rand_W(rng)
    await write(dut, 1, write_seq(W))
    for trial in range(10):
        n = int(rng.integers(1, 6))
        new = rand_W(rng)
        seq = []
        for _ in range(n):
            p, g = int(rng.integers(m.N_PART)), int(rng.integers(N_WGRP))
            seq.append((p, g, group_word(new[p], g)))
            W[p, WR_BYTES * g: WR_BYTES * (g + 1)] = new[p, WR_BYTES * g: WR_BYTES * (g + 1)]
        await write(dut, 1, seq)
        uns = bool(trial & 1)
        a = rand_a(rng, uns)
        check(await run_pass(dut, a, 1, uns), expected(W, a, uns), f"trial {trial}")


@cocotb.test()
async def test_latency_and_hold(dut):
    """Results appear exactly at c0 + MAC_LAT; hold over the drain; back-to-back."""
    await start(dut)
    pt, lat = pipe_tree(dut), mac_lat(dut)
    rng = np.random.default_rng(5150)
    W = rand_W(rng)
    await write(dut, 0, write_seq(W))
    passes = [(rand_a(rng, bool(i & 1)), bool(i & 1)) for i in range(4)]
    # stream: pass i starts in cycle 8i (no gaps), then idle; snapshot all 32
    # accumulators every cycle through an out_phase sweep inside the cycle.
    snaps = []
    stream = [(a, uns, c) for a, uns in passes for c in range(m.DATA_W)]
    stream += [(None, False, 0)] * (lat + m.MUX_PHASES)
    for a, uns, c in stream:
        if a is not None:
            dut.x_bits_i.value = bitplanes(a)[c]
            dut.act_unsigned_i.value = int(uns)
            dut.bank_sel_i.value = 0
            dut.mac_en_i.value = 1
            dut.mac_first_i.value = int(c == 0)
        else:
            dut.mac_en_i.value = 0
            dut.mac_first_i.value = 0
        snaps.append(await sweep(dut))
        await RisingEdge(dut.clk_i)
    for i, (a, uns) in enumerate(passes):
        tr = ref_trace(W, a, uns)
        c0 = m.DATA_W * i
        for j in range(m.DATA_W):                 # one update per cycle after the pipe
            check(snaps[c0 + 1 + pt + j], tr[j], f"pass {i} cycle c0+{1 + pt + j}")
        check(snaps[c0 + lat], expected(W, a, uns), f"pass {i} at c0+MAC_LAT")
    # the last pass holds through a full drain window
    last = expected(W, *passes[-1])
    for c in range(m.DATA_W * (len(passes) - 1) + lat, len(snaps)):
        check(snaps[c], last, f"hold cycle {c}")
