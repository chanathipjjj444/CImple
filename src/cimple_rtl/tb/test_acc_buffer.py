"""cocotb test: acc_buffer (the ACC + Buffer after CIM OUT).

Spec being tested (see the header of rtl/acc_buffer.sv):
    acc_en=1, acc_first=1 : buf[addr] <= sign-extended cim_in
    acc_en=1, acc_first=0 : buf[addr] <= buf[addr] + sign-extended cim_in
    acc_en=0              : hold
    rd_o = buf[addr_i], combinational on addr_i
Assumptions: DEPTH = 64 (the module default), 4 lanes, and the buffer contents
are NOT assumed to be zero after reset -- every entry is written with
acc_first before it is read.
"""
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from cimple_rtl.model import cimple_model as m

DEPTH = 64
MAC_MIN, MAC_MAX = -(1 << (m.MAC_W - 1)), (1 << (m.MAC_W - 1)) - 1


async def reset(dut):
    dut.rst_ni.value = 0
    dut.acc_en_i.value = 0
    dut.acc_first_i.value = 0
    dut.addr_i.value = 0
    for l in range(m.N_LANES):
        dut.cim_in_i[l].value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


async def push(dut, addr, lanes, first):
    """One accumulate/overwrite cycle."""
    dut.addr_i.value = addr
    for l, v in enumerate(lanes):
        dut.cim_in_i[l].value = int(v)
    dut.acc_en_i.value = 1
    dut.acc_first_i.value = 1 if first else 0
    await RisingEdge(dut.clk_i)
    dut.acc_en_i.value = 0
    dut.acc_first_i.value = 0


async def read(dut, addr):
    dut.addr_i.value = addr
    await Timer(1, unit="ns")
    return [dut.rd_o[l].value.to_signed() for l in range(m.N_LANES)]


ACC_MIN, ACC_MAX = -(1 << (m.ACC_W - 1)), (1 << (m.ACC_W - 1)) - 1


def wrap32(x):
    """Two's-complement wrap to ACC_W bits -- what a plain adder does."""
    x &= (1 << m.ACC_W) - 1
    return x - (1 << m.ACC_W) if x > ACC_MAX else x


async def start(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)


# --- the three cases in the header -------------------------------------------

@cocotb.test()
async def test_first_overwrites(dut):
    """acc_first=1 replaces the stored value; it must not add to it."""
    await start(dut)
    vals = [11, 22, 33, 44]
    vals2 = [1, 2, 3, 4]
    for a in range(DEPTH):
        await push(dut, a, vals, True)
        await push(dut, a, vals2, True)
        read_value = await read(dut, a)
        assert read_value == vals2, f"addr {a}: expected {vals2}, got {read_value}"


@cocotb.test()
async def test_accumulates(dut):
    """acc_first=0 adds to the stored value, including negative addends."""
    await start(dut)
    await push(dut, 5, [100, -200, 300, -400], first=True)
    await push(dut, 5, [1, 2, 3, 4], first=False)
    await push(dut, 5, [-10, 20, -30, 40], first=False)
    assert await read(dut, 5) == [91, -178, 273, -356]


@cocotb.test()
async def test_sign_extension(dut):
    """23b -> 32b must sign-extend: try MAC_MIN, -1, MAC_MAX, 0."""
    await start(dut)
    vals = [MAC_MIN, -1, MAC_MAX, 0]
    await push(dut, 0, vals, first=True)
    assert await read(dut, 0) == vals
    # a zero-extended MAC_MIN would read back as +2^22 here
    await push(dut, 0, [MAC_MIN] * m.N_LANES, first=False)
    assert await read(dut, 0) == [v + MAC_MIN for v in vals]


@cocotb.test()
async def test_hold_when_not_enabled(dut):
    """acc_en=0 holds the entry, even with acc_first=1 and cim_in changing."""
    await start(dut)
    await push(dut, 7, [5, 6, 7, 8], first=True)
    dut.acc_en_i.value = 0
    dut.acc_first_i.value = 1                  # must be ignored without acc_en
    for i in range(3):
        for l in range(m.N_LANES):
            dut.cim_in_i[l].value = 1000 + i
        await RisingEdge(dut.clk_i)
    dut.acc_first_i.value = 0
    assert await read(dut, 7) == [5, 6, 7, 8]


@cocotb.test()
async def test_address_isolation(dut):
    """Writing one entry must not disturb any of the other DEPTH-1."""
    await start(dut)
    for a in range(DEPTH):
        await push(dut, a, [a, -a, 2 * a, -3 * a], first=True)
    await push(dut, 20, [1000] * m.N_LANES, first=False)
    for a in range(DEPTH):
        exp = [a, -a, 2 * a, -3 * a]
        if a == 20:
            exp = [e + 1000 for e in exp]
        assert await read(dut, a) == exp, f"entry {a}"


@cocotb.test()
async def test_random_against_model(dut):
    """Random pushes vs a Python dict model (rng = np.random.default_rng(seed))."""
    await start(dut)
    rng = np.random.default_rng(99)
    model = {}
    for _ in range(400):
        a = int(rng.integers(0, DEPTH))
        lanes = [int(v) for v in rng.integers(MAC_MIN, MAC_MAX + 1, m.N_LANES)]
        first = a not in model or rng.random() < 0.2
        await push(dut, a, lanes, first)
        model[a] = lanes if first else [wrap32(x + y) for x, y in zip(model[a], lanes)]
    for a, exp in model.items():
        assert await read(dut, a) == exp, f"entry {a}"


# --- ideas beyond the original set ------------------------------------------

@cocotb.test()
async def test_lane_independence(dut):
    """Different value per lane; no lane's sum leaks into a neighbour."""
    await start(dut)
    steps = [([1000, -2000, 3000, -4000], True),
             ([1, 2, 3, 4], False),
             ([-50, 60, -70, 80], False)]
    exp = [0] * m.N_LANES
    for lanes, first in steps:
        await push(dut, 9, lanes, first)
        exp = list(lanes) if first else [e + v for e, v in zip(exp, lanes)]
        assert await read(dut, 9) == exp


@cocotb.test()
async def test_read_follows_addr_combinationally(dut):
    """Changing addr_i alone (no clock edge) changes rd_o."""
    await start(dut)
    await push(dut, 1, [1, 1, 1, 1], first=True)
    await push(dut, 2, [-2, -2, -2, -2], first=True)
    await RisingEdge(dut.clk_i)                # settle, then stay between edges
    await Timer(1, unit="ns")
    for addr, exp in ((1, [1] * 4), (2, [-2] * 4), (1, [1] * 4)):
        dut.addr_i.value = addr
        await Timer(1, unit="ns")              # 2 ns total < half a period
        got = [dut.rd_o[l].value.to_signed() for l in range(m.N_LANES)]
        assert got == exp, f"addr {addr}: expected {exp}, got {got}"


@cocotb.test()
async def test_back_to_back_same_address(dut):
    """Push to one address on consecutive cycles with no idle gap."""
    await start(dut)
    seq = [[i, -i, 3 * i, -5 * i] for i in range(1, 11)]
    dut.addr_i.value = 4
    dut.acc_en_i.value = 1
    for i, lanes in enumerate(seq):
        for l, v in enumerate(lanes):
            dut.cim_in_i[l].value = v
        dut.acc_first_i.value = 1 if i == 0 else 0
        await RisingEdge(dut.clk_i)
    dut.acc_en_i.value = 0
    dut.acc_first_i.value = 0
    exp = [sum(col) for col in zip(*seq)]
    assert await read(dut, 4) == exp


@cocotb.test()
async def test_32b_wraparound(dut):
    """What happens past +/-2^31? Decide the spec, then assert it.

    [A] DECISION: the ACC is a plain 32b adder -- it WRAPS, it does not
    saturate. The paper is silent. It is unreachable for real attention: one
    23b pass is at most 2^22 in magnitude, so overflow needs > 2^9 = 512 passes,
    i.e. a 32768-long dot product. If you ever want saturation, change the RTL
    and this test together.
    """
    await start(dut)
    await push(dut, 0, [MAC_MAX, MAC_MIN, MAC_MAX, MAC_MIN], first=True)
    exp = [MAC_MAX, MAC_MIN, MAC_MAX, MAC_MIN]
    n = 520
    for _ in range(n - 1):
        await push(dut, 0, [MAC_MAX, MAC_MIN, 1, -1], first=False)
        exp = [wrap32(e + v) for e, v in zip(exp, [MAC_MAX, MAC_MIN, 1, -1])]
    got = await read(dut, 0)
    assert got == exp
    assert got[0] < 0 and got[1] > 0, "lanes 0/1 should have wrapped"


# --- against the golden model: the job the paper gives this block ------------
#
# [P] "The partial results are stored and accumulated in the intermediate
#     result buffer, after which they are quantized to INT8."
# A 64-row array gives one 23b partial dot product per pass. Anything longer
# than 64 -- a d_model > 64 projection, or A'V over > 64 tokens -- is several
# passes summed here. The model's partition_mac is the 8-cycle bit-serial pass
# and asserts itself against np.dot, so these tests check RTL vs model vs maths.

def passes(W, x):
    """W: (N_LANES, K) INT8, x: (K,) INT8, K a multiple of N_ROWS.
    Returns the per-pass 23b partials the core would put on CIM OUT."""
    k = W.shape[1]
    assert k % m.N_ROWS == 0
    return [[m.partition_mac(W[l, t:t + m.N_ROWS], x[t:t + m.N_ROWS])[0]
             for l in range(m.N_LANES)]
            for t in range(0, k, m.N_ROWS)]


@cocotb.test()
async def test_tiled_projection_vs_model(dut):
    """Q = X Wq with d_model = 256 (BERT-mini width): 4 passes per output.

    Every buffer entry holds 4 output columns of one token. 16 tokens x
    8 column groups = 128 entries > DEPTH, so groups are processed in two
    rounds of 64 entries, which is also what a sequencer would have to do.
    Then the 32b sums go through the model's requantiser and are compared to
    float X @ Wq -- logged, not asserted, like the softmax error.
    """
    await start(dut)
    rng = np.random.default_rng(7)
    n_tok, d_model, d_head = 16, 256, 32
    X = rng.normal(0, 1.0, (n_tok, d_model))
    Wq = rng.normal(0, 0.06, (d_model, d_head))
    sx, sw = np.abs(X).max() / 127, np.abs(Wq).max() / 127
    qX, qW = m.to_int8(X, sx), m.to_int8(Wq, sw)

    acc = np.zeros((n_tok, d_head), dtype=np.int64)
    groups = d_head // m.N_LANES
    entries = [(i, g) for i in range(n_tok) for g in range(groups)]
    for r0 in range(0, len(entries), DEPTH):
        chunk = entries[r0:r0 + DEPTH]
        for addr, (i, g) in enumerate(chunk):
            cols = slice(g * m.N_LANES, (g + 1) * m.N_LANES)
            for t, lanes in enumerate(passes(qW[:, cols].T, qX[i])):
                await push(dut, addr, lanes, first=(t == 0))
        for addr, (i, g) in enumerate(chunk):
            acc[i, g * m.N_LANES:(g + 1) * m.N_LANES] = await read(dut, addr)

    exact = qX @ qW
    assert np.array_equal(acc, exact), "RTL ACC != integer X @ W"

    real = X @ Wq
    sq = np.abs(real).max() / 127
    scl, sh = m.ratio_to_scale_shift(sx * sw / sq)
    q8 = np.array([[m.quantise(a, scl, sh) for a in row] for row in acc])
    err = np.abs(q8 * sq - real)
    dut._log.info(f"projection d_model={d_model}: INT8 vs float "
                  f"mean {err.mean() / sq:.3f} LSB, max {err.max() / sq:.3f} LSB")


@cocotb.test()
async def test_tiled_av_vs_model(dut):
    """A'V over 256 tokens: e (exp-LUT values) x V, 4 token tiles of 64.

    The softmax numerator for one query row is accumulated across tiles; one
    entry per group of 4 output columns. e is kept 0..127 here: the core reads
    activations as SIGNED INT8 (see the xfail in tests/test_attention_model.py).
    """
    await start(dut)
    rng = np.random.default_rng(11)
    n_tok, d_head = 256, 32
    lut = m.build_exp_lut(0.05)
    z = rng.integers(60, 128, n_tok)                       # INT8 scores
    e = np.minimum(np.array([lut[m.u8(int(s))] for s in z]), 127)
    V = rng.integers(-128, 128, (n_tok, d_head))

    groups = d_head // m.N_LANES
    for g in range(groups):
        cols = slice(g * m.N_LANES, (g + 1) * m.N_LANES)
        for t, lanes in enumerate(passes(V[:, cols].T, e)):
            await push(dut, g, lanes, first=(t == 0))
    got = np.concatenate([await read(dut, g) for g in range(groups)])
    assert np.array_equal(got, e @ V), "RTL ACC != e @ V"
    dut._log.info(f"A'V over {n_tok} tokens: {n_tok // m.N_ROWS} passes, "
                  f"|sum| max {np.abs(got).max()} (> 23b: {np.abs(got).max() >= 1 << 22})")
