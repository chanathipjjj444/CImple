"""cocotb test: wdata_select (the Reg + write data select).

Conventions this test assumes -- they are DECISIONS, change them here and in
the RTL together (see the header of rtl/wdata_select.sv):

  * Each beat carries 4 lanes x 8b. Lane l of beat b lands in byte (4*b + l)
    of the wide word, byte 0 = bits [7:0].
  * dest_i = 0 : weights, 128b word = 4 beats, shown on wbl_o  / wbl_valid_o
    dest_i = 1 : activations, 64b word = 2 beats (8 x INT8), on xin_o / xin_valid_o
  * The word is complete on the clock edge that accepts its last beat; the
    output and its valid flag are visible right after that edge, for one cycle.
  * flush_i pushes out a partly filled word: unfilled bytes are 0, then the
    collector restarts.
  * Only the selected destination's valid flag ever pulses. full_o mirrors it.
"""
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

LANES = 4
BEATS = {0: 4, 1: 2}                          # beats per word, by destination
WIDTH = {0: 128, 1: 64}


async def reset(dut):
    dut.rst_ni.value = 0
    dut.score_valid_i.value = 0
    dut.dest_i.value = 0
    dut.flush_i.value = 0
    for l in range(LANES):
        dut.score_i[l].value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


def pack(beats):
    """Golden packing: list of 4-lane beats -> integer word."""
    word = 0
    for b, lanes in enumerate(beats):
        for l, v in enumerate(lanes):
            word |= (int(v) & 0xFF) << (8 * (LANES * b + l))
    return word


async def beat(dut, lanes, dest):
    """Send one beat, return (wbl, wbl_valid, xin, xin_valid, full) after it."""
    dut.dest_i.value = dest
    for l, v in enumerate(lanes):
        dut.score_i[l].value = int(v)
    dut.score_valid_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.score_valid_i.value = 0
    await Timer(1, unit="ns")
    return snapshot(dut)


def snapshot(dut):
    return dict(wbl=int(dut.wbl_o.value), wv=int(dut.wbl_valid_o.value),
                xin=int(dut.xin_o.value), xv=int(dut.xin_valid_o.value),
                full=int(dut.full_o.value))


async def send_word(dut, beats, dest):
    snaps = []
    for lanes in beats:
        snaps.append(await beat(dut, lanes, dest))
    return snaps


def rand_beats(rng, n):
    return [[int(v) for v in rng.integers(-128, 128, LANES)] for _ in range(n)]


@cocotb.test()
async def test_weight_word(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)
    rng = np.random.default_rng(1)
    beats = rand_beats(rng, BEATS[0])
    snaps = await send_word(dut, beats, dest=0)

    for s in snaps[:-1]:                              # not complete yet
        assert s["wv"] == 0 and s["xv"] == 0 and s["full"] == 0
    last = snaps[-1]
    assert last["wv"] == 1 and last["xv"] == 0 and last["full"] == 1
    assert last["wbl"] == pack(beats), f"{last['wbl']:#x} != {pack(beats):#x}"


@cocotb.test()
async def test_activation_word(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)
    rng = np.random.default_rng(2)
    beats = rand_beats(rng, BEATS[1])
    snaps = await send_word(dut, beats, dest=1)

    assert snaps[0]["wv"] == 0 and snaps[0]["xv"] == 0
    last = snaps[-1]
    assert last["xv"] == 1 and last["wv"] == 0 and last["full"] == 1
    assert last["xin"] == pack(beats)


@cocotb.test()
async def test_valid_is_one_cycle(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)
    await send_word(dut, rand_beats(np.random.default_rng(3), BEATS[0]), dest=0)
    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")
    s = snapshot(dut)
    assert s["wv"] == 0 and s["full"] == 0


@cocotb.test()
async def test_back_to_back_words(dut):
    """The collector restarts cleanly after a word is emitted."""
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)
    rng = np.random.default_rng(4)
    for _ in range(5):
        beats = rand_beats(rng, BEATS[0])
        snaps = await send_word(dut, beats, dest=0)
        assert snaps[-1]["wv"] == 1 and snaps[-1]["wbl"] == pack(beats)


@cocotb.test()
async def test_signed_bytes_do_not_bleed(dut):
    """Negative lanes must not sign-extend into the neighbouring bytes."""
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)
    beats = [[-1, 0, -128, 127]] * BEATS[0]
    snaps = await send_word(dut, beats, dest=0)
    assert snaps[-1]["wbl"] == pack(beats)


@cocotb.test()
async def test_flush_partial_word(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)
    beats = [[1, 2, 3, 4], [5, 6, 7, 8]]                  # 2 of 4 beats
    await send_word(dut, beats, dest=0)
    dut.flush_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.flush_i.value = 0
    await Timer(1, unit="ns")
    s = snapshot(dut)
    assert s["wv"] == 1 and s["xv"] == 0
    assert s["wbl"] == pack(beats), "unfilled bytes must be zero"

    # and the collector is empty again afterwards
    beats2 = rand_beats(np.random.default_rng(5), BEATS[0])
    snaps = await send_word(dut, beats2, dest=0)
    assert snaps[-1]["wbl"] == pack(beats2)


@cocotb.test()
async def test_no_valid_without_input(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)
    for _ in range(8):
        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")
        s = snapshot(dut)
        assert s["wv"] == 0 and s["xv"] == 0 and s["full"] == 0
