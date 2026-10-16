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


@cocotb.test()
async def test_first_overwrites(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)
    await push(dut, 3, [1, 2, 3, 4], first=True)
    await push(dut, 3, [10, 20, 30, 40], first=True)      # overwrite, not add
    assert await read(dut, 3) == [10, 20, 30, 40]


@cocotb.test()
async def test_accumulates(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)
    await push(dut, 5, [100, -200, 300, -400], first=True)
    await push(dut, 5, [1, 2, 3, 4], first=False)
    await push(dut, 5, [10, 20, 30, 40], first=False)
    assert await read(dut, 5) == [111, -178, 333, -356]


@cocotb.test()
async def test_sign_extension(dut):
    """23b -> 32b must sign-extend, including the most negative value."""
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)
    vals = [MAC_MIN, -1, MAC_MAX, 0]
    await push(dut, 0, vals, first=True)
    assert await read(dut, 0) == vals
    await push(dut, 0, [-1, -1, -1, -1], first=False)
    assert await read(dut, 0) == [v - 1 for v in vals]


@cocotb.test()
async def test_hold_when_not_enabled(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)
    await push(dut, 7, [5, 6, 7, 8], first=True)
    dut.acc_en_i.value = 0
    dut.acc_first_i.value = 1                  # must be ignored without acc_en
    for i in range(3):
        for l in range(m.N_LANES):
            dut.cim_in_i[l].value = 1000 + i
        await RisingEdge(dut.clk_i)
    assert await read(dut, 7) == [5, 6, 7, 8]


@cocotb.test()
async def test_address_isolation(dut):
    """A write to one entry must not disturb any other."""
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)
    for a in range(DEPTH):
        await push(dut, a, [a, -a, 2 * a, 3 * a], first=True)
    await push(dut, 20, [1000, 1000, 1000, 1000], first=False)
    for a in range(DEPTH):
        exp = [a, -a, 2 * a, 3 * a]
        if a == 20:
            exp = [e + 1000 for e in exp]
        assert await read(dut, a) == exp, f"entry {a}"


@cocotb.test()
async def test_random_against_model(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)
    rng = np.random.default_rng(99)
    model = {}
    for _ in range(400):
        a = int(rng.integers(0, DEPTH))
        # keep each addend small enough that a few thousand adds cannot leave 32b
        lanes = [int(v) for v in rng.integers(MAC_MIN, MAC_MAX + 1, m.N_LANES)]
        first = a not in model or rng.random() < 0.2
        await push(dut, a, lanes, first)
        model[a] = lanes if first else [x + y for x, y in zip(model[a], lanes)]
    for a, exp in model.items():
        assert await read(dut, a) == exp, f"entry {a}"
