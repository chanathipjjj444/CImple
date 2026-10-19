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


# Each stub is skipped so `make regress` stays green while you fill them in.
# Delete `skip=True` once a test has a body. Every test starts the same way:
#     cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
#     await reset(dut)
# Helpers: push(dut, addr, lanes, first) -> one clock edge;
#          read(dut, addr) -> list of 4 signed ints (combinational read).


@cocotb.test()
async def test_first_overwrites(dut):
    """acc_first=1 replaces the stored value; it must not add to it."""
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)
    vals = [11,22,33,44]
    vals2 = [1, 2, 3, 4] 
    for l in range(m.N_LANES):
        for a in range(DEPTH):
            await push(dut, a, vals, True)
            await push(dut, a, vals2, True)
            read_value = await read(dut, a)
            assert read_value == vals2, f"Error checking the signed value expected 1, got {read_value[l][0]}"




@cocotb.test(skip=True)
async def test_accumulates(dut):
    """acc_first=0 adds to the stored value, including negative addends."""
    # TODO


@cocotb.test(skip=True)
async def test_sign_extension(dut):
    """23b -> 32b must sign-extend: try MAC_MIN, -1, MAC_MAX, 0."""
    # TODO


@cocotb.test(skip=True)
async def test_hold_when_not_enabled(dut):
    """acc_en=0 holds the entry, even with acc_first=1 and cim_in changing."""
    # TODO


@cocotb.test(skip=True)
async def test_address_isolation(dut):
    """Writing one entry must not disturb any of the other DEPTH-1."""
    # TODO


@cocotb.test(skip=True)
async def test_random_against_model(dut):
    """Random pushes vs a Python dict model (rng = np.random.default_rng(seed))."""
    # TODO


# --- ideas beyond the original set ------------------------------------------

@cocotb.test(skip=True)
async def test_lane_independence(dut):
    """Different value per lane; no lane's sum leaks into a neighbour."""
    # TODO


@cocotb.test(skip=True)
async def test_read_follows_addr_combinationally(dut):
    """Changing addr_i alone (no clock edge) changes rd_o."""
    # TODO


@cocotb.test(skip=True)
async def test_back_to_back_same_address(dut):
    """Push to one address on consecutive cycles with no idle gap."""
    # TODO


@cocotb.test(skip=True)
async def test_32b_wraparound(dut):
    """What happens past +/-2^31? Decide the spec, then assert it."""
    # TODO
