"""cocotb test: one CIM partition against the golden model."""
import os

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from cimple_rtl.model import cimple_model as m


async def reset(dut):
    dut.rst_ni.value = 0
    dut.we_i.value = 0
    dut.wbank_i.value = 0
    dut.waddr_i.value = 0
    dut.wdata_i.value = 0
    dut.bank_sel_i.value = 0
    dut.x_bits_i.value = 0
    dut.mac_en_i.value = 0
    dut.mac_first_i.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


async def write_weights(dut, weights, bank):
    dut.wbank_i.value = bank
    for r, w in enumerate(weights):
        dut.we_i.value = 1
        dut.waddr_i.value = r
        dut.wdata_i.value = int(m.u8(w))
        await RisingEdge(dut.clk_i)
    dut.we_i.value = 0
    await RisingEdge(dut.clk_i)


async def run_mac(dut, acts, bank):
    """Drive the 8 bit-serial cycles and return the final accumulator."""
    dut.bank_sel_i.value = bank
    raw = m.u8(np.asarray(acts))
    for i, k in enumerate(range(m.DATA_W - 1, -1, -1)):
        bits = 0
        for r in range(m.N_ROWS):
            bits |= int((int(raw[r]) >> k) & 1) << r
        dut.x_bits_i.value = bits
        dut.mac_en_i.value = 1
        dut.mac_first_i.value = 1 if i == 0 else 0
        await RisingEdge(dut.clk_i)
    dut.mac_en_i.value = 0
    dut.mac_first_i.value = 0
    await Timer(1, unit="ns")
    return dut.acc_o.value.to_signed()


# Test random dot products
@cocotb.test()
async def test_random_dot_products_x2(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)
    
    rng = np.random.default_rng(1234)
    for i in range(20):
        w = rng.integers(-128, 128, m.N_ROWS, dtype=np.int64)
        a = rng.integers(-128, 128, m.N_ROWS, dtype=np.int64)
        await write_weights(dut, w, bank=0)
        got = await run_mac(dut, a, bank=0)
        exp, _ = m.partition_mac(w, a)
        assert exp == got, f"trial {i}: RTL{got} != model {exp}"

@cocotb.test()
async def test_random_dot_products(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)

    rng = np.random.default_rng(1234)
    for trial in range(20):
        w = rng.integers(-128, 128, m.N_ROWS, dtype=np.int64)
        a = rng.integers(-128, 128, m.N_ROWS, dtype=np.int64)
        await write_weights(dut, w, bank=0)
        got = await run_mac(dut, a, bank=0)
        exp, _ = m.partition_mac(w, a)
        assert got == exp, f"trial {trial}: RTL {got} != model {exp}"


@cocotb.test()
async def test_corner_values(dut):
    """The signed corners are where a nibble-split MAC usually breaks."""
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)

    cases = [
        (np.full(m.N_ROWS, -128), np.full(m.N_ROWS, -128)),   # max positive
        (np.full(m.N_ROWS, -128), np.full(m.N_ROWS,  127)),   # max negative
        (np.full(m.N_ROWS,  127), np.full(m.N_ROWS,  127)),
        (np.zeros(m.N_ROWS, dtype=np.int64), np.full(m.N_ROWS, -128)),
        (np.full(m.N_ROWS,   -1), np.full(m.N_ROWS,    1)),
    ]
    for i, (w, a) in enumerate(cases):
        await write_weights(dut, w, bank=0)
        got = await run_mac(dut, a, bank=0)
        exp, _ = m.partition_mac(w, a)
        assert got == exp, f"corner {i}: RTL {got} != model {exp}"


@cocotb.test()
async def test_dual_bank_isolation(dut):
    """Writing bank 1 must not disturb a MAC running out of bank 0."""
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)

    rng = np.random.default_rng(7)
    w0 = rng.integers(-128, 128, m.N_ROWS, dtype=np.int64)
    w1 = rng.integers(-128, 128, m.N_ROWS, dtype=np.int64)
    a = rng.integers(-128, 128, m.N_ROWS, dtype=np.int64)

    await write_weights(dut, w0, bank=0)
    await write_weights(dut, w1, bank=1)

    for bank, w in ((0, w0), (1, w1)):
        got = await run_mac(dut, a, bank=bank)
        exp, _ = m.partition_mac(w, a)
        assert got == exp, f"bank {bank}: RTL {got} != model {exp}"
