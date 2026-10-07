"""cocotb test: full 32-partition core, including the 32->4 output mux."""

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from cimple_rtl.model import cimple_model as m


async def reset(dut):
    dut.rst_ni.value = 0
    for sig in ("we_i", "wbank_i", "wpart_i", "waddr_i", "wdata_i",
                "bank_sel_i", "x_bits_i", "mac_en_i", "mac_first_i",
                "out_adv_i", "out_clr_i"):
        getattr(dut, sig).value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


async def load_weights(dut, W, bank=0):
    """W : (N_PART, N_ROWS) INT8"""
    dut.wbank_i.value = bank
    dut.we_i.value = 1
    for p in range(m.N_PART):
        dut.wpart_i.value = p
        for r in range(m.N_ROWS):
            dut.waddr_i.value = r
            dut.wdata_i.value = int(m.u8(W[p, r]))
            await RisingEdge(dut.clk_i)
    dut.we_i.value = 0
    await RisingEdge(dut.clk_i)


async def run_pass(dut, acts, bank=0):
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


async def drain(dut):
    """Walk the 3b counter through all 8 phases, 4 lanes each."""
    dut.out_clr_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.out_clr_i.value = 0

    out = [None] * m.N_PART
    for phase in range(m.MUX_PHASES):
        await Timer(1, unit="ns")
        for l in range(m.N_LANES):
            out[l * m.MUX_PHASES + phase] = dut.cim_out_o[l].value.to_signed()
        dut.out_adv_i.value = 1
        await RisingEdge(dut.clk_i)
    dut.out_adv_i.value = 0
    return np.array(out, dtype=np.int64)


@cocotb.test()
async def test_core_matvec(dut):
    """The core computes a 64x32 INT8 matrix-vector product in 8 cycles."""
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)

    rng = np.random.default_rng(31337)
    W = rng.integers(-128, 128, (m.N_PART, m.N_ROWS), dtype=np.int64)
    await load_weights(dut, W, bank=0)

    for _ in range(3):
        a = rng.integers(-128, 128, m.N_ROWS, dtype=np.int64)
        await run_pass(dut, a, bank=0)
        got = await drain(dut)
        exp = W @ a
        assert np.array_equal(got, exp), \
            f"mismatch at {np.flatnonzero(got != exp)[:5]}: {got[:4]} vs {exp[:4]}"
