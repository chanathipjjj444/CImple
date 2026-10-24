"""cocotb test: cpl_lut -- exp LUT RAM (4 replicas) + reciprocal ROM (4 replicas).

    uv run make -C fpga/tb MODULE=test_cpl_lut TOPLEVEL=cpl_lut

Checks: RECIP == gm.build_recip_lut() on every port; exp LUT loaded line by
line (16 entries per write) reads back on every port with 1-cycle latency;
a partial reload changes only the written line; a read in the write cycle
returns the old entry, the next cycle the new one.
"""
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge

from cimple_rtl.model import cimple_model as gm

N_RD = 4


async def start(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    dut.exp_we_i.value = 0
    dut.exp_waddr_i.value = 0
    dut.exp_wdata_i.value = 0
    for l in range(N_RD):
        dut.exp_raddr_i[l].value = 0
        dut.recip_raddr_i[l].value = 0
    await FallingEdge(dut.clk_i)


def line(table, a):
    return sum(int(table[16 * a + i]) << (8 * i) for i in range(16))


async def load(dut, table, lines=range(16)):
    for a in lines:
        dut.exp_we_i.value = 1
        dut.exp_waddr_i.value = a
        dut.exp_wdata_i.value = line(table, a)
        await FallingEdge(dut.clk_i)
    dut.exp_we_i.value = 0


async def read_all(dut, rng):
    """Read every entry on every port (each port a different permutation)."""
    perms = [rng.permutation(256) for _ in range(N_RD)]
    exp_got = np.zeros((N_RD, 256), dtype=np.int64)
    rec_got = np.zeros((N_RD, 256), dtype=np.int64)
    for n in range(257):
        if n > 0:   # data for the address driven one cycle ago
            for l in range(N_RD):
                exp_got[l, perms[l][n - 1]] = int(dut.exp_rdata_o[l].value)
                rec_got[l, perms[(l + 1) % N_RD][n - 1]] = int(dut.recip_rdata_o[l].value)
        if n < 256:
            for l in range(N_RD):
                dut.exp_raddr_i[l].value = int(perms[l][n])
                dut.recip_raddr_i[l].value = int(perms[(l + 1) % N_RD][n])
        await FallingEdge(dut.clk_i)
    return exp_got, rec_got


@cocotb.test()
async def test_recip_rom(dut):
    await start(dut)
    rng = np.random.default_rng(1)
    _, rec = await read_all(dut, rng)
    ref = gm.build_recip_lut()
    for l in range(N_RD):
        bad = np.flatnonzero(rec[l] != ref)
        assert bad.size == 0, f"port {l}: recip mismatch at m={bad[:8]}: {rec[l][bad[:8]]} vs {ref[bad[:8]]}"


@cocotb.test()
async def test_exp_lut_load_read(dut):
    await start(dut)
    rng = np.random.default_rng(2)
    for table in (gm.build_exp_lut(0.05), rng.integers(0, 256, 256)):
        await load(dut, table)
        ex, _ = await read_all(dut, rng)
        for l in range(N_RD):
            bad = np.flatnonzero(ex[l] != table)
            assert bad.size == 0, f"port {l}: exp mismatch at {bad[:8]}"
    # partial reload of line 5 only
    new = table.copy()
    new[80:96] = rng.integers(0, 256, 16)
    await load(dut, new, lines=[5])
    ex, _ = await read_all(dut, rng)
    for l in range(N_RD):
        assert np.array_equal(ex[l], new), f"port {l}: partial reload wrong"


@cocotb.test()
async def test_exp_lut_read_during_write(dut):
    await start(dut)
    old = np.arange(256) ^ 0x5A
    await load(dut, old)
    # write line 3 and read entry 50 (line 3, byte 2) in the same cycle
    for l in range(N_RD):
        dut.exp_raddr_i[l].value = 50
    dut.exp_we_i.value = 1
    dut.exp_waddr_i.value = 3
    dut.exp_wdata_i.value = 0xEE << 16
    await FallingEdge(dut.clk_i)
    dut.exp_we_i.value = 0
    for l in range(N_RD):
        assert int(dut.exp_rdata_o[l].value) == old[50], "same-cycle read must see the old entry"
    await FallingEdge(dut.clk_i)
    for l in range(N_RD):
        assert int(dut.exp_rdata_o[l].value) == 0xEE, "next-cycle read must see the new entry"
