"""cocotb test: LUT-based split softmax against the golden model."""
import os
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from cimple_rtl.model import cimple_model as m

SCORE_SCALE = float(os.environ.get("SCORE_SCALE", "0.05"))
EXP_LUT = m.build_exp_lut(SCORE_SCALE)
RECIP_LUT = m.build_recip_lut()


async def reset(dut):
    dut.rst_ni.value = 0
    dut.row_start_i.value = 0
    dut.z_valid_i.value = 0
    dut.z_i.value = 0
    dut.row_done_i.value = 0
    dut.partial_i.value = 0
    dut.partial_valid_i.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


async def run_row(dut, scores, values):
    dut.row_start_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.row_start_i.value = 0

    got_e = []
    # stream the scores; the LUT has 1 cycle of latency so drain one extra
    for z in list(scores) + [None]:
        dut.z_valid_i.value = 0 if z is None else 1
        if z is not None:
            dut.z_i.value = int(m.u8(z))
        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")
        if dut.e_valid_o.value == 1:
            got_e.append(int(dut.e_o.value))
    dut.z_valid_i.value = 0
    await RisingEdge(dut.clk_i)

    exp = m.softmax_row(scores, values, EXP_LUT, RECIP_LUT)
    assert got_e == list(exp["e"]), f"exp stream {got_e} != {list(exp['e'])}"
    assert int(dut.sum_o.value) == exp["sum"], \
        f"sum {int(dut.sum_o.value)} != {exp['sum']}"

    # denominator: pulse row_done, then apply the CIM partial
    dut.row_done_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.row_done_i.value = 0
    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")
    assert int(dut.recip_o.value) == exp["M"], \
        f"M {int(dut.recip_o.value)} != {exp['M']}"
    assert int(dut.recip_shift_o.value) == exp["shift"], \
        f"shift {int(dut.recip_shift_o.value)} != {exp['shift']}"

    dut.partial_i.value = exp["partial"]
    dut.partial_valid_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.partial_valid_i.value = 0
    await Timer(1, unit="ns")
    got = dut.score_o.value.to_signed()
    assert got == exp["score"], f"score {got} != {exp['score']}"
    return exp


@cocotb.test()
async def test_rows(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)

    rng = np.random.default_rng(2024)
    errs = []
    for _ in range(10):
        n = int(rng.integers(8, 33))
        scores = rng.integers(-128, 128, n, dtype=np.int64)
        values = rng.integers(-128, 128, n, dtype=np.int64)
        exp = await run_row(dut, scores, values)
        ref = m.reference_softmax_row(scores, values, SCORE_SCALE)
        errs.append(abs(exp["score"] - ref))

    # Not a correctness assert -- a visible record of how much the INT8 LUT
    # approximation costs versus float softmax. Tighten as the design evolves.
    dut._log.info(f"mean |int - float| error over rows: {np.mean(errs):.2f} LSB "
                  f"(max {np.max(errs):.2f})")


@cocotb.test()
async def test_uniform_row_is_mean(dut):
    """All scores equal -> softmax is uniform -> output is the mean of V."""
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)

    values = np.arange(-16, 16, dtype=np.int64)
    exp = await run_row(dut, np.zeros(32, dtype=np.int64), values)
    mean = float(values.mean())
    assert abs(exp["score"] - mean) <= 2, f"{exp['score']} vs mean {mean}"
