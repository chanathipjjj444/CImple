"""cocotb test: 32b -> 8b requantiser."""

import numpy as np
import cocotb
from cocotb.triggers import Timer

from cimple_rtl.model import cimple_model as m


async def check(dut, acc, scale, shift):
    dut.acc_i.value = int(acc)
    dut.scale_i.value = int(scale)
    dut.shift_i.value = int(shift)
    await Timer(1, unit="ns")
    got = dut.q_o.value.to_signed()
    exp = m.quantise(acc, scale, shift)
    assert got == exp, f"acc={acc} scale={scale} shift={shift}: RTL {got} != model {exp}"


@cocotb.test()
async def test_random(dut):
    rng = np.random.default_rng(99)
    for _ in range(300):
        acc = int(rng.integers(-(1 << 22), 1 << 22))
        scale = int(rng.integers(1, 1 << 16))
        shift = int(rng.integers(0, 25))
        await check(dut, acc, scale, shift)


@cocotb.test()
async def test_saturation_and_rounding(dut):
    # saturation both ways
    await check(dut, 1 << 22, 0xFFFF, 0)
    await check(dut, -(1 << 22), 0xFFFF, 0)
    # exact half, positive and negative -> rounds away from zero
    await check(dut, 3, 1, 1)        #  1.5 ->  2
    await check(dut, -3, 1, 1)       # -1.5 -> -2
    await check(dut, 0, 12345, 7)
    # identity
    for v in (-128, -1, 0, 1, 127):
        await check(dut, v, 1, 0)
