import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

async def check(dut, expected):
    await Timer(1, unit="ns")
    assert int(dut)