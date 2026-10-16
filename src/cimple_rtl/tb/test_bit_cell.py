import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

async def check(dut, expected):
    await Timer(1, unit="ns")
    assert int(dut.q_o.value) == expected
    assert int(dut.qb_o.value) == 1 - expected
    
@cocotb.test()
async def bit_cell(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    dut.we_i.value = 0
    dut.d_i.value = 0
    await RisingEdge(dut.clk_i)
    
    for bit in(1,0,1):
        dut.we_i.value, dut.d_i.value = 1, bit
        await RisingEdge(dut.clk_i)
        dut.we_i.value = 0
        await check(dut, bit)
    
    for _ in range(8):
        dut.d_i.value = 1 - int(dut.d_i.value)
        await RisingEdge(dut.clk_i)
        await check(dut, 1)
        
    ref, rng = 1, random.Random(1)
    for _ in range(500):
        we,d = rng.randint(0,1), rng.randint(0,1)
        dut.we_i.value, dut.d_i.value = we, d
        await RisingEdge(dut.clk_i)
        if we:
            ref = d 
        await check(dut, ref)