import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

async def check(dut, expected):
    await Timer(1, unit="ns")
    
@cocotb.test
async def wl_decode(dut):
    # we_i, wbank_i -> 
    cases = [
        (0, 0, 0b00),
        (0, 1, 0b00),
        (1, 0, 0b01),
        (1, 1, 0b10),
    ]
    
    for we, bank, exp in cases:
        dut.we_i.value = we
        dut.wbank_i.value = bank
        await Timer(1, unit="ns")
        assert dut.wl.value == exp, f"we={we} bank={bank}: wl={dut.wl.value}, expeceted {exp:02b}"


async def assign_inp(dut, we, wbank, wdata):
    dut.we_i.value = we
    dut.wbank_i.value = wbank
    dut.wdata_i.value = wdata
    await RisingEdge(dut.clk_i)
    
def bank_val(dut, b):
    return sum(int(dut.g_bank[b].g_bit[i].u_cell.q_o.value) << i for i in range(8))


@cocotb.test
async def bank_data(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    rand1 = random.getrandbits(8)
    rand2 = random.getrandbits(8)
    exp = [rand1, rand2]
    
    for bank in range(2):
        await assign_inp(dut, 1, bank,0x00)
    await assign_inp(dut, 0,0, 0x00)
    await Timer(1, unit="ns")
    
    for i in range(int(2)):
        for j in range(int(8)):
            assert dut.g_bank[i].g_bit[j].u_cell.q_o.value == 0, f"bank{i} bit{j}"
    
    for bank in range(2):
        await assign_inp(dut, 1, bank, 0xff)
        await assign_inp(dut, 0,0, 0xff)
    
    for i in range(int(2)):
        for j in range(int(8)):
            assert dut.g_bank[i].g_bit[j].u_cell.q_o.value == 1, f"bank{i} bit{j}"
    
    for bank in range(2):
        await assign_inp(dut, 1, bank, exp[bank])
    
    await assign_inp(dut, 0,0,0)
    await Timer(1, unit="ns")
    
    for i in range(int(2)):
        assert exp[i] == bank_val(dut, i), f"bank{i}: got {bank_val(dut, i):#04x}, expected {exp[i]:#04x}"
    
@cocotb.test
async def check_data_hold(dut):
    cocotb.start_soon(Clock(dut.clk_i,10, unit="ns").start())
    rand1 = random.getrandbits(8)
    rand2 = random.getrandbits(8)
    exp = [rand1, rand2]
    
    for bank in range(2):
        await assign_inp(dut, 1, bank, exp[bank])
    
    for bank in range(2):
        await assign_inp(dut,0,bank,  ~exp[bank] & 0xFF)
    
    await Timer(1, unit="ns")
    assert rand1 == bank_val(dut, 0)
    assert rand2 == bank_val(dut, 1)
        

@cocotb.test
async def oai_port_test(dut):
    cocotb.start_soon(Clock(dut.clk_i,10, unit="ns").start())
    dut.bank_sel_i.value = 0
    dut.x_bit_i.value = 0
    
    await Timer(1, unit="ns")
    for i in range(8):
        assert dut.g_oai[i].u_oai.y_o.value == ~((dut.q[0][i].value | dut.bank_sel_i.value) & (dut.q[1][i].value | dut.sel_n.value) & dut.x_bit_i.value), f"OAI221 Result: {dut.g_oai[i].u_oai.y_o.value}, Result to validate {~((dut.q[0][i].value | dut.bank_sel_i.value) & (dut.q[1][i].value | dut.sel_n.value) & dut.x_bit_i.value)}"

@cocotb.test
async def oai_compute(dut):
    cocotb.start_soon(Clock(dut.clk_i,10, "ns").start())
    w = [0xA5, 0x3C]
    for bank in range(2):
        await assign_inp(dut,1, bank, w[bank])
    
    await assign_inp(dut,0,0,0)
    await Timer(1, unit="ns")
    
    for sel in range(2):
        for x in range(2):
            dut.bank_sel_i.value = sel
            dut.x_bit_i.value = x
            await Timer(1, unit="ns")
            exp = w[sel] if x else 0
            got = (dut.p_msb_o.value.to_unsigned() << 4) | dut.p_lsb_o.value.to_unsigned()
            assert got == exp, f"sel={sel} x={x}: got {got:#04x}, expected {exp:#04x}"
            