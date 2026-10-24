"""cocotb system test: cpl_top (the whole accelerator) through its AXI4-Lite port.

    uv run make -C fpga/tb MODULE=test_cpl_top TOPLEVEL=cpl_top PARAMS="-GGBUF_BYTES=65536"
    uv run make -C fpga/tb MODULE=test_cpl_top TOPLEVEL=cpl_top            # default 512 KiB
    ... COCOTB_TEST_FILTER="test_fuzz"                                      # one test

Reference: the bit-exact ISA simulator fpga/host/cpl/sim.py (CplSim). Every
test loads the SAME program and GBUF bytes into the RTL (through AXI only --
no backdoor anywhere) and into the simulator, runs both and compares STATUS,
err_code, PC, INSTRET, PASSES and the GBUF image (full image unless a test
says otherwise; CYCLES is only logged, the simulator's is an estimate).
The end-to-end tests drive the RTL with the host driver (cpl.driver.Device
over cpl.cocotb_backend.CocotbBackend) and compare with the compiler's
integer reference.

GBUF size is read from the CFG register, so the same module runs on any
GBUF_BYTES build. RND_FIX builds: set CPL_RND_FIX=1 in the environment.

All tests of a module share one simulation, so the RTL's memories (GBUF,
banks, ACC buffer, exp LUT) carry over from test to test. The simulator's
state is carried along with them (_CARRY); after a failed test the next one
re-synchronises (GBUF read back through AXI, ACC and exp LUT rewritten by a
scrub program, banks reloaded by every test that computes).
For GBUF_BYTES > 64 KiB, "full image" compares cover [0, 64 KiB) and the
last 4 KiB (all tests place their data there); smaller builds compare all.
"""
from __future__ import annotations

import os

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, with_timeout

from axil_bfm import AxilMaster, OKAY
from cpl import compiler as C
from cpl import isa
from cpl.cocotb_backend import CocotbBackend
from cpl.driver import Device
from cpl.isa import Acc, Err, Instr, Op, Out
from cpl.sim import (CTRL_CLEAR, CTRL_START, GBUF_BASE, IMEM_BASE, REG_CFG, REG_CTRL, REG_CYCLES,
                     REG_ID, REG_INSTRET, REG_PASSES, REG_PC, REG_PC_START, REG_STATUS, CplSim)

ACC_DEPTH = 1024
RND_FIX = bool(int(os.environ.get("CPL_RND_FIX", "0")))
CLK_NS = 5                                   # 200 MHz
PAPER_CYCLES_PER_PASS = 16                   # 8 MAC + 8 drain (paper Fig. 4 / ARCH.md §6)
SCAN_MAX = 65536                             # full-compare cap (see module docstring)

_FRESH = "fresh"
_CARRY = _FRESH                              # simulator state matching the RTL, or None


# ---------------------------------------------------------------------------
# harness: DUT + simulator in lock step
# ---------------------------------------------------------------------------
class Top:
    def __init__(self, dut):
        self.dut = dut

    async def start(self, *, jitter=False, seed=0):
        d = self.dut
        cocotb.start_soon(Clock(d.clk_i, CLK_NS, unit="ns").start())
        d.rst_ni.value = 0
        self.bfm = AxilMaster(d, d.clk_i, jitter=jitter, seed=seed)
        await ClockCycles(d.clk_i, 5)
        d.rst_ni.value = 1
        await ClockCycles(d.clk_i, 2)
        cfg = await self.bfm.read32(REG_CFG)
        self.G = (cfg & 0xFFFF) * 1024
        assert cfg >> 16 == 256, hex(cfg)
        self.sim = CplSim(gbuf_bytes=self.G, acc_depth=ACC_DEPTH, rnd_fix=RND_FIX)
        self.log = d._log
        Top.current = self
        global _CARRY
        carry, _CARRY = _CARRY, None
        if isinstance(carry, dict):
            for k, v in carry.items():
                getattr(self.sim, k)[...] = v
        elif carry is None:                     # previous test failed: resynchronise
            self.log.warning("re-synchronising RTL and simulator state after a failed test")
            for i in range(self.G // 4):
                self.sim.gbuf[4 * i:4 * i + 4] = np.frombuffer(
                    (await self.bfm.read32(GBUF_BASE + 4 * i)).to_bytes(4, "little"), np.uint8)
            await self.program([isa.wload(0, 0, 64, 0, 32), isa.wload(1, 0, 64, 0, 32),
                                isa.lutload(0),
                                isa.matvec(ntok=ACC_DEPTH // 8, aaddr=0, astride=0,
                                           acc_mode=Acc.FIRST, acc_base=0)],
                               full=False, what="scrub")
        return self

    def finish(self):
        """Mark the end of a passing test: hand the matching state to the next test."""
        global _CARRY
        _CARRY = {k: getattr(self.sim, k).copy() for k in ("gbuf", "banks", "acc", "explut", "imem")}

    # -- memories (RTL via AXI, sim via its own AXI model) -----------------------
    async def gbuf_write(self, addr: int, data: bytes, only_diff: bool = True):
        """Write bytes at any alignment; words equal to the current image are skipped."""
        data = bytes(data)
        a0, a1 = addr & ~3, (addr + len(data) + 3) & ~3
        new = bytearray(self.sim.gbuf[a0:a1].tobytes())
        new[addr - a0:addr - a0 + len(data)] = data
        old = self.sim.gbuf[a0:a1].tobytes()
        for i in range(0, a1 - a0, 4):
            if only_diff and new[i:i + 4] == old[i:i + 4]:
                continue
            w = int.from_bytes(new[i:i + 4], "little")
            assert await self.bfm.write(GBUF_BASE + a0 + i, w) == OKAY
            assert self.sim.axi_write(GBUF_BASE + a0 + i, w) == OKAY

    async def load_prog(self, prog, pc: int = 0):
        words = isa.assemble(prog)
        for i, w in enumerate(words):
            a = IMEM_BASE + 32 * pc + 4 * i
            assert await self.bfm.write(a, int(w)) == OKAY
            assert self.sim.axi_write(a, int(w)) == OKAY

    async def wait_idle(self, max_cycles: int = 5_000_000):
        if int(self.dut.irq_o.value) == 0:
            await with_timeout(RisingEdge(self.dut.irq_o), max_cycles * CLK_NS, "ns")

    async def regs(self) -> dict:
        b = self.bfm
        s = await b.read32(REG_STATUS)
        return dict(busy=bool(s & 1), done=bool(s & 2), error=bool(s & 4), err_code=(s >> 8) & 0xFF,
                    status_rsvd=s & 0xFFFF00F8, pc=await b.read32(REG_PC),
                    instret=await b.read32(REG_INSTRET), passes=await b.read32(REG_PASSES),
                    cycles=await b.read32(REG_CYCLES))

    async def run(self, pc: int = 0, *, check=True, what="") -> dict:
        assert await self.bfm.write(REG_PC_START, pc) == OKAY
        self.sim.axi_write(REG_PC_START, pc)
        assert await self.bfm.write(REG_CTRL, CTRL_START) == OKAY
        await self.wait_idle()
        self.sim.axi_write(REG_CTRL, CTRL_START)          # autorun: runs to completion
        r = await self.regs()
        if check:
            s = self.sim.status()
            assert r["status_rsvd"] == 0
            got = {k: r[k] for k in ("busy", "done", "error", "err_code", "pc", "instret", "passes")}
            exp = {k: s[k] for k in got}
            assert got == exp, f"{what}: status RTL {got} != sim {exp}"
        return r

    async def gbuf_compare(self, lo: int = 0, hi: int | None = None, what=""):
        if hi is None and self.G > SCAN_MAX:
            await self.gbuf_compare(self.G - 4096, self.G, what)
            hi = SCAN_MAX
        hi = self.G if hi is None else hi
        lo, hi = lo & ~3, (hi + 3) & ~3
        got = np.zeros((hi - lo) // 4, dtype=np.uint32)
        for i in range(len(got)):
            got[i] = await self.bfm.read32(GBUF_BASE + lo + 4 * i)
        exp = self.sim.gbuf[lo:hi].view("<u4")
        if not np.array_equal(got, exp):
            bad = np.flatnonzero(got != exp)
            j = bad[0]
            raise AssertionError(f"{what}: GBUF mismatch at {len(bad)} words, first byte "
                                 f"{lo + 4 * j:#x}: RTL {got[j]:#010x} sim {exp[j]:#010x}")

    async def program(self, prog, *, pc=0, what="", lo=0, hi=None, full=True):
        """Load + run + compare status and GBUF. Appends END if missing."""
        prog = list(prog)
        if not prog or prog[-1].opcode != Op.END:
            prog.append(isa.end())
        await self.load_prog(prog, pc)
        r = await self.run(pc, what=what)
        if full or hi is not None:
            await self.gbuf_compare(lo, hi, what)
        return r


def top_test(fn):
    """cocotb.test that hands the state on to the next test when it passes."""
    async def wrapper(dut):
        await fn(dut)
        Top.current.finish()
    wrapper.__name__, wrapper.__qualname__, wrapper.__doc__ = fn.__name__, fn.__qualname__, fn.__doc__
    return cocotb.test()(wrapper)


def rand_bytes(rng, n):
    return rng.integers(0, 256, n, dtype=np.uint8).tobytes()


async def init_banks(t: Top, rng, base: int = 0):
    """Fill both banks completely (bitcells have no reset) from random GBUF data."""
    await t.gbuf_write(base, rand_bytes(rng, 32 * 64 + 32 * 80))
    await t.program([isa.wload(0, base, 64, 0, 32), isa.wload(1, base + 2048, 80, 0, 32)],
                    what="init banks", full=False)


# ---------------------------------------------------------------------------
# 1. register map, ID / CFG, SLVERR rules, protocol jitter
# ---------------------------------------------------------------------------
@top_test
async def test_regs(dut):
    t = await Top(dut).start(jitter=True, seed=1)
    b, sim, G = t.bfm, t.sim, t.G
    assert await b.read32(REG_ID) == 0x43504C31
    assert await b.read32(REG_CFG) == (G // 1024) | (256 << 16)
    r = await t.regs()
    assert not (r["busy"] or r["done"] or r["error"]) and r["pc"] == 0

    probe = [0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x20, 0x24, 0x28, 0x3C, 0x40, 0x100,
             0xFFFC, 0x10000, 0x10004, 0x10020, 0x11FFC, 0x12000, 0x20000, 0xFFFFC, 0x100000,
             GBUF_BASE + G - 4, GBUF_BASE + G, 0x1FFFFC, 0x1, 0x2, 0x3, 0x10001, 0x10002, 0x100003,
             0x100006]
    for a in probe:                                    # reads: data + response == sim
        assert await b.read(a) == sim.axi_read(a), hex(a)

    rng = np.random.default_rng(2)
    for a in probe:                                    # writes (not CTRL: START would run)
        if a == REG_CTRL:
            continue
        for _ in range(2):
            v, s = int(rng.integers(0, 1 << 32)), int(rng.integers(0, 16))
            assert await b.write(a, v, s) == sim.axi_write(a, v, s), hex(a)
    for a in probe:
        assert await b.read(a) == sim.axi_read(a), hex(a)

    # PC_START: only wstrb[0] matters, bits above 7 ignored; RO registers drop writes
    for v, s, exp in ((0x1AB, 0x1, 0xAB), (0x55, 0x2, 0xAB), (0x77, 0x0, 0xAB), (0xFFFFFF12, 0xF, 0x12)):
        assert await b.write(REG_PC_START, v, s) == OKAY
        assert await b.read32(REG_PC_START) == exp
    for a in (REG_ID, REG_STATUS, REG_PC, REG_CYCLES, REG_INSTRET, REG_PASSES, REG_CFG):
        before = await b.read32(a)
        assert await b.write(a, 0xDEADBEEF) == OKAY
        assert await b.read32(a) == before
    assert await b.read(REG_CTRL) == (0, OKAY)

    # START with an all-END IMEM (instruction 0x12 = END): done, INSTRET = 1, PC = 0x12
    await t.load_prog([isa.end()], 0x12)
    for i in range(0x12):
        await t.load_prog([isa.nop()], i)
    assert await b.write(REG_PC_START, 0x12) == OKAY
    sim.axi_write(REG_PC_START, 0x12)
    assert await b.write(REG_CTRL, CTRL_START, 0x0) == OKAY           # strobe 0: no START
    await ClockCycles(dut.clk_i, 20)
    assert (await t.regs())["done"] is False
    r = await t.run(0x12, what="END")
    assert r["done"] and r["pc"] == 0x12 and r["instret"] == 1 and r["cycles"] > 0
    assert int(dut.irq_o.value) == 1
    assert await b.write(REG_CTRL, CTRL_CLEAR) == OKAY
    sim.axi_write(REG_CTRL, CTRL_CLEAR)
    r = await t.regs()
    assert not r["done"] and not r["error"] and r["instret"] == 1      # CLEAR keeps counters
    assert int(dut.irq_o.value) == 0
    r = await t.run(0, what="NOPs + END")                              # 0x12 NOPs then END
    assert r["instret"] == 0x13
    t.log.info("register map / SLVERR rules OK (%d writes, %d reads, jittered AW/W/B/R)",
               b.n_wr, b.n_rd)


# ---------------------------------------------------------------------------
# 2. IMEM / GBUF read-back with byte strobes
# ---------------------------------------------------------------------------
@top_test
async def test_mem_readback(dut):
    t = await Top(dut).start(jitter=False)
    b, sim, G = t.bfm, t.sim, t.G
    rng = np.random.default_rng(3)
    # IMEM: every word of every instruction, then strobed overwrites
    words = rng.integers(0, 1 << 32, 256 * 8, dtype=np.uint64)
    for i, w in enumerate(words):
        assert await b.write(IMEM_BASE + 4 * i, int(w)) == OKAY
        sim.axi_write(IMEM_BASE + 4 * i, int(w))
    for i in rng.choice(2048, 300, replace=False):
        v, s = int(rng.integers(0, 1 << 32)), int(rng.integers(0, 16))
        assert await b.write(IMEM_BASE + 4 * int(i), v, s) == OKAY
        sim.axi_write(IMEM_BASE + 4 * int(i), v, s)
    for i in range(2048):
        assert await b.read(IMEM_BASE + 4 * i) == sim.axi_read(IMEM_BASE + 4 * i), i
    # GBUF: random words incl. first / last, strobes, jittered protocol for part of them
    idx = np.unique(np.concatenate([[0, 1, 2, 3, G // 4 - 1, G // 4 - 2],
                                    rng.choice(G // 4, 1500, replace=False)]))
    for n, i in enumerate(idx):
        a = GBUF_BASE + 4 * int(i)
        v, s = int(rng.integers(0, 1 << 32)), int(rng.integers(0, 16)) if n % 3 else 0xF
        assert await b.write(a, v, s, jitter=bool(n % 2)) == OKAY
        sim.axi_write(a, v, s)
    for n, i in enumerate(idx):
        a = GBUF_BASE + 4 * int(i)
        assert await b.read(a, jitter=bool(n % 2)) == sim.axi_read(a), hex(a)
    await t.gbuf_compare(what="full GBUF scan")
    t.log.info("IMEM 2048 words + GBUF %d words (full %d KiB scan) read back OK", len(idx), G // 1024)


# ---------------------------------------------------------------------------
# 3. every opcode and mode vs the ISA simulator
# ---------------------------------------------------------------------------
def directed_programs(G: int, rng):
    """(name, [instr]) pairs; data regions: A = activations 0x800.., outputs above."""
    A = 0x0800
    O = 0x1800
    lut = 0x1400
    sc = dict(scale=int(rng.integers(200, 900)), shift=14)
    mv = isa.matvec
    P = []
    P.append(("LUTLOAD", [isa.lutload(lut)]))
    for am in (Acc.NONE, Acc.FIRST, Acc.ADD):
        P.append((f"Q8 acc={am.name} fast unaligned",
                  [mv(ntok=5, aaddr=A, astride=64, acc_mode=am, out_mode=Out.Q8, oaddr=O + 3,
                      tstride=37, cstride=1, ncols=32, acc_base=40, **sc)]))
    P.append(("Q8 ncols=17 cstride=1, oaddr at line end",
              [mv(ntok=4, aaddr=A + 16, astride=48, out_mode=Out.Q8, oaddr=O + 15, tstride=50,
                  cstride=1, ncols=17, **sc)]))
    P.append(("Q8 transposed (tstride=1, cstride=48) bank 1",
              [mv(bank=1, ntok=7, aaddr=A, astride=32, out_mode=Out.Q8, oaddr=O, tstride=1,
                  cstride=48, ncols=32, **sc)]))
    P.append(("act_unsigned Q8",
              [mv(ntok=3, aaddr=A, astride=64, out_mode=Out.Q8, oaddr=O, tstride=32, cstride=1,
                  ncols=32, act_unsigned=True, scale=sc["scale"], shift=16)]))
    P.append(("RAW32 cstride=4", [mv(ntok=3, aaddr=A, out_mode=Out.RAW32, oaddr=O, tstride=128,
                                      cstride=4, ncols=32)]))
    P.append(("RAW32 cstride=6 straddling", [mv(ntok=2, aaddr=A, out_mode=Out.RAW32, oaddr=O + 1,
                                                 tstride=200, cstride=6, ncols=32)]))
    P.append(("RAW32 cstride=1 overlapping", [mv(ntok=2, aaddr=A, out_mode=Out.RAW32, oaddr=O + 5,
                                                  tstride=3, cstride=1, ncols=31)]))
    P.append(("Q8 cstride=0 (last column wins)", [mv(ntok=3, aaddr=A, out_mode=Out.Q8, oaddr=O + 7,
                                                      tstride=1, cstride=0, ncols=20, **sc)]))
    P.append(("Q8 tstride=0 (last token wins)", [mv(ntok=4, aaddr=A, astride=16, out_mode=Out.Q8,
                                                     oaddr=O, tstride=0, cstride=1, ncols=32, **sc)]))
    P.append(("FIRST/NONE then ADD/Q8", [
        mv(ntok=4, aaddr=A, acc_mode=Acc.FIRST, out_mode=Out.NONE, acc_base=1020),
        mv(bank=1, ntok=4, aaddr=A + 256, acc_mode=Acc.ADD, out_mode=Out.Q8, acc_base=1020,
           oaddr=O, tstride=32, cstride=1, ncols=32, **sc)]))
    for sbt in (0, 1):
        for causal in (0, 1):
            P.append((f"EXP+NORM sum_by_t={sbt} causal={causal}", [
                isa.lutload(lut),
                mv(ntok=6, aaddr=A, astride=64, out_mode=Out.EXP, oaddr=O, tstride=32 if sbt else 1,
                   cstride=1 if sbt else 32, ncols=29, sum_by_t=bool(sbt), sum_clr=True,
                   causal=bool(causal), qbase=3, kbase=1, scale=sc["scale"], shift=15),
                mv(ntok=6, aaddr=A, astride=64, out_mode=Out.EXP, oaddr=O + 512, tstride=32,
                   cstride=1, ncols=29, sum_by_t=bool(sbt), causal=bool(causal), qbase=3, kbase=2,
                   scale=sc["scale"], shift=15),
                mv(ntok=6, aaddr=A + 64, astride=64, out_mode=Out.NORM, oaddr=O + 1024,
                   tstride=32, cstride=1, ncols=29, sum_by_t=bool(sbt))]))
    P.append(("sum_clr with ntok=0, NORM of zero sums", [
        mv(ntok=0, aaddr=0, sum_clr=True, out_mode=Out.Q8),
        mv(ntok=2, aaddr=A, out_mode=Out.NORM, oaddr=O, tstride=40, cstride=1, ncols=32)]))
    P.append(("ncols=0 and out NONE write nothing", [
        mv(ntok=3, aaddr=A, out_mode=Out.Q8, oaddr=O, ncols=0, cstride=1, **sc),
        mv(ntok=3, aaddr=A, out_mode=Out.NONE, oaddr=O, ncols=32, cstride=1)]))
    P.append(("ncols=1, cstride=3", [mv(ntok=5, aaddr=A, out_mode=Out.Q8, oaddr=O, tstride=1,
                                        cstride=3, ncols=1, **sc)]))
    P.append(("aliasing: outputs feed the next activation vector", [
        mv(ntok=8, aaddr=A, astride=64, out_mode=Out.Q8, oaddr=A + 64 + 5, tstride=64,
           cstride=1, ncols=32, **sc)]))
    P.append(("aliasing transposed", [
        mv(ntok=6, aaddr=A, astride=16, out_mode=Out.Q8, oaddr=A + 40, tstride=3, cstride=2,
           ncols=32, **sc)]))
    P.append(("WLOAD partial (p0=5, np=7, pstride=0) + MATVEC", [
        isa.wload(0, A + 512, 0, 5, 7),
        mv(ntok=3, aaddr=A, out_mode=Out.RAW32, oaddr=O, tstride=128, cstride=4, ncols=32)]))
    P.append(("WLOAD p0=31 np=1 bank 1 + NOP", [
        isa.wload(1, A + 1024, 16, 31, 1), isa.nop(),
        mv(bank=1, ntok=2, aaddr=A, out_mode=Out.RAW32, oaddr=O, tstride=128, cstride=4, ncols=32)]))
    P.append(("long MATVEC (ACC entry wrap within the instruction)", [
        mv(ntok=140, aaddr=A, astride=16, acc_mode=Acc.FIRST, out_mode=Out.Q8, oaddr=O,
           tstride=32, cstride=1, ncols=32, acc_base=900, **sc),
        mv(ntok=140, aaddr=A + 16, astride=16, acc_mode=Acc.ADD, out_mode=Out.Q8, oaddr=O,
           tstride=32, cstride=1, ncols=32, acc_base=900, **sc)]))
    return P


@top_test
async def test_opcodes(dut):
    t = await Top(dut).start()
    rng = np.random.default_rng(5)
    await init_banks(t, rng)
    await t.gbuf_write(0x0800, rand_bytes(rng, 0x1000))                 # activations, LUT
    for name, prog in directed_programs(t.G, rng):
        r = await t.program(prog, what=name, lo=0, hi=0x8000 if t.G > 0x8000 else None)
        t.log.info("%-55s OK  cycles=%6d passes=%4d", name, r["cycles"], r["passes"])
    await t.gbuf_compare(what="final full image")


# ---------------------------------------------------------------------------
# 4. every error code, priorities, fetch past 255
# ---------------------------------------------------------------------------
def error_cases(G: int):
    mv, wl = isa.matvec, isa.wload
    big = 1 << 31
    q8 = dict(out_mode=Out.Q8, scale=1, shift=8)
    cases = []
    for op in range(5, 16):
        cases.append((f"opcode {op}", [Instr(opcode=op, ntok=3, aaddr=17)], Err.OPCODE))
    cases.append(("acc_mode 3", [Instr(opcode=Op.MATVEC, acc_mode=3, ntok=1)], Err.OPCODE))
    for om in (5, 6, 7):
        cases.append((f"out_mode {om}", [Instr(opcode=Op.MATVEC, out_mode=om, ntok=1)], Err.OPCODE))
    cases += [
        ("WLOAD gaddr%16", [wl(0, 8, 64, 0, 4)], Err.ALIGN),
        ("WLOAD pstride%16", [wl(0, 0, 65, 0, 4)], Err.ALIGN),
        ("LUTLOAD gaddr%16", [isa.lutload(4)], Err.ALIGN),
        ("MATVEC aaddr%16", [mv(ntok=1, aaddr=3)], Err.ALIGN),
        ("MATVEC astride%16", [mv(ntok=2, aaddr=0, astride=24)], Err.ALIGN),
        ("MATVEC astride%16, ntok=0", [mv(ntok=0, aaddr=0, astride=8)], Err.ALIGN),
        ("WLOAD np=0", [wl(0, 0, 64, 0, 0)], Err.PART),
        ("WLOAD p0+np=33", [wl(0, 0, 64, 31, 2)], Err.PART),
        ("WLOAD np=63", [wl(1, 0, 64, 0, 63)], Err.PART),
        ("MATVEC ncols=33", [mv(ntok=1, aaddr=0, ncols=33)], Err.PART),
        ("MATVEC ncols=63", [mv(ntok=1, aaddr=0, ncols=63, out_mode=Out.NONE)], Err.PART),
        ("WLOAD range", [wl(0, G - 48, 16, 0, 1)], Err.RANGE),
        ("WLOAD range via pstride", [wl(0, 0, G // 2, 0, 3)], Err.RANGE),
        ("WLOAD huge pstride", [wl(0, 0, big, 0, 2)], Err.RANGE),
        ("LUTLOAD range", [isa.lutload(G - 240)], Err.RANGE),
        ("LUTLOAD huge", [isa.lutload(0xFFFFFFF0)], Err.RANGE),
        ("MATVEC aaddr range", [mv(ntok=1, aaddr=G - 48)], Err.RANGE),
        ("MATVEC astride range", [mv(ntok=3, aaddr=0, astride=G // 2)], Err.RANGE),
        ("MATVEC huge astride", [mv(ntok=2, aaddr=0, astride=0xFFFFFFF0)], Err.RANGE),
        ("MATVEC ntok=65535 astride", [mv(ntok=65535, aaddr=0, astride=16)], Err.RANGE),
        ("MATVEC oaddr range", [mv(ntok=1, aaddr=0, oaddr=G - 31, cstride=1, ncols=32, **q8)], Err.RANGE),
        ("MATVEC tstride range", [mv(ntok=3, aaddr=0, oaddr=0, tstride=G // 2, ncols=1, **q8)], Err.RANGE),
        ("MATVEC cstride range", [mv(ntok=1, aaddr=0, oaddr=0, cstride=65535, ncols=32, **q8)], Err.RANGE),
        ("MATVEC RAW32 last byte", [mv(ntok=1, aaddr=0, oaddr=G - 3, ncols=1,
                                       out_mode=Out.RAW32)], Err.RANGE),
        ("MATVEC huge oaddr", [mv(ntok=1, aaddr=0, oaddr=big, ncols=1, **q8)], Err.RANGE),
        # no error: nothing touched
        ("ntok=0 huge addresses", [mv(ntok=0, aaddr=0xFFFFFFF0, oaddr=big, ncols=32, **q8)], Err.NONE),
        ("out NONE huge oaddr", [mv(ntok=1, aaddr=0, oaddr=big, ncols=32)], Err.NONE),
        ("ncols=0 huge oaddr", [mv(ntok=1, aaddr=0, oaddr=big, ncols=0, **q8)], Err.NONE),
        ("RAW32 last word in range", [mv(ntok=1, aaddr=0, oaddr=G - 4, ncols=1,
                                         out_mode=Out.RAW32)], Err.NONE),
        ("WLOAD last line in range", [wl(0, G - 64, 0, 0, 32)], Err.NONE),
        # priorities 1 > 2 > 4 > 3
        ("prio 1>2", [Instr(opcode=Op.MATVEC, acc_mode=3, ntok=1, aaddr=3)], Err.OPCODE),
        ("prio 2>4", [mv(ntok=1, aaddr=3, ncols=40)], Err.ALIGN),
        ("prio 4>3", [mv(ntok=1, aaddr=G, ncols=40)], Err.PART),
        ("prio WLOAD 2>4", [wl(0, 8, 64, 0, 0)], Err.ALIGN),
        ("prio WLOAD 4>3", [wl(0, G, 64, 31, 5)], Err.PART),
        # errors after some instructions ran: PC / INSTRET / PASSES of the partial run
        ("error after work", [mv(ntok=3, aaddr=0, oaddr=0x400, tstride=32, cstride=1, ncols=32, **q8),
                              isa.nop(), mv(ntok=1, aaddr=0, ncols=50)], Err.PART),
        ("reserved w0[15:14] ignored", [Instr(opcode=Op.NOP, rsvd=3)], Err.NONE),
    ]
    return cases


@top_test
async def test_errors(dut):
    t = await Top(dut).start()
    rng = np.random.default_rng(6)
    await init_banks(t, rng)
    await t.gbuf_write(0, rand_bytes(rng, 0x800))
    for n, (name, prog, err) in enumerate(error_cases(t.G)):
        pc = int(rng.integers(0, 200))
        r = await t.program(prog, pc=pc, what=name, lo=0, hi=0x1000)
        assert r["err_code"] == int(err), f"{name}: err {r['err_code']} != {int(err)}"
        assert r["error"] == (err != Err.NONE)
        t.log.info("%-32s err=%d pc=%3d instret=%d passes=%d", name, r["err_code"], r["pc"],
                   r["instret"], r["passes"])
    # fetch past instruction 255: two NOPs at 254 / 255 retire, then error 1 at PC = 255
    await t.load_prog([isa.nop(), isa.nop()], 254)
    r = await t.run(254, what="fetch past 255")
    assert r["error"] and r["err_code"] == Err.OPCODE and r["pc"] == 255 and r["instret"] == 2
    # error, then CLEAR, then a good run
    await t.bfm.write(REG_CTRL, CTRL_CLEAR)
    t.sim.axi_write(REG_CTRL, CTRL_CLEAR)
    r = await t.regs()
    assert not r["error"] and r["err_code"] == 0
    await t.program([isa.nop()], what="after CLEAR")
    await t.gbuf_compare(what="final full image")


# ---------------------------------------------------------------------------
# 5. busy-time rules (ARCH.md §3)
# ---------------------------------------------------------------------------
@top_test
async def test_busy_rules(dut):
    t = await Top(dut).start()
    b, G = t.bfm, t.G
    rng = np.random.default_rng(7)
    await init_banks(t, rng)
    await t.gbuf_write(0, rand_bytes(rng, 0x400))
    prog = [isa.matvec(ntok=400, aaddr=0, astride=0, out_mode=Out.Q8, oaddr=0x400, tstride=0,
                       cstride=1, ncols=32, scale=300, shift=14), isa.end()]
    await t.load_prog(prog)
    sim = t.sim
    sim.autorun = False                      # START only sets busy; finish() runs it
    gw = int.from_bytes(t.sim.gbuf[0x200:0x204].tobytes(), "little")
    iw = int(t.sim.imem[0, 1])
    assert await b.write(REG_CTRL, CTRL_START) == OKAY
    sim.axi_write(REG_CTRL, CTRL_START)
    accesses = [("r", GBUF_BASE + 0x200), ("w", GBUF_BASE + 0x200, 0x12345678),
                ("r", IMEM_BASE + 4), ("w", IMEM_BASE + 4, 0x0BADF00D), ("r", REG_STATUS),
                ("w", REG_PC_START, 7), ("r", REG_PC_START), ("w", REG_CTRL, CTRL_START),
                ("w", REG_CTRL, CTRL_CLEAR), ("r", REG_STATUS), ("r", REG_ID), ("r", 0x24),
                ("w", GBUF_BASE + G, 1)]
    for acc in accesses:
        if acc[0] == "r":
            got, exp = await b.read(acc[1]), sim.axi_read(acc[1])
            if acc[1] == REG_STATUS:
                assert got[0] & 1, "not busy during the run"
            assert got == exp, f"{acc}: {got} != {exp}"
        else:
            got, exp = await b.write(acc[1], acc[2]), sim.axi_write(acc[1], acc[2])
            assert got == exp, f"{acc}: {got} != {exp}"
    assert int(dut.irq_o.value) == 0
    sim.finish()
    await t.wait_idle()
    r = await t.regs()
    s = sim.status()
    assert (r["done"], r["error"], r["pc"], r["instret"], r["passes"]) == \
        (s["done"], s["error"], s["pc"], s["instret"], s["passes"]), (r, s)
    assert await b.read32(GBUF_BASE + 0x200) == gw              # busy-time write dropped
    assert await b.read32(IMEM_BASE + 4) == iw
    assert await b.read32(REG_PC_START) == 7                    # register writes still work
    sim.autorun = True
    await t.gbuf_compare(what="after busy run")
    t.log.info("busy rules OK: %d cycles for 400 passes (%.2f cycles/pass)", r["cycles"],
               r["cycles"] / 400)
    assert r["cycles"] / 400 < PAPER_CYCLES_PER_PASS + 1


# ---------------------------------------------------------------------------
# 6. randomised programs vs the simulator
# ---------------------------------------------------------------------------
class Fuzzer:
    def __init__(self, G: int, rng):
        self.G, self.rng = G, rng

    def r(self, lo, hi):            # inclusive
        return int(self.rng.integers(lo, hi + 1))

    def line(self, hi):
        return 16 * self.r(0, max(0, (hi - 1) // 16))

    def wload(self):
        p0 = self.r(0, 31)
        np_ = self.r(1, 32 - p0)
        pstride = 16 * self.rng.choice([0, 1, 4, 5, 8])
        hi = self.G - (np_ - 1) * pstride - 64
        return isa.wload(self.r(0, 1), self.line(hi), int(pstride), p0, np_)

    def matvec(self):
        r = self.r
        om = int(self.rng.choice([Out.NONE, Out.Q8, Out.Q8, Out.EXP, Out.NORM, Out.RAW32]))
        ntok = int(self.rng.choice([0, 1, 2, 3, 5, 8, 13, r(1, 40)]))
        ncols = int(self.rng.choice([0, 1, 7, 8, 9, 31, 32, r(0, 32)]))
        width = 4 if om == Out.RAW32 else 1
        astride = 16 * int(self.rng.choice([0, 1, 2, 4, 5, r(0, 12)]))
        mode = self.rng.random()
        if mode < 0.35:                                     # contiguous rows
            cstride, tstride = width, r(0, 3) * 16 + width * ncols
        elif mode < 0.55:                                   # transposed
            cstride, tstride = int(self.rng.choice([33, 48, 64, 100])), int(self.rng.choice([1, width]))
        elif mode < 0.7:                                    # overlapping
            cstride, tstride = int(self.rng.choice([0, 1, 2, 3])), r(0, 8)
        else:
            cstride, tstride = r(0, 70), r(0, 300)
        amax = (max(ntok, 1) - 1) * astride + 64
        aaddr = self.line(self.G - amax)
        span = (max(ntok, 1) - 1) * tstride + (max(ncols, 1) - 1) * cstride + width
        if self.rng.random() < 0.2:                         # aim at the activations
            oaddr = max(0, min(self.G - span, aaddr + r(-64, 200)))
        else:
            oaddr = r(0, max(0, self.G - span))
        if om == Out.NORM:
            scale, shift = 0, 0
        elif self.rng.random() < 0.15:
            scale, shift = r(0, 65535), r(0, 31)
        else:
            scale, shift = r(1, 3000), r(10, 20)
        return isa.matvec(bank=r(0, 1), ntok=ntok, aaddr=aaddr, astride=astride,
                          acc_mode=int(self.rng.choice([Acc.NONE, Acc.FIRST, Acc.ADD])),
                          out_mode=om, oaddr=oaddr, tstride=tstride, cstride=cstride, ncols=ncols,
                          scale=scale, shift=shift, acc_base=r(0, 65535),
                          act_unsigned=bool(r(0, 1)), sum_by_t=bool(r(0, 1)),
                          sum_clr=self.rng.random() < 0.3, causal=bool(r(0, 1)),
                          qbase=r(0, 40), kbase=r(0, 40))

    def illegal(self):
        r = self.r
        k = r(0, 6)
        if k == 0:
            return Instr(opcode=r(5, 15), ntok=r(0, 5))
        if k == 1:
            return Instr(opcode=Op.MATVEC, acc_mode=3 if r(0, 1) else 0,
                         out_mode=r(5, 7) if r(0, 1) else 1, ntok=1)
        if k == 2:
            return isa.matvec(ntok=r(1, 3), aaddr=16 * r(0, 10) + r(1, 15))
        if k == 3:
            return isa.wload(0, 16 * r(0, 10), 64, r(0, 31), r(33, 63))
        if k == 4:
            return isa.matvec(ntok=1, aaddr=0, ncols=r(33, 63), out_mode=Out.Q8)
        if k == 5:
            return isa.matvec(ntok=r(1, 4), aaddr=self.G - 16 * r(1, 3))
        return isa.lutload(self.G - 16 * r(1, 15))

    def program(self, legal: bool):
        prog = []
        for _ in range(self.r(2, 7)):
            x = self.rng.random()
            prog.append(self.wload() if x < 0.2 else isa.lutload(self.line(self.G - 256))
                        if x < 0.27 else isa.nop() if x < 0.3 else self.matvec())
        if not legal:
            prog.insert(self.r(0, len(prog)), self.illegal())
        return prog


@top_test
async def test_fuzz(dut):
    n_prog = int(os.environ.get("CPL_FUZZ_N", "36"))
    seed = int(os.environ.get("CPL_FUZZ_SEED", "2024"))
    t = await Top(dut).start(jitter=True, seed=seed)
    rng = np.random.default_rng(seed)
    G = t.G
    await init_banks(t, rng)
    await t.gbuf_write(0, rand_bytes(rng, G), only_diff=False)
    fz = Fuzzer(G, rng)
    stats = dict(passes=0, cycles=0, errors={})
    full_every = int(os.environ.get("CPL_FUZZ_FULL_EVERY", "1" if G <= 32768 else "4"))
    for n in range(n_prog):
        legal = n % 5 != 4
        prog = fz.program(legal)
        # refresh some data so the next program reads new activations / weights
        for _ in range(8):
            a = 4 * int(rng.integers(0, G // 4 - 16))
            await t.gbuf_write(a, rand_bytes(rng, 64))
        full = (n % full_every == full_every - 1) or n == n_prog - 1
        r = await t.program(prog, pc=int(rng.integers(0, 256 - len(prog) - 1)),
                            what=f"fuzz #{n}\n" + isa.disassemble(prog), full=full)
        stats["passes"] += r["passes"]
        stats["cycles"] += r["cycles"]
        stats["errors"][r["err_code"]] = stats["errors"].get(r["err_code"], 0) + 1
        t.log.info("fuzz #%2d %-7s %d instr -> err=%d instret=%d passes=%4d cycles=%6d%s", n,
                   "legal" if legal else "illegal", len(prog), r["err_code"], r["instret"],
                   r["passes"], r["cycles"], " (full GBUF compared)" if full else "")
    t.log.info("fuzz: %d programs, %d passes, %d cycles, error codes %s", n_prog, stats["passes"],
               stats["cycles"], stats["errors"])


# ---------------------------------------------------------------------------
# 7. end to end: compiler programs through the host driver
# ---------------------------------------------------------------------------
def _fit(G, candidates, build):
    """First candidate whose layout fits this GBUF (small builds get small problems)."""
    for c in candidates:
        try:
            b = build(*c)
        except MemoryError:                       # planner: GBUF overflow
            continue
        if b.layout.top <= G:
            return c, b
    raise AssertionError(f"no candidate layout fits a {G} B GBUF")


async def _execute(t: Top, build, what):
    be = CocotbBackend(t.bfm)
    dev = Device(be)
    await be.run(dev.check_id)
    outs, tot = await be.run(dev.execute, build, timeout=1e9)
    for k in build.layout.outputs:
        assert np.array_equal(outs[k], build.ref[k]), f"{what}: output {k} != integer reference"
    sim_outs, sim_tot, _ = C.run_sim(build, sim=t.sim)          # same flow on the carried sim
    for k in build.layout.outputs:
        assert np.array_equal(outs[k], sim_outs[k]), f"{what}: output {k} != simulator"
    assert (tot["instret"], tot["passes"]) == (sim_tot["instret"], sim_tot["passes"]), (tot, sim_tot)
    return outs, tot


@top_test
async def test_encoder_head(dut):
    t = await Top(dut).start()
    rng = np.random.default_rng(8)

    def build(N, dm, d):
        X = rng.standard_normal((N, dm))
        Ws = [rng.standard_normal((dm, d)) / np.sqrt(dm) for _ in range(3)]
        cal = C.calibrate(X, *Ws)
        b = C.encoder_attention_head(cal.xq, cal.wq, cal.wk, cal.wv, cal.scales, gbuf_bytes=t.G)
        b.meta = (X, Ws, cal)
        return b

    (N, dm, d), b = _fit(t.G, ((128, 64, 64), (96, 64, 64), (64, 64, 64), (48, 32, 32), (32, 16, 16)), build)
    outs, tot = await _execute(t, b, f"encoder N={N}")
    X, Ws, cal = b.meta
    cos = C.cosine(outs["O"] * cal.scales.out_scale, C.float_attention(X, *Ws))
    t.log.info("encoder head N=%d d_model=%d d=%d: %d instr in %d run(s), %d passes, %d cycles = "
               "%.2f cycles/pass (paper lower bound %d), cosine vs float %.4f", N, dm, d,
               tot["instret"], tot["runs"], tot["passes"], tot["cycles"],
               tot["cycles"] / tot["passes"], PAPER_CYCLES_PER_PASS, cos)
    assert cos > 0.95


@top_test
async def test_decoder_step(dut):
    t = await Top(dut).start()
    rng = np.random.default_rng(9)

    def build(L, dm, d):
        X = rng.standard_normal((L + 1, dm))
        Ws = [rng.standard_normal((dm, d)) / np.sqrt(dm) for _ in range(3)]
        cal = C.calibrate(X, *Ws, causal=True)
        kc = C.ref_matmul(cal.xq[:L], cal.wk, cal.scales.k)
        vc = C.ref_matmul(cal.xq[:L], cal.wv, cal.scales.v)
        b = C.decoder_attention_step(cal.xq[L:], cal.wq, cal.wk, cal.wv, kc, vc, cal.scales,
                                     gbuf_bytes=t.G)
        b.meta = (X, Ws, cal)
        return b

    (L, dm, d), b = _fit(t.G, ((100, 64, 64), (40, 32, 32)), build)
    outs, tot = await _execute(t, b, "decoder step")
    X, Ws, cal = b.meta
    cos = C.cosine(outs["O"] * cal.scales.out_scale, C.float_attention(X, *Ws, causal=True)[L:])
    t.log.info("decoder step L=%d d=%d: %d instr, %d passes, %d cycles = %.2f cycles/pass, "
               "cosine vs float %.4f", L, d, tot["instret"], tot["passes"], tot["cycles"],
               tot["cycles"] / tot["passes"], cos)
    assert cos > 0.9


@top_test
async def test_matmul_raw32_cross(dut):
    """Raw int32 matmul (K > 64 through the ACC buffer) and cross attention."""
    t = await Top(dut).start()
    rng = np.random.default_rng(10)

    def mm(N, K, M):
        X, W = rng.integers(-128, 128, (N, K)), rng.integers(-128, 128, (K, M))
        b = C.matmul(X, W, raw32=True, gbuf_bytes=t.G)
        b.meta = (X, W)
        return b

    (N, K, M), b = _fit(t.G, ((40, 160, 48), (24, 96, 32)), mm)
    outs, tot = await _execute(t, b, "matmul raw32")
    assert np.array_equal(outs["Y"], b.meta[0] @ b.meta[1])
    t.log.info("matmul %dx%dx%d raw32: %d passes, %d cycles", N, K, M, tot["passes"], tot["cycles"])

    def cross(Nd, Ne, d):
        Xd, He = rng.standard_normal((Nd, d)), rng.standard_normal((Ne, d))
        Ws = [rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(3)]
        cal = C.calibrate(Xd, *Ws, X_kv=He)
        return C.cross_attention(cal.xq, cal.xkv, cal.wq, cal.wk, cal.wv, cal.scales, gbuf_bytes=t.G)

    (Nd, Ne, d), b = _fit(t.G, ((12, 70, 64), (8, 40, 32)), cross)
    outs, tot = await _execute(t, b, "cross attention")
    t.log.info("cross attention %dx%d d=%d: %d passes, %d cycles", Nd, Ne, d, tot["passes"],
               tot["cycles"])
