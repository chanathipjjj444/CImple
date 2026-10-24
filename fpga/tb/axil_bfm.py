"""
Small cocotb AXI4-Lite master for the cpl_top tests (ARCH.md §3).

    bfm = AxilMaster(dut, dut.clk_i)            # signals s_axil_*
    resp = await bfm.write(addr, data, strb=0xF)
    data, resp = await bfm.read(addr)
    await bfm.write32(addr, data) / await bfm.read32(addr)   # raise SlvErr on SLVERR

Timing discipline: every input is driven and every output sampled at the
FALLING edge of the clock. The slave's READY / VALID outputs are registers
(cpl_axil), so a value sampled at the falling edge is the value the next
rising edge acts on: VALID=1 and READY=1 sampled together means the handshake
happens at that rising edge. VALID is held until the handshake completes
(AXI rule A3.2.1); the master never waits for READY before asserting VALID.

``jitter=True`` (per call or per master) randomises, from a seeded numpy
Generator: the delay of AW and of W (independently, so W may come before
AW), and BREADY / RREADY back-pressure -- the protocol corners cpl_axil must
handle. One transaction at a time (a cocotb Lock), like the SmartConnect
configuration ARCH.md assumes.
"""
from __future__ import annotations

import numpy as np
from cocotb.triggers import FallingEdge, Lock

OKAY, SLVERR = 0, 2


class SlvErr(Exception):
    """An access answered with SLVERR (same meaning as cpl.sim.SlvErr)."""


class AxilMaster:
    def __init__(self, dut, clk, prefix: str = "s_axil", *, jitter: bool = False,
                 seed: int = 0, timeout: int = 10000):
        self.clk = clk
        self.jitter = jitter
        self.rng = np.random.default_rng(seed)
        self.timeout = timeout
        self.lock = Lock()
        self.n_rd = self.n_wr = 0
        g = lambda n: getattr(dut, f"{prefix}_{n}")  # noqa: E731
        self.awaddr, self.awprot, self.awvalid, self.awready = (g(n) for n in
                                                                ("awaddr", "awprot", "awvalid", "awready"))
        self.wdata, self.wstrb, self.wvalid, self.wready = (g(n) for n in
                                                            ("wdata", "wstrb", "wvalid", "wready"))
        self.bresp, self.bvalid, self.bready = (g(n) for n in ("bresp", "bvalid", "bready"))
        self.araddr, self.arprot, self.arvalid, self.arready = (g(n) for n in
                                                                ("araddr", "arprot", "arvalid", "arready"))
        self.rdata, self.rresp, self.rvalid, self.rready = (g(n) for n in
                                                            ("rdata", "rresp", "rvalid", "rready"))
        for s in (self.awvalid, self.wvalid, self.arvalid, self.bready, self.rready,
                  self.awaddr, self.awprot, self.wdata, self.wstrb, self.araddr, self.arprot):
            s.value = 0

    def _delay(self, jitter: bool, hi: int = 4) -> int:
        return int(self.rng.integers(0, hi)) if jitter else 0

    async def write(self, addr: int, data: int, strb: int = 0xF, *, jitter: bool | None = None) -> int:
        """One AXI4-Lite write; returns BRESP."""
        jit = self.jitter if jitter is None else jitter
        async with self.lock:
            d_aw, d_w, d_b = self._delay(jit), self._delay(jit), self._delay(jit)
            aw_on = w_on = aw_hs = w_hs = False
            aw_done = w_done = False
            cyc = 0
            while not (aw_done and w_done):
                await FallingEdge(self.clk)
                if aw_on and aw_hs:                      # handshake at the last rising edge
                    self.awvalid.value = 0
                    aw_on, aw_done = False, True
                if w_on and w_hs:
                    self.wvalid.value = 0
                    w_on, w_done = False, True
                if not aw_on and not aw_done and cyc >= d_aw:
                    self.awaddr.value = addr & 0x1FFFFF
                    self.awvalid.value = 1
                    aw_on = True
                if not w_on and not w_done and cyc >= d_w:
                    self.wdata.value = data & 0xFFFFFFFF
                    self.wstrb.value = strb & 0xF
                    self.wvalid.value = 1
                    w_on = True
                aw_hs = aw_on and int(self.awready.value) == 1
                w_hs = w_on and int(self.wready.value) == 1
                cyc += 1
                if cyc > self.timeout:
                    raise TimeoutError(f"AXI write {addr:#x}: AW/W not accepted")
            # B channel
            cyc = 0
            while True:
                if cyc >= d_b:
                    self.bready.value = 1
                    if int(self.bvalid.value) == 1:
                        resp = int(self.bresp.value)
                        await FallingEdge(self.clk)      # handshake at this rising edge
                        self.bready.value = 0
                        break
                cyc += 1
                if cyc > self.timeout:
                    raise TimeoutError(f"AXI write {addr:#x}: no BVALID")
                await FallingEdge(self.clk)
            self.n_wr += 1
            return resp

    async def read(self, addr: int, *, jitter: bool | None = None) -> tuple[int, int]:
        """One AXI4-Lite read; returns (RDATA, RRESP)."""
        jit = self.jitter if jitter is None else jitter
        async with self.lock:
            d_ar, d_r = self._delay(jit), self._delay(jit)
            for _ in range(d_ar):
                await FallingEdge(self.clk)
            await FallingEdge(self.clk)
            self.araddr.value = addr & 0x1FFFFF
            self.arvalid.value = 1
            cyc = 0
            while True:
                hs = int(self.arready.value) == 1
                await FallingEdge(self.clk)
                if hs:
                    break
                cyc += 1
                if cyc > self.timeout:
                    raise TimeoutError(f"AXI read {addr:#x}: AR not accepted")
            self.arvalid.value = 0
            cyc = 0
            while True:
                if cyc >= d_r:
                    self.rready.value = 1
                    if int(self.rvalid.value) == 1:
                        data, resp = int(self.rdata.value), int(self.rresp.value)
                        await FallingEdge(self.clk)
                        self.rready.value = 0
                        break
                cyc += 1
                if cyc > self.timeout:
                    raise TimeoutError(f"AXI read {addr:#x}: no RVALID")
                await FallingEdge(self.clk)
            self.n_rd += 1
            return data, resp

    # -- convenience -----------------------------------------------------------------
    async def write32(self, addr: int, data: int, strb: int = 0xF) -> None:
        if await self.write(addr, data, strb) != OKAY:
            raise SlvErr(f"write {addr:#x}")

    async def read32(self, addr: int) -> int:
        d, r = await self.read(addr)
        if r != OKAY:
            raise SlvErr(f"read {addr:#x}")
        return d

    async def write_words(self, addr: int, words) -> None:
        for i, w in enumerate(np.asarray(words, dtype=np.uint32)):
            await self.write32(addr + 4 * i, int(w))

    async def read_words(self, addr: int, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.uint32)
        for i in range(n):
            out[i] = await self.read32(addr + 4 * i)
        return out
