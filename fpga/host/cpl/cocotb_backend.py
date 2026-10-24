"""
cocotb backend for cpl.driver.Device: MMIO through the AXI4-Lite BFM of the
cpl_top testbench (fpga/tb/axil_bfm.py), i.e. through the real AXI port of
the RTL.

The driver is synchronous (``read32`` returns an int) and cocotb 2.x has no
thread bridge, so the synchronous code runs in a worker thread and every
MMIO access is handed to the cocotb task, which performs it with the BFM and
hands the result back:

    be = CocotbBackend(bfm)
    outs, totals = await be.run(lambda: Device(be).execute(build, timeout=1e6))

While the worker thread computes, the cocotb task blocks in ``queue.get()``,
so no simulation time passes between accesses: the cycle-level behaviour is
deterministic and independent of host speed. A SLVERR response raises
``cpl.sim.SlvErr`` in the worker, like ``SimBackend``. Only the cocotb task
touches simulator handles.

This module imports nothing from cocotb itself (the BFM does), so it can be
imported -- not used -- outside a simulation.
"""
from __future__ import annotations

import queue
import threading

import numpy as np

from .sim import SlvErr

_OKAY = 0


class CocotbBackend:
    def __init__(self, bfm):
        self.bfm = bfm
        self._req: queue.Queue = queue.Queue()
        self._rsp: queue.Queue = queue.Queue()
        self._active = False

    # -- synchronous MMIO (called from the worker thread) ----------------------------
    def _call(self, req):
        if not self._active:
            raise RuntimeError("CocotbBackend: MMIO outside CocotbBackend.run()")
        self._req.put(req)
        ok, val = self._rsp.get()
        if not ok:
            raise SlvErr(val)
        return val

    def read32(self, addr: int) -> int:
        return self._call(("r", addr))

    def write32(self, addr: int, value: int) -> None:
        self._call(("w", addr, value & 0xFFFFFFFF))

    def write_block(self, addr: int, words) -> None:
        self._call(("wb", addr, np.asarray(words, dtype=np.uint32).copy()))

    def read_block(self, addr: int, n: int) -> np.ndarray:
        return self._call(("rb", addr, n))

    # -- the cocotb side --------------------------------------------------------------
    async def _serve(self, req):
        kind, addr = req[0], req[1]
        if kind == "r":
            d, r = await self.bfm.read(addr)
            return (True, d) if r == _OKAY else (False, f"read {addr:#x}")
        if kind == "w":
            r = await self.bfm.write(addr, req[2])
            return (True, None) if r == _OKAY else (False, f"write {addr:#x}")
        if kind == "wb":
            for i, v in enumerate(req[2]):
                if await self.bfm.write(addr + 4 * i, int(v)) != _OKAY:
                    return False, f"write {addr + 4 * i:#x}"
            return True, None
        if kind == "rb":
            out = np.zeros(req[2], dtype=np.uint32)
            for i in range(req[2]):
                d, r = await self.bfm.read(addr + 4 * i)
                if r != _OKAY:
                    return False, f"read {addr + 4 * i:#x}"
                out[i] = d
            return True, out
        raise ValueError(kind)

    async def run(self, fn, *args, **kw):
        """Run the synchronous ``fn(*args, **kw)`` (which uses this backend) to completion."""
        result: dict = {}

        def worker():
            try:
                result["v"] = fn(*args, **kw)
            except BaseException as e:   # re-raised in the cocotb task
                result["e"] = e
            finally:
                self._req.put(None)

        self._active = True
        th = threading.Thread(target=worker, name="cpl-driver", daemon=True)
        th.start()
        try:
            while True:
                req = self._req.get()
                if req is None:
                    break
                self._rsp.put(await self._serve(req))
        finally:
            self._active = False
            th.join()
        if "e" in result:
            raise result["e"]
        return result.get("v")
