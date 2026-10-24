"""
Host driver for the CIMple-FPGA accelerator (ARCH.md §3 AXI4-Lite map).

``Device`` talks to the accelerator through a minimal MMIO backend:

    read32(addr) -> int            addr = byte offset inside the 2 MiB window
    write32(addr, value)
    write_block(addr, words)       optional bulk path (np.uint32 array)
    read_block(addr, nwords)       optional bulk path -> np.uint32 array

Backends here:
  * SimBackend      -- the bit-exact ISA simulator (cpl.sim.CplSim)
  * DevMemBackend   -- mmap of /dev/mem (physical base address) or a UIO node,
                       for Linux on the Versal APU (the accelerator's AXI4-Lite
                       slave behind the CIPS M_AXI_FPD/LPD port)
  * XsdbBackend     -- JTAG through Vivado's xsdb/xsct (mwr / mrd) in one
                       persistent subprocess
The cocotb backend lives in fpga/tb (it drives the AXI4-Lite BFM).

An access answered with SLVERR raises ``SlvErr`` (SimBackend, XsdbBackend);
on /dev/mem an SLVERR becomes a bus error (SIGBUS / SError) of the process,
so ``Device`` never issues IMEM/GBUF accesses while ``busy``.
"""
from __future__ import annotations

import mmap
import os
import queue
import re
import subprocess
import threading
import time

import numpy as np

from . import isa
from .sim import (CTRL_CLEAR, CTRL_START, GBUF_BASE, ID_VALUE, IMEM_BASE, REG_CFG, REG_CTRL,
                  REG_CYCLES, REG_ID, REG_INSTRET, REG_PASSES, REG_PC, REG_PC_START, REG_STATUS,
                  ST_BUSY, ST_DONE, ST_ERROR, CplSim, SlvErr)

WINDOW = 1 << 21


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class SimBackend:
    """MMIO onto an in-process CplSim (AXI semantics incl. SLVERR)."""

    def __init__(self, sim: CplSim | None = None, **simkw):
        self.sim = CplSim(**simkw) if sim is None else sim

    def read32(self, addr: int) -> int:
        return self.sim.read32(addr)

    def write32(self, addr: int, value: int) -> None:
        self.sim.write32(addr, value)

    def _gbuf_span(self, addr: int, n: int):
        a = addr - GBUF_BASE
        ok = addr % 4 == 0 and 0 <= a and a + 4 * n <= self.sim.gbuf_bytes and not self.sim.busy
        return a if ok else None

    def write_block(self, addr: int, words) -> None:
        w = np.asarray(words, dtype=np.uint32)
        a = self._gbuf_span(addr, w.size)
        if a is None:                      # not a plain GBUF span: word by word (SLVERR rules)
            for i, v in enumerate(w):
                self.write32(addr + 4 * i, int(v))
            return
        self.sim.gbuf[a:a + 4 * w.size] = w.astype("<u4").view(np.uint8)

    def read_block(self, addr: int, n: int) -> np.ndarray:
        a = self._gbuf_span(addr, n)
        if a is None:
            return np.array([self.read32(addr + 4 * i) for i in range(n)], dtype=np.uint32)
        return self.sim.gbuf[a:a + 4 * n].copy().view("<u4").astype(np.uint32)


class DevMemBackend:
    """mmap of the AXI4-Lite window from Linux user space.

    DevMemBackend(base_addr=0xA4000000)            -> /dev/mem at that physical address
    DevMemBackend(uio="/dev/uio0")                 -> UIO map 0 (size from sysfs)
    Every access is a single aligned 32-bit load/store through a numpy uint32
    view (never memcpy: wide or unaligned accesses to Device memory fault on
    the A72, and the slave is AXI4-Lite).
    """

    def __init__(self, base_addr: int | None = None, size: int = WINDOW, *,
                 path: str = "/dev/mem", uio: str | None = None):
        if uio is not None:
            path, offset = uio, 0
            name = os.path.basename(uio)
            try:
                with open(f"/sys/class/uio/{name}/maps/map0/size") as f:
                    size = int(f.read().strip(), 0)
            except OSError:
                pass
        else:
            if base_addr is None:
                raise ValueError("base_addr is required for /dev/mem")
            offset = base_addr
        if offset % mmap.PAGESIZE:
            raise ValueError("base address must be page aligned")
        self.fd = os.open(path, os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
                            offset=offset)
        self.w = np.frombuffer(self.mm, dtype=np.uint32)

    def read32(self, addr: int) -> int:
        return int(self.w[addr >> 2])

    def write32(self, addr: int, value: int) -> None:
        self.w[addr >> 2] = value & 0xFFFFFFFF

    def write_block(self, addr: int, words) -> None:
        i0 = addr >> 2
        for i, v in enumerate(np.asarray(words, dtype=np.uint32)):
            self.w[i0 + i] = v

    def read_block(self, addr: int, n: int) -> np.ndarray:
        i0 = addr >> 2
        return np.array([self.w[i0 + i] for i in range(n)], dtype=np.uint32)

    def close(self) -> None:
        del self.w
        self.mm.close()
        os.close(self.fd)


class XsdbError(RuntimeError):
    pass


def hw_server_url(url: str | None) -> str | None:
    """xsdb ``connect -url`` wants "tcp:<host>:<port>"; xsdb_smoke.tcl /
    program_jtag.tcl take "<host>:<port>" and add "tcp:" themselves. Accept
    both forms here so one string works everywhere: add "tcp:" when no
    scheme is given ("host:port" or a bare "host")."""
    if not url:
        return url
    _, sep, rest = url.partition(":")
    if sep and not rest.isdigit():
        return url                      # already "tcp:host:port" (or another scheme)
    return "tcp:" + url


class XsdbBackend:
    """JTAG MMIO through one persistent xsdb (or xsct) process.

    Command syntax relied on (Xilinx Software Command-Line Tool, xsdb):
      connect [-url tcp:<host>:<port>]                      hw_server connection
                                                            (``url`` may omit "tcp:")
      targets -set -nocase -filter {name =~ "<pattern>"}    select the access target
      mwr -force <addr> {<w0> <w1> ...}                     write 32-bit words
      mrd -force <addr> <n>                                 read n words; result text
                                                            is one "<ADDR>:   <WORD>" line
                                                            per word, both in hex
    Every command is wrapped as one Tcl line

      puts "@@BEGIN"; set __rc [catch {<cmd>} __r]; puts "@@RC $__rc"; puts $__r; puts "@@END"

    so the reply is delimited, a Tcl error (e.g. a bus error on SLVERR) gives
    rc=1 with the message, and interactive prompts are ignored. The default
    target pattern "*A72*#0" accesses the PL through APU core 0's view of the
    address map; pass ``target`` (e.g. "DPC" or an AXI-AP) or ``init`` to change it.
    Untestable here; tests/fake_xsdb.py implements the same protocol.
    """

    CHUNK = 256          # words per mwr / mrd command

    def __init__(self, base_addr: int, *, xsdb: str = "xsdb", url: str | None = None,
                 target: str | None = "*A72*#0", init: list | None = None,
                 cmd: list | None = None, timeout: float = 30.0):
        self.base = base_addr
        self.timeout = timeout
        self.proc = subprocess.Popen(cmd or [xsdb], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        if init is None:
            url = hw_server_url(url)
            init = ["connect" + (f" -url {url}" if url else "")]
            if target:
                init.append(f'targets -set -nocase -filter {{name =~ "{target}"}}')
        self.init = list(init)          # the commands actually sent (tests, logs)
        for c in init:
            self.tcl(c)

    def _reader(self):
        for line in self.proc.stdout:
            self._q.put(line.rstrip("\r\n"))
        self._q.put(None)

    def _line(self, deadline: float) -> str:
        try:
            line = self._q.get(timeout=max(deadline - time.monotonic(), 0.001))
        except queue.Empty:
            raise TimeoutError("xsdb did not answer") from None
        if line is None:
            raise XsdbError("xsdb exited")
        return line

    def tcl(self, cmd: str) -> str:
        """Run one xsdb command; return its result text, raise XsdbError on a Tcl error."""
        assert "\n" not in cmd
        self.proc.stdin.write(f'puts "@@BEGIN"; set __rc [catch {{{cmd}}} __r]; '
                              f'puts "@@RC $__rc"; puts $__r; puts "@@END"\n')
        self.proc.stdin.flush()
        deadline = time.monotonic() + self.timeout
        while "@@BEGIN" not in self._line(deadline):
            pass
        m = None
        while m is None:
            m = re.search(r"@@RC (\d+)", self._line(deadline))
        rc, out = int(m.group(1)), []
        while True:
            line = self._line(deadline)
            if "@@END" in line:
                break
            out.append(line)
        text = "\n".join(out)
        if rc:
            raise XsdbError(f"{cmd}: {text}")
        return text

    def write_block(self, addr: int, words) -> None:
        w = np.asarray(words, dtype=np.uint32)
        for i in range(0, w.size, self.CHUNK):
            vals = " ".join(f"0x{int(v):08X}" for v in w[i:i + self.CHUNK])
            try:
                self.tcl(f"mwr -force 0x{self.base + addr + 4 * i:X} {{{vals}}}")
            except XsdbError as e:
                raise SlvErr(str(e)) from None

    def read_block(self, addr: int, n: int) -> np.ndarray:
        out = []
        for i in range(0, n, self.CHUNK):
            k = min(self.CHUNK, n - i)
            try:
                text = self.tcl(f"mrd -force 0x{self.base + addr + 4 * i:X} {k}")
            except XsdbError as e:
                raise SlvErr(str(e)) from None
            vals = [int(m.group(2), 16)
                    for m in re.finditer(r"([0-9A-Fa-f]+):\s+([0-9A-Fa-f]{1,8})\b", text)]
            if len(vals) != k:
                raise XsdbError(f"mrd returned {len(vals)} words, expected {k}: {text!r}")
            out += vals
        return np.array(out, dtype=np.uint32)

    def read32(self, addr: int) -> int:
        return int(self.read_block(addr, 1)[0])

    def write32(self, addr: int, value: int) -> None:
        self.write_block(addr, [value & 0xFFFFFFFF])

    def close(self) -> None:
        try:
            self.proc.stdin.write("exit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
class Device:
    def __init__(self, backend):
        self.be = backend

    # -- registers -------------------------------------------------------------
    def check_id(self) -> None:
        v = self.be.read32(REG_ID)
        if v != ID_VALUE:
            raise RuntimeError(f"ID register reads {v:#010x}, expected {ID_VALUE:#010x}")

    def cfg(self) -> dict:
        v = self.be.read32(REG_CFG)
        return dict(gbuf_bytes=(v & 0xFFFF) * 1024, imem_depth=v >> 16)

    def status(self) -> dict:
        s = self.be.read32(REG_STATUS)
        return dict(busy=bool(s & ST_BUSY), done=bool(s & ST_DONE), error=bool(s & ST_ERROR),
                    err_code=(s >> 8) & 0xFF, pc=self.be.read32(REG_PC),
                    cycles=self.be.read32(REG_CYCLES), instret=self.be.read32(REG_INSTRET),
                    passes=self.be.read32(REG_PASSES))

    def clear(self) -> None:
        self.be.write32(REG_CTRL, CTRL_CLEAR)

    # -- bulk memory -------------------------------------------------------------
    def _wblock(self, addr: int, words) -> None:
        if hasattr(self.be, "write_block"):
            self.be.write_block(addr, words)
        else:
            for i, v in enumerate(np.asarray(words, dtype=np.uint32)):
                self.be.write32(addr + 4 * i, int(v))

    def _rblock(self, addr: int, n: int) -> np.ndarray:
        if hasattr(self.be, "read_block"):
            return np.asarray(self.be.read_block(addr, n), dtype=np.uint32)
        return np.array([self.be.read32(addr + 4 * i) for i in range(n)], dtype=np.uint32)

    def write_gbuf(self, addr: int, data) -> None:
        """Write bytes at GBUF byte address ``addr`` (any alignment: edge words are RMW)."""
        data = bytes(data)
        if not data:
            return
        a0, a1 = addr & ~3, (addr + len(data) + 3) & ~3
        buf = bytearray(a1 - a0)
        if a0 != addr or a1 != addr + len(data):
            head = self.read_gbuf(a0, 4)
            tail = self.read_gbuf(a1 - 4, 4)
            buf[:4], buf[-4:] = head, tail
        buf[addr - a0:addr - a0 + len(data)] = data
        self._wblock(GBUF_BASE + a0, np.frombuffer(bytes(buf), dtype="<u4"))

    def read_gbuf(self, addr: int, n: int) -> bytes:
        a0, a1 = addr & ~3, (addr + n + 3) & ~3
        w = self._rblock(GBUF_BASE + a0, (a1 - a0) // 4)
        return w.astype("<u4").tobytes()[addr - a0:addr - a0 + n]

    # -- programs ------------------------------------------------------------------
    def load_program(self, instrs, pc: int = 0) -> None:
        words = isa.assemble(instrs)
        if pc + words.size // 8 > isa.IMEM_DEPTH:
            raise ValueError("program does not fit IMEM at this pc")
        self._wblock(IMEM_BASE + 32 * pc, words)
        self.be.write32(REG_PC_START, pc)

    def run(self, timeout: float = 10.0, poll: float = 0.0) -> dict:
        """START, wait for done/error, return the status + counters."""
        if self.status()["busy"]:
            raise RuntimeError("device busy")
        self.be.write32(REG_CTRL, CTRL_START)
        t_end = time.monotonic() + timeout
        while True:
            st = self.status()
            if not st["busy"] and (st["done"] or st["error"]):
                return st
            if time.monotonic() > t_end:
                raise TimeoutError(f"run did not finish: {st}")
            if poll:
                time.sleep(poll)

    def load_lut(self, lut, gaddr: int, pc: int = isa.IMEM_DEPTH - 2, timeout: float = 5.0) -> dict:
        """Place the 256-byte exp LUT at ``gaddr`` and execute LUTLOAD from a 2-slot stub at ``pc``."""
        lut = np.asarray(lut, dtype=np.int64)
        assert lut.shape == (256,) and lut.min() >= 0 and lut.max() <= 255
        self.write_gbuf(gaddr, lut.astype(np.uint8).tobytes())
        self.load_program([isa.lutload(gaddr), isa.end()], pc)
        st = self.run(timeout)
        if st["error"]:
            raise RuntimeError(f"LUTLOAD failed: {st}")
        return st

    def execute(self, build, timeout: float = 10.0):
        """Run a compiler.Build: image -> GBUF, each segment from pc 0, read the outputs.

        Returns (outputs dict, totals dict with cycles / instret / passes / runs).
        """
        self.write_gbuf(0, build.layout.image())
        total = dict(cycles=0, instret=0, passes=0, runs=0)
        for seg in build.segments():
            self.load_program(seg, 0)
            st = self.run(timeout)
            if st["error"] or not st["done"]:
                raise RuntimeError(f"program failed: {st}")
            for k in ("cycles", "instret", "passes"):
                total[k] += st[k]
            total["runs"] += 1
        outs = {n: build.layout.extract(n, self.read_gbuf) for n in build.layout.outputs}
        return outs, total
