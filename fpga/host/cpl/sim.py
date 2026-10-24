"""
Bit-exact ISA simulator of the CIMple-FPGA accelerator (ARCH.md §3 and §5).

This is THE reference the RTL is tested against. It is written straight from
the normative text of fpga/docs/ARCH.md §5 (not from any RTL): state is the
GBUF bytes, 2 banks x 32 partitions x 64 rows of INT8 weights, the ACC buffer
(ACC_DEPTH x 4 lanes x 32b), the 32 exp-sums (24b), the exp LUT, IMEM and the
AXI4-Lite register file. Arithmetic is exact integer maths with the golden
model's semantics (cimple_rtl.model.cimple_model: quantise, mantissa8,
msb_pos, build_recip_lut).

Paper: Ahn et al., "CIMple", arXiv:2604.15944v2, Fig. 4 (datapath), Sec. IV-B
(split softmax: constant-max exp LUT, exp-sum, reciprocal LUT + quantiser).

Choices the contract leaves open, fixed here and reported to the RTL side:
  [A] Writes to read-only registers are dropped with OKAY; reads of the
      write-only CTRL return 0 with OKAY (only *unmapped* addresses SLVERR).
  [A] Address bits above bit 20 are ignored (the AXI slave sees 21 bits).
  [A] w0 bits [15:14] ("reserved, must be 0") are ignored, like the other
      unused fields; acc_mode / out_mode are checked only for MATVEC.
  [A] Error priority when several apply: 1 (encoding) > 2 (alignment) >
      4 (partition / ncols) > 3 (range).
  [A] Error 3 in MATVEC is detected before any state changes (the contract
      allows mid-instruction detection; state after error 3 is unspecified,
      so tests compare only STATUS / err_code / PC after it).
  [A] Instruction 255 that is not END executes and retires, then the fetch of
      256 raises error 1 with PC = 255.
  [A] CYCLES is an estimate from ARCH.md §6 (not part of the contract).
"""
from __future__ import annotations

import numpy as np

from .isa import Acc, Err, Instr, Op, Out

ID_VALUE = 0x43504C31
N_ROWS, N_PART, N_BANKS, N_LANES = 64, 32, 2, 4
N_SUMS, SUM_W, LUT_DEPTH = 32, 24, 256
IMEM_DEPTH = 256
ADDR_W = 21

REG_ID, REG_CTRL, REG_STATUS, REG_PC_START, REG_PC = 0x00, 0x04, 0x08, 0x0C, 0x10
REG_CYCLES, REG_INSTRET, REG_PASSES, REG_CFG = 0x14, 0x18, 0x1C, 0x20
IMEM_BASE = 0x010000
GBUF_BASE = 0x100000
CTRL_START, CTRL_CLEAR = 1, 2
ST_BUSY, ST_DONE, ST_ERROR = 1, 2, 4
OKAY, SLVERR = 0, 2


def _recip_lut() -> np.ndarray:
    # = gm.build_recip_lut() = round(2^15 / (256 + m)); no ties for m in 0..255
    m = np.arange(LUT_DEPTH, dtype=np.int64)
    return (2 * 32768 + (256 + m)) // (2 * (256 + m))


RECIP_LUT = _recip_lut()


class SlvErr(Exception):
    """An AXI4-Lite access answered with SLVERR."""


def s8(x):
    return ((np.asarray(x, dtype=np.int64) + 128) & 0xFF) - 128


def wrap32(x):
    return ((np.asarray(x, dtype=np.int64) + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)


def quantise(v, scale, shift, rnd_fix: bool = False):
    """Vectorised gm.quantise: sat8(rnd(v * scale >> shift)), exact int64.

    Golden-model rounding (default): +2^(s-1) if v*scale >= 0 else -2^(s-1),
    then an arithmetic shift (floor). ``rnd_fix``: true round-half-away
    (ARCH.md §9, RND_FIX=1). scale / shift may be arrays (NORM).
    """
    prod = np.asarray(v, dtype=np.int64) * np.asarray(scale, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.int64)
    half = np.where(shift > 0, np.left_shift(1, np.maximum(shift - 1, 0)), 0)
    if rnd_fix:
        r = np.sign(prod) * ((np.abs(prod) + half) >> shift)
    else:
        r = np.where(prod >= 0, prod + half, prod - half) >> shift
    return np.clip(r, -128, 127)


def msb_pos(s: int) -> int:
    return 0 if s == 0 else int(s).bit_length() - 1


def mantissa8(s: int) -> int:
    p = msb_pos(s)
    return (s >> (p - 8)) & 0xFF if p >= 8 else (s << (8 - p)) & 0xFF


class CplSim:
    """ISA + register-level model of cpl_top.

    ``autorun`` (default True): a START write runs the program to completion
    inside the write, so STATUS reads ``done`` immediately. With
    ``autorun=False`` START only sets ``busy``; call ``step()`` / ``finish()``
    (this is how the busy-time SLVERR rules are exercised).
    ``fast``: vectorise MATVEC over t when that cannot change the result
    (no activation/output alias, no repeated output byte, no ACC entry reuse);
    ``fast=False`` always runs the literal t-by-t, p-by-p loop of §5.
    """

    def __init__(self, gbuf_bytes: int = 512 * 1024, acc_depth: int = 1024,
                 rnd_fix: bool = False, autorun: bool = True, fast: bool = True):
        assert gbuf_bytes % 1024 == 0 and gbuf_bytes <= (1 << 20)
        self.gbuf_bytes = gbuf_bytes
        self.acc_depth = acc_depth
        self.rnd_fix = rnd_fix
        self.autorun = autorun
        self.fast = fast
        self.gbuf = np.zeros(gbuf_bytes, dtype=np.uint8)
        self.banks = np.zeros((N_BANKS, N_PART, N_ROWS), dtype=np.int64)
        self.acc = np.zeros((acc_depth, N_LANES), dtype=np.int64)   # signed 32b values
        self.sums = np.zeros(N_SUMS, dtype=np.int64)
        self.explut = np.zeros(LUT_DEPTH, dtype=np.int64)
        self.imem = np.zeros((IMEM_DEPTH, 8), dtype=np.uint32)
        self.pc_start = 0
        self.pc = 0
        self.busy = self.done = self.error = False
        self.err_code = 0
        self.cycles = self.instret = self.passes = 0

    # ------------------------------------------------------------------
    # AXI4-Lite slave (ARCH.md §3)
    # ------------------------------------------------------------------
    def _decode_addr(self, addr: int):
        addr &= (1 << ADDR_W) - 1
        if addr % 4:
            return None, addr
        if addr <= REG_CFG:
            return "reg", addr
        if IMEM_BASE <= addr < IMEM_BASE + 32 * IMEM_DEPTH:
            return "imem", addr - IMEM_BASE
        if GBUF_BASE <= addr < GBUF_BASE + self.gbuf_bytes:
            return "gbuf", addr - GBUF_BASE
        return None, addr

    def status_word(self) -> int:
        return int(self.busy) | int(self.done) << 1 | int(self.error) << 2 | (self.err_code & 0xFF) << 8

    def axi_read(self, addr: int) -> tuple[int, int]:
        """-> (rdata, rresp)."""
        kind, a = self._decode_addr(addr)
        if kind == "reg":
            return {
                REG_ID: ID_VALUE, REG_CTRL: 0, REG_STATUS: self.status_word(),
                REG_PC_START: self.pc_start, REG_PC: self.pc,
                REG_CYCLES: self.cycles & 0xFFFFFFFF, REG_INSTRET: self.instret & 0xFFFFFFFF,
                REG_PASSES: self.passes & 0xFFFFFFFF,
                REG_CFG: (self.gbuf_bytes // 1024) | IMEM_DEPTH << 16,
            }[a], OKAY
        if kind in ("imem", "gbuf"):
            if self.busy:
                return 0, SLVERR
            if kind == "imem":
                return int(self.imem[a // 32, (a % 32) // 4]), OKAY
            return int.from_bytes(self.gbuf[a:a + 4].tobytes(), "little"), OKAY
        return 0, SLVERR

    def axi_write(self, addr: int, data: int, strb: int = 0xF) -> int:
        """-> bresp."""
        kind, a = self._decode_addr(addr)
        data &= 0xFFFFFFFF
        if kind == "reg":
            if a == REG_CTRL and strb & 1:
                if data & CTRL_CLEAR:
                    self.done = self.error = False
                    self.err_code = 0
                if data & CTRL_START and not self.busy:
                    self.start()
            elif a == REG_PC_START and strb & 1:
                self.pc_start = data & 0xFF
            return OKAY                      # RO registers: dropped, OKAY [A]
        if kind in ("imem", "gbuf"):
            if self.busy:
                return SLVERR
            if kind == "imem":
                i, w = a // 32, (a % 32) // 4
                old = int(self.imem[i, w])
                self.imem[i, w] = _merge(old, data, strb)
            else:
                old = int.from_bytes(self.gbuf[a:a + 4].tobytes(), "little")
                self.gbuf[a:a + 4] = np.frombuffer(
                    _merge(old, data, strb).to_bytes(4, "little"), dtype=np.uint8)
            return OKAY
        return SLVERR

    def read32(self, addr: int) -> int:
        d, r = self.axi_read(addr)
        if r != OKAY:
            raise SlvErr(f"read {addr:#x}")
        return d

    def write32(self, addr: int, data: int, strb: int = 0xF) -> None:
        if self.axi_write(addr, data, strb) != OKAY:
            raise SlvErr(f"write {addr:#x}")

    # ------------------------------------------------------------------
    # Backdoors (tests / SimBackend bulk path; no busy check)
    # ------------------------------------------------------------------
    def gbuf_write(self, addr: int, data) -> None:
        b = np.frombuffer(bytes(data), dtype=np.uint8)
        self.gbuf[addr:addr + b.size] = b

    def gbuf_read(self, addr: int, n: int) -> bytes:
        return self.gbuf[addr:addr + n].tobytes()

    def load_program(self, prog, pc: int = 0) -> None:
        for i, ins in enumerate(prog):
            self.imem[pc + i] = ins.encode() if isinstance(ins, Instr) else ins

    # ------------------------------------------------------------------
    # Run control
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.busy:
            return
        self.done = self.error = False
        self.err_code = 0
        self.cycles = self.instret = self.passes = 0
        self.sums[:] = 0
        self.pc = self.pc_start
        self.busy = True
        if self.autorun:
            self.finish()

    def step(self) -> bool:
        """Execute one instruction; returns True while still busy."""
        if not self.busy:
            return False
        ins = Instr.decode(self.imem[self.pc])
        err, cyc = self._exec(ins)
        self.cycles += 2 + cyc              # [A] fetch + decode estimate
        if err:
            self.error, self.err_code, self.busy = True, int(err), False
            return False
        self.instret += 1
        if ins.opcode == Op.END:
            self.done, self.busy = True, False
            return False
        if self.pc == IMEM_DEPTH - 1:
            self.error, self.err_code, self.busy = True, int(Err.OPCODE), False
            return False
        self.pc += 1
        return True

    def finish(self) -> None:
        while self.step():
            pass

    def run(self, pc_start: int | None = None) -> dict:
        if pc_start is not None:
            self.pc_start = pc_start & 0xFF
        self.start()
        self.finish()
        return self.status()

    def status(self) -> dict:
        return dict(busy=self.busy, done=self.done, error=self.error, err_code=self.err_code,
                    pc=self.pc, cycles=self.cycles, instret=self.instret, passes=self.passes)

    # ------------------------------------------------------------------
    # Instruction semantics (ARCH.md §5)
    # ------------------------------------------------------------------
    def _exec(self, ins: Instr):
        op = ins.opcode
        if op == Op.END or op == Op.NOP:
            return 0, 0
        if op == Op.WLOAD:
            return self._wload(ins)
        if op == Op.LUTLOAD:
            return self._lutload(ins)
        if op == Op.MATVEC:
            return self._matvec(ins)
        return Err.OPCODE, 0

    def _wload(self, ins: Instr):
        if ins.gaddr % 16 or ins.pstride % 16:
            return Err.ALIGN, 0
        if ins.np < 1 or ins.p0 + ins.np > N_PART:
            return Err.PART, 0
        if ins.gaddr + (ins.np - 1) * ins.pstride + N_ROWS - 1 >= self.gbuf_bytes:
            return Err.RANGE, 0
        for i in range(ins.np):
            a = ins.gaddr + i * ins.pstride
            self.banks[ins.bank, ins.p0 + i] = s8(self.gbuf[a:a + N_ROWS])
        return 0, 4 * ins.np + 2

    def _lutload(self, ins: Instr):
        if ins.gaddr % 16:
            return Err.ALIGN, 0
        if ins.gaddr + LUT_DEPTH - 1 >= self.gbuf_bytes:
            return Err.RANGE, 0
        self.explut[:] = self.gbuf[ins.gaddr:ins.gaddr + LUT_DEPTH]
        return 0, LUT_DEPTH // 16 + 2

    def _matvec(self, ins: Instr):
        if ins.acc_mode not in (Acc.NONE, Acc.FIRST, Acc.ADD) or ins.out_mode > Out.RAW32:
            return Err.OPCODE, 0
        if ins.aaddr % 16 or ins.astride % 16:
            return Err.ALIGN, 0
        if ins.ncols > N_PART:
            return Err.PART, 0
        ntok, ncols, om = ins.ntok, ins.ncols, ins.out_mode
        width = 4 if om == Out.RAW32 else 1
        writes = om != Out.NONE and ncols > 0
        if ntok:
            amax = ins.aaddr + (ntok - 1) * ins.astride + N_ROWS - 1
            if amax >= self.gbuf_bytes:
                return Err.RANGE, 0
            if writes:
                omax = ins.oaddr + (ntok - 1) * ins.tstride + (ncols - 1) * ins.cstride + width - 1
                if omax >= self.gbuf_bytes:
                    return Err.RANGE, 0
        if ins.sum_clr:
            self.sums[:] = 0
        if ntok == 0:
            return 0, 1
        t = np.arange(ntok, dtype=np.int64)
        p = np.arange(ncols, dtype=np.int64)
        oaddrs = None
        if writes:
            oaddrs = ins.oaddr + t[:, None] * ins.tstride + p[None, :] * ins.cstride
        if self.fast and self._can_vectorise(ins, oaddrs, width):
            self._matvec_fast(ins, oaddrs)
        else:
            self._matvec_slow(ins)
        self.passes += ntok
        wcyc = ((ncols + 15) // 16 + 1) if ins.cstride == 1 else ncols * width
        return 0, ntok * (4 + 9 + 8 + (wcyc if writes else 0))

    def _can_vectorise(self, ins: Instr, oaddrs, width: int) -> bool:
        if ins.acc_mode != Acc.NONE and ins.ntok * 8 > self.acc_depth:
            return False                     # ACC entries reused within the instruction
        if oaddrs is None:
            return True
        lo = int(oaddrs.min())
        hi = int(oaddrs.max()) + width - 1
        alo = ins.aaddr
        ahi = ins.aaddr + (ins.ntok - 1) * ins.astride + N_ROWS - 1
        if not (hi < alo or ahi < lo):
            return False                     # outputs may feed later activations
        b = (oaddrs[..., None] + np.arange(width)).ravel()
        return np.unique(b).size == b.size  # no byte written twice

    # -- shared pieces -------------------------------------------------------
    def _acts(self, ins: Instr, t):
        rows = ins.aaddr + np.asarray(t, dtype=np.int64)[:, None] * ins.astride + np.arange(N_ROWS)
        raw = self.gbuf[rows].astype(np.int64)
        return raw if ins.act_unsigned else s8(raw)

    def _acc_update(self, ins: Instr, t, mac):
        """mac (len(t), 32) -> v (len(t), 32) per step 3 of §5."""
        if ins.acc_mode == Acc.NONE:
            return mac
        p = np.arange(N_PART)
        e = (ins.acc_base + 8 * np.asarray(t)[:, None] + (p % 8)[None, :]) % self.acc_depth
        lane = np.broadcast_to(p // 8, e.shape)
        if ins.acc_mode == Acc.FIRST:
            new = wrap32(mac)
        else:
            new = wrap32(self.acc[e, lane] + mac)
        self.acc[e, lane] = new
        return new

    def _out_values(self, ins: Instr, t, v):
        """Byte values (and EXP sums) for the ncols outputs of steps t."""
        t = np.asarray(t, dtype=np.int64)
        nc = ins.ncols
        vv = v[:, :nc]
        om = ins.out_mode
        if om == Out.Q8:
            return quantise(vv, ins.scale, ins.shift, self.rnd_fix) & 0xFF, None
        if om == Out.EXP:
            z = quantise(vv, ins.scale, ins.shift, self.rnd_fix)
            ev = self.explut[z & 0xFF]
            p = np.arange(nc)[None, :]
            tt = t[:, None]
            if ins.sum_by_t:
                q, k, idx = ins.qbase + tt + 0 * p, ins.kbase + p + 0 * tt, (tt % N_SUMS) + 0 * p
            else:
                q, k, idx = ins.qbase + p + 0 * tt, ins.kbase + tt + 0 * p, p + 0 * tt
            if ins.causal:
                ev = np.where(k > q, 0, ev)
            return ev, idx
        if om == Out.NORM:
            p = np.arange(nc)[None, :]
            idx = (t[:, None] % N_SUMS) + 0 * p if ins.sum_by_t else p + 0 * t[:, None]
            s = self.sums[idx]
            msb = np.vectorize(msb_pos, otypes=[np.int64])(s) if s.size else s
            man = np.vectorize(mantissa8, otypes=[np.int64])(s) if s.size else s
            return quantise(vv, RECIP_LUT[man], 7 + msb, self.rnd_fix) & 0xFF, None
        if om == Out.RAW32:
            u = vv & 0xFFFFFFFF
            return np.stack([(u >> (8 * k)) & 0xFF for k in range(4)], axis=-1), None
        raise AssertionError(om)

    # -- the two execution strategies ----------------------------------------
    def _matvec_fast(self, ins: Instr, oaddrs):
        t = np.arange(ins.ntok)
        mac = self._acts(ins, t) @ self.banks[ins.bank].T
        v = self._acc_update(ins, t, mac)
        if oaddrs is None:
            return
        vals, idx = self._out_values(ins, t, v)
        if ins.out_mode == Out.RAW32:
            self.gbuf[(oaddrs[..., None] + np.arange(4)).ravel()] = vals.ravel()
        else:
            self.gbuf[oaddrs.ravel()] = vals.ravel()
        if idx is not None:
            np.add.at(self.sums, idx.ravel(), vals.ravel())
            self.sums &= (1 << SUM_W) - 1

    def _matvec_slow(self, ins: Instr):
        W = self.banks[ins.bank]
        writes = ins.out_mode != Out.NONE and ins.ncols > 0
        for t in range(ins.ntok):
            tt = np.array([t])
            mac = self._acts(ins, tt) @ W.T                      # step 1 + 2
            v = self._acc_update(ins, tt, mac)                     # step 3
            if not writes:
                continue
            # the values of step t depend only on v, the LUT and (NORM) the sums,
            # none of which a byte write of this step can change
            vals, idx = self._out_values(ins, tt, v)
            for p in range(ins.ncols):                             # step 4, increasing p
                o = ins.oaddr + t * ins.tstride + p * ins.cstride
                if ins.out_mode == Out.RAW32:
                    self.gbuf[o:o + 4] = vals[0, p]
                else:
                    self.gbuf[o] = vals[0, p]
                if idx is not None:
                    i = int(idx[0, p])
                    self.sums[i] = (self.sums[i] + int(vals[0, p])) & ((1 << SUM_W) - 1)


def _merge(old: int, new: int, strb: int) -> int:
    m = 0
    for b in range(4):
        if strb >> b & 1:
            m |= 0xFF << (8 * b)
    return (old & ~m & 0xFFFFFFFF) | (new & m)
