"""
CIMple-FPGA instruction set: field encode/decode, assembler, disassembler.

Contract: fpga/docs/ARCH.md §4 (256-bit instructions = 8 little-endian 32-bit
words). Field positions here are normative and are checked bit by bit in
tests/test_isa.py; cpl_pkg.sv::decode() is the RTL side of the same table.

Paper: Ahn et al., "CIMple", arXiv:2604.15944v2. The paper has no ISA; the
whole instruction format is our assumption [A] built around the paper's
datapath (Fig. 4: 64b XIN, 32 partitions x 2 banks, 4-lane output, ACC
buffer, quantiser, LUT softmax).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, replace

import numpy as np

WORDS = 8                # 32-bit words per instruction
IMEM_DEPTH = 256


class Op(enum.IntEnum):
    END = 0
    WLOAD = 1
    MATVEC = 2
    LUTLOAD = 3
    NOP = 4


class Acc(enum.IntEnum):
    NONE = 0
    FIRST = 1
    ADD = 2


class Out(enum.IntEnum):
    NONE = 0
    Q8 = 1
    EXP = 2
    NORM = 3
    RAW32 = 4


class Err(enum.IntEnum):
    NONE = 0
    OPCODE = 1   # bad opcode / reserved encoding / fetch past instruction 255
    ALIGN = 2
    RANGE = 3    # GBUF byte address out of range
    PART = 4     # bad partition range or ncols > 32


# (field, word, lsb, width) -- ARCH.md §4. Word 0 is common to all opcodes.
W0_FIELDS = (
    ("opcode", 0, 0, 4),
    ("bank", 0, 4, 1),
    ("acc_mode", 0, 5, 2),
    ("out_mode", 0, 7, 3),
    ("act_unsigned", 0, 10, 1),
    ("sum_by_t", 0, 11, 1),
    ("sum_clr", 0, 12, 1),
    ("causal", 0, 13, 1),
    ("rsvd", 0, 14, 2),
    ("ntok", 0, 16, 16),
)
MATVEC_FIELDS = (
    ("aaddr", 1, 0, 32),
    ("astride", 2, 0, 32),
    ("oaddr", 3, 0, 32),
    ("tstride", 4, 0, 16),
    ("cstride", 4, 16, 16),
    ("scale", 5, 0, 16),
    ("shift", 5, 16, 5),
    ("ncols", 5, 24, 6),
    ("acc_base", 6, 0, 16),
    ("qbase", 7, 0, 16),
    ("kbase", 7, 16, 16),
)
WLOAD_FIELDS = (
    ("gaddr", 1, 0, 32),
    ("pstride", 2, 0, 32),
    ("p0", 3, 0, 5),
    ("np", 3, 8, 6),
)
LUTLOAD_FIELDS = (("gaddr", 1, 0, 32),)

OP_FIELDS = {
    Op.MATVEC: MATVEC_FIELDS,
    Op.WLOAD: WLOAD_FIELDS,
    Op.LUTLOAD: LUTLOAD_FIELDS,
}


@dataclass
class Instr:
    """One instruction; every field of ARCH.md §4.

    Opcode-specific fields that do not apply to ``opcode`` are ignored by
    ``encode`` (and are 0 after ``decode``). Values are range-checked against
    the field widths only, so reserved encodings (acc_mode 3, out_mode 5..7,
    unknown opcodes) can be built on purpose to test the error paths.
    """

    opcode: int = Op.NOP
    bank: int = 0
    acc_mode: int = Acc.NONE
    out_mode: int = Out.NONE
    act_unsigned: int = 0
    sum_by_t: int = 0
    sum_clr: int = 0
    causal: int = 0
    rsvd: int = 0
    ntok: int = 0
    # MATVEC
    aaddr: int = 0
    astride: int = 0
    oaddr: int = 0
    tstride: int = 0
    cstride: int = 0
    scale: int = 0
    shift: int = 0
    ncols: int = 0
    acc_base: int = 0
    qbase: int = 0
    kbase: int = 0
    # WLOAD / LUTLOAD
    gaddr: int = 0
    pstride: int = 0
    p0: int = 0
    np: int = 0

    # -- encoding ------------------------------------------------------------
    def _layout(self):
        try:
            op = Op(self.opcode)
        except ValueError:
            op = None
        # unknown opcodes keep the MATVEC view of words 1..7 so that
        # decode(encode(x)) stays lossless for error-path tests
        return W0_FIELDS + OP_FIELDS.get(op, MATVEC_FIELDS if op is None else ())

    def encode(self) -> np.ndarray:
        """-> 8 x uint32 (w0..w7)."""
        w = [0] * WORDS
        for name, word, lsb, width in self._layout():
            v = int(getattr(self, name))
            if not 0 <= v < (1 << width):
                raise ValueError(f"{name}={v} does not fit {width} bits")
            w[word] |= v << lsb
        return np.array(w, dtype=np.uint32)

    @classmethod
    def decode(cls, words) -> "Instr":
        w = [int(x) & 0xFFFFFFFF for x in np.asarray(words, dtype=np.uint64).ravel()]
        if len(w) != WORDS:
            raise ValueError("an instruction is 8 words")
        ins = cls(opcode=w[0] & 0xF)
        for name, word, lsb, width in ins._layout():
            setattr(ins, name, (w[word] >> lsb) & ((1 << width) - 1))
        return ins

    # -- text ----------------------------------------------------------------
    def __str__(self) -> str:
        try:
            op = Op(self.opcode)
        except ValueError:
            return f"OP{self.opcode} " + " ".join(f"{x:#x}" for x in self.encode())
        if op in (Op.END, Op.NOP):
            return op.name
        if op == Op.LUTLOAD:
            return f"LUTLOAD gaddr={self.gaddr:#x}"
        if op == Op.WLOAD:
            return (f"WLOAD bank={self.bank} gaddr={self.gaddr:#x} pstride={self.pstride} "
                    f"p0={self.p0} np={self.np}")
        flags = [n for n in ("act_unsigned", "sum_by_t", "sum_clr", "causal") if getattr(self, n)]
        acc = Acc(self.acc_mode).name if self.acc_mode in Acc._value2member_map_ else str(self.acc_mode)
        out = Out(self.out_mode).name if self.out_mode in Out._value2member_map_ else str(self.out_mode)
        return (f"MATVEC bank={self.bank} acc={acc} out={out} ntok={self.ntok} "
                f"a={self.aaddr:#x}+{self.astride}t o={self.oaddr:#x}+{self.tstride}t+{self.cstride}p "
                f"ncols={self.ncols} scale={self.scale} shift={self.shift} acc_base={self.acc_base} "
                f"q/k={self.qbase}/{self.kbase}" + (" " + ",".join(flags) if flags else ""))


def encode(ins: Instr) -> np.ndarray:
    return ins.encode()


def decode(words) -> Instr:
    return Instr.decode(words)


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------
def end() -> Instr:
    return Instr(opcode=Op.END)


def nop() -> Instr:
    return Instr(opcode=Op.NOP)


def lutload(gaddr: int) -> Instr:
    return Instr(opcode=Op.LUTLOAD, gaddr=gaddr)


def wload(bank: int, gaddr: int, pstride: int, p0: int = 0, np: int = 32) -> Instr:
    return Instr(opcode=Op.WLOAD, bank=bank, gaddr=gaddr, pstride=pstride, p0=p0, np=np)


def matvec(*, bank: int = 0, ntok: int, aaddr: int, astride: int = 64,
           acc_mode: int = Acc.NONE, out_mode: int = Out.NONE, oaddr: int = 0,
           tstride: int = 0, cstride: int = 0, ncols: int = 32, scale: int = 0,
           shift: int = 0, acc_base: int = 0, act_unsigned: bool = False,
           sum_by_t: bool = False, sum_clr: bool = False, causal: bool = False,
           qbase: int = 0, kbase: int = 0) -> Instr:
    return Instr(opcode=Op.MATVEC, bank=bank, acc_mode=int(acc_mode), out_mode=int(out_mode),
                 act_unsigned=int(act_unsigned), sum_by_t=int(sum_by_t), sum_clr=int(sum_clr),
                 causal=int(causal), ntok=ntok, aaddr=aaddr, astride=astride, oaddr=oaddr,
                 tstride=tstride, cstride=cstride, scale=scale, shift=shift, ncols=ncols,
                 acc_base=acc_base, qbase=qbase, kbase=kbase)


# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------
def assemble(prog) -> np.ndarray:
    """List of Instr -> flat uint32 array (8 words per instruction), IMEM order."""
    prog = list(prog)
    if len(prog) > IMEM_DEPTH:
        raise ValueError(f"{len(prog)} instructions > IMEM depth {IMEM_DEPTH}")
    if not prog:
        return np.zeros(0, dtype=np.uint32)
    return np.concatenate([i.encode() for i in prog]).astype(np.uint32)


def assemble_bytes(prog) -> bytes:
    return assemble(prog).astype("<u4").tobytes()


def split_words(words) -> list[Instr]:
    w = np.asarray(words, dtype=np.uint64).ravel()
    if w.size % WORDS:
        raise ValueError("word count is not a multiple of 8")
    return [Instr.decode(w[i:i + WORDS]) for i in range(0, w.size, WORDS)]


def disassemble(words, base: int = 0) -> str:
    """uint32 words (or bytes, or a list of Instr) -> one line per instruction."""
    if isinstance(words, (bytes, bytearray)):
        words = np.frombuffer(bytes(words), dtype="<u4")
    if isinstance(words, list) and words and isinstance(words[0], Instr):
        prog = words
    else:
        prog = split_words(words)
    return "\n".join(f"{base + i:3d}: {ins}" for i, ins in enumerate(prog))


__all__ = ["Op", "Acc", "Out", "Err", "Instr", "encode", "decode", "end", "nop", "lutload",
           "wload", "matvec", "assemble", "assemble_bytes", "disassemble", "split_words",
           "replace", "WORDS", "IMEM_DEPTH", "W0_FIELDS", "MATVEC_FIELDS", "WLOAD_FIELDS",
           "LUTLOAD_FIELDS"]
