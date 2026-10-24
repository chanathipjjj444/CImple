"""ISA encode/decode: field positions exactly as ARCH.md §4, round trips, assembler."""
import numpy as np
import pytest

from cpl import isa
from cpl.isa import Acc, Instr, Op, Out

# (field, word, mask) straight from the ARCH.md §4 tables
W0 = [("opcode", 0x0000000F), ("bank", 0x00000010), ("acc_mode", 0x00000060),
      ("out_mode", 0x00000380), ("act_unsigned", 0x00000400), ("sum_by_t", 0x00000800),
      ("sum_clr", 0x00001000), ("causal", 0x00002000), ("rsvd", 0x0000C000),
      ("ntok", 0xFFFF0000)]
MATVEC = [("aaddr", 1, 0xFFFFFFFF), ("astride", 2, 0xFFFFFFFF), ("oaddr", 3, 0xFFFFFFFF),
          ("tstride", 4, 0x0000FFFF), ("cstride", 4, 0xFFFF0000), ("scale", 5, 0x0000FFFF),
          ("shift", 5, 0x001F0000), ("ncols", 5, 0x3F000000), ("acc_base", 6, 0x0000FFFF),
          ("qbase", 7, 0x0000FFFF), ("kbase", 7, 0xFFFF0000)]
WLOAD = [("gaddr", 1, 0xFFFFFFFF), ("pstride", 2, 0xFFFFFFFF), ("p0", 3, 0x0000001F),
         ("np", 3, 0x00003F00)]


def _ones(mask):
    lsb = (mask & -mask).bit_length() - 1
    return mask >> lsb


@pytest.mark.parametrize("name,mask", W0)
def test_w0_fields(name, mask):
    base = dict(opcode=Op.MATVEC)
    if name != "opcode":
        ins = Instr(**base, **{name: _ones(mask)})
        w = ins.encode()
        assert int(w[0]) == (mask | Op.MATVEC)
    else:
        w = Instr(opcode=0xF).encode()
        assert int(w[0]) == 0xF
    assert np.all(w[1:] == 0)


@pytest.mark.parametrize("name,word,mask", MATVEC)
def test_matvec_fields(name, word, mask):
    w = Instr(opcode=Op.MATVEC, **{name: _ones(mask)}).encode()
    exp = np.zeros(8, np.uint32)
    exp[0] = Op.MATVEC
    exp[word] = mask
    assert np.array_equal(w, exp)


@pytest.mark.parametrize("name,word,mask", WLOAD)
def test_wload_fields(name, word, mask):
    w = Instr(opcode=Op.WLOAD, **{name: _ones(mask)}).encode()
    exp = np.zeros(8, np.uint32)
    exp[0] = Op.WLOAD
    exp[word] = mask
    assert np.array_equal(w, exp)


def test_lutload_end_nop():
    assert list(isa.lutload(0x1230).encode()) == [3, 0x1230, 0, 0, 0, 0, 0, 0]
    assert list(isa.end().encode()) == [0] * 8
    assert list(isa.nop().encode()) == [4] + [0] * 7
    # MATVEC-only fields are not encoded for other opcodes
    assert list(Instr(opcode=Op.WLOAD, scale=5, gaddr=16).encode()) == [1, 16, 0, 0, 0, 0, 0, 0]


def test_example_encoding():
    ins = isa.matvec(bank=1, ntok=300, aaddr=0x40, astride=64, acc_mode=Acc.ADD,
                     out_mode=Out.NORM, oaddr=0x1001, tstride=7, cstride=64, ncols=32,
                     scale=0xBEEF, shift=17, acc_base=513, act_unsigned=True, sum_by_t=True,
                     sum_clr=True, causal=True, qbase=3, kbase=65535)
    w = [int(x) for x in ins.encode()]
    assert w[0] == (2 | 1 << 4 | 2 << 5 | 3 << 7 | 1 << 10 | 1 << 11 | 1 << 12 | 1 << 13 | 300 << 16)
    assert w[1:] == [0x40, 64, 0x1001, 7 | 64 << 16, 0xBEEF | 17 << 16 | 32 << 24, 513, 3 | 65535 << 16]


def test_roundtrip_random():
    rng = np.random.default_rng(0)
    for _ in range(300):
        op = int(rng.choice([0, 1, 2, 3, 4, 7, 15]))
        kw = dict(opcode=op, bank=int(rng.integers(2)), acc_mode=int(rng.integers(4)),
                  out_mode=int(rng.integers(8)), act_unsigned=int(rng.integers(2)),
                  sum_by_t=int(rng.integers(2)), sum_clr=int(rng.integers(2)),
                  causal=int(rng.integers(2)), rsvd=int(rng.integers(4)),
                  ntok=int(rng.integers(1 << 16)))
        if op in (2, 7, 15):
            kw.update(aaddr=int(rng.integers(1 << 32)), astride=int(rng.integers(1 << 32)),
                      oaddr=int(rng.integers(1 << 32)), tstride=int(rng.integers(1 << 16)),
                      cstride=int(rng.integers(1 << 16)), scale=int(rng.integers(1 << 16)),
                      shift=int(rng.integers(32)), ncols=int(rng.integers(64)),
                      acc_base=int(rng.integers(1 << 16)), qbase=int(rng.integers(1 << 16)),
                      kbase=int(rng.integers(1 << 16)))
        elif op == 1:
            kw.update(gaddr=int(rng.integers(1 << 32)), pstride=int(rng.integers(1 << 32)),
                      p0=int(rng.integers(32)), np=int(rng.integers(64)))
        elif op == 3:
            kw.update(gaddr=int(rng.integers(1 << 32)))
        ins = Instr(**kw)
        w = ins.encode()
        assert isa.decode(w) == ins
        assert np.array_equal(isa.decode(w).encode(), w)


def test_field_overflow_rejected():
    with pytest.raises(ValueError):
        Instr(opcode=Op.MATVEC, ncols=64).encode()
    with pytest.raises(ValueError):
        Instr(opcode=Op.WLOAD, p0=32).encode()
    with pytest.raises(ValueError):
        Instr(opcode=Op.MATVEC, ntok=1 << 16).encode()


def test_assemble_disassemble():
    prog = [isa.lutload(0x100), isa.wload(1, 0x200, 64, 3, 5),
            isa.matvec(ntok=4, aaddr=0x300, out_mode=Out.EXP, sum_clr=True), isa.nop(), isa.end()]
    words = isa.assemble(prog)
    assert words.dtype == np.uint32 and words.shape == (40,)
    assert isa.split_words(words) == prog
    assert isa.assemble_bytes(prog) == words.astype("<u4").tobytes()
    text = isa.disassemble(words)
    assert len(text.splitlines()) == 5
    assert "LUTLOAD" in text and "WLOAD bank=1" in text and "EXP" in text and "END" in text
    assert isa.disassemble(isa.assemble_bytes(prog)) == text
    with pytest.raises(ValueError):
        isa.assemble([isa.nop()] * 257)
