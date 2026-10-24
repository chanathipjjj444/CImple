"""ISA simulator: every opcode / mode / error code of ARCH.md §5 and the AXI map of §3.

The MATVEC checks compare against ``literal()``, a scalar, line-by-line
transcription of §5 using the golden model's scalar functions (independent of
cpl.sim's vectorised code), on random instructions -- including aliasing
outputs, repeated output bytes and ACC entry reuse -- for both the fast and the
literal t-by-t path of the simulator.
"""
import numpy as np
import pytest

from cimple_rtl.model import cimple_model as gm
from cpl import isa
from cpl.isa import Acc, Err, Instr, Out
from cpl.sim import (GBUF_BASE, IMEM_BASE, OKAY, REG_CFG, REG_CTRL, REG_CYCLES, REG_ID,
                     REG_INSTRET, REG_PASSES, REG_PC, REG_PC_START, REG_STATUS, SLVERR, CplSim,
                     SlvErr)

RECIP = gm.build_recip_lut()


def s8(b):
    return b - 256 if b >= 128 else b


def wrap32(v):
    return ((v + (1 << 31)) % (1 << 32)) - (1 << 31)


def quantise(v, scale, shift, rnd_fix=False):
    if not rnd_fix:
        return gm.quantise(v, scale, shift)
    prod = v * scale
    half = (1 << (shift - 1)) if shift else 0
    r = (abs(prod) + half) >> shift
    return max(-128, min(127, r if prod >= 0 else -r))


class Lit:
    """Plain-Python state for the literal model."""

    def __init__(self, sim: CplSim):
        self.G = bytearray(sim.gbuf.tobytes())
        self.W = [[[int(v) for v in sim.banks[b, p]] for p in range(32)] for b in range(2)]
        self.A = [[int(v) for v in row] for row in sim.acc]
        self.S = [0] * 32                         # START clears the exp-sums
        self.LUT = [int(v) for v in sim.explut]
        self.depth = sim.acc_depth

    def same(self, sim: CplSim):
        assert bytes(self.G) == sim.gbuf.tobytes()
        assert self.A == [[int(v) for v in row] for row in sim.acc]
        assert self.S == [int(v) for v in sim.sums]


def literal(st: Lit, ins: Instr, rnd_fix=False):
    """ARCH.md §5 MATVEC, one statement per sentence (no error handling)."""
    G = st.G
    if ins.sum_clr:
        st.S = [0] * 32
    for t in range(ins.ntok):
        base = ins.aaddr + t * ins.astride
        x = [G[base + r] if ins.act_unsigned else s8(G[base + r]) for r in range(64)]
        W = st.W[ins.bank]
        if ins.act_unsigned:
            acc = [sum(W[p][r] * x[r] for r in range(64)) for p in range(32)]
        else:   # through the golden bit-serial MAC
            acc = [int(a) for a in gm.core_mac(np.array(W), np.array(x))]
        v = [0] * 32
        for p in range(32):
            e = (ins.acc_base + 8 * t + p % 8) % st.depth
            ln = p // 8
            if ins.acc_mode == Acc.NONE:
                v[p] = acc[p]
            elif ins.acc_mode == Acc.FIRST:
                st.A[e][ln] = acc[p]
                v[p] = st.A[e][ln]
            else:
                st.A[e][ln] = wrap32(st.A[e][ln] + acc[p])
                v[p] = st.A[e][ln]
        for p in range(ins.ncols):
            o = ins.oaddr + t * ins.tstride + p * ins.cstride
            if ins.out_mode == Out.Q8:
                G[o] = quantise(v[p], ins.scale, ins.shift, rnd_fix) & 0xFF
            elif ins.out_mode == Out.EXP:
                z = quantise(v[p], ins.scale, ins.shift, rnd_fix)
                q, k = (ins.qbase + t, ins.kbase + p) if ins.sum_by_t else (ins.qbase + p, ins.kbase + t)
                ev = 0 if (ins.causal and k > q) else st.LUT[z & 0xFF]
                G[o] = ev
                i = t % 32 if ins.sum_by_t else p
                st.S[i] = (st.S[i] + ev) % (1 << 24)
            elif ins.out_mode == Out.NORM:
                i = t % 32 if ins.sum_by_t else p
                s = st.S[i]
                msb = gm.msb_pos(s)
                G[o] = quantise(v[p], int(RECIP[gm.mantissa8(s)]), 7 + msb, rnd_fix) & 0xFF
            elif ins.out_mode == Out.RAW32:
                G[o:o + 4] = (v[p] & 0xFFFFFFFF).to_bytes(4, "little")


def new_sim(rng, gbuf=8192, depth=64, **kw):
    sim = CplSim(gbuf_bytes=gbuf, acc_depth=depth, **kw)
    sim.gbuf[:] = rng.integers(0, 256, gbuf)
    sim.banks[:] = rng.integers(-128, 128, sim.banks.shape)
    sim.acc[:] = rng.integers(-(1 << 31), 1 << 31, sim.acc.shape)
    sim.sums[:] = rng.integers(0, 1 << 24, 32)
    sim.explut[:] = rng.integers(0, 256, 256)
    return sim


def run1(sim, ins):
    sim.imem[:] = 0
    sim.load_program([ins, isa.end()])
    return sim.run(0)


def rand_matvec(rng, gbuf=8192, alias=False):
    om = int(rng.integers(5))
    ntok = int(rng.integers(0, 12))
    ncols = int(rng.integers(0, 33))
    am = int(rng.integers(3))
    aaddr = int(rng.integers(0, 40)) * 16
    astride = int(rng.choice([0, 16, 64, 80]))
    if alias:   # outputs inside the activation window
        oaddr = aaddr + int(rng.integers(0, 64))
        tstride, cstride = int(rng.choice([0, 1, 64, 3])), int(rng.choice([0, 1, 2, 64]))
    else:
        oaddr = 4096 + int(rng.integers(0, 64))
        tstride, cstride = int(rng.choice([0, 1, 32, 48])), int(rng.choice([0, 1, 4, 48]))
    return isa.matvec(
        bank=int(rng.integers(2)), ntok=ntok, aaddr=aaddr, astride=astride, acc_mode=am,
        out_mode=om, oaddr=oaddr, tstride=tstride, cstride=cstride, ncols=ncols,
        scale=int(rng.integers(0, 1 << 16)), shift=int(rng.integers(0, 32)),
        acc_base=int(rng.integers(0, 1 << 16)), act_unsigned=bool(rng.integers(2)),
        sum_by_t=bool(rng.integers(2)), sum_clr=bool(rng.integers(2)),
        causal=bool(rng.integers(2)), qbase=int(rng.integers(0, 20)), kbase=int(rng.integers(0, 20)))


@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize("alias", [False, True])
def test_matvec_random_vs_literal(fast, alias):
    rng = np.random.default_rng(10 + fast + 2 * alias)
    for _ in range(40):
        sim = new_sim(rng, fast=fast)
        prog = [rand_matvec(rng, alias=alias) for _ in range(3)]   # sums carry over
        lit = Lit(sim)
        sim.imem[:] = 0
        sim.load_program(prog + [isa.end()])
        st = sim.run(0)
        assert st["done"] and not st["error"], (st, prog)
        for ins in prog:
            literal(lit, ins)
        lit.same(sim)
        assert st["passes"] == sum(i.ntok for i in prog) and st["instret"] == 4


def test_rnd_fix_vs_literal():
    rng = np.random.default_rng(3)
    for _ in range(30):
        sim = new_sim(rng, rnd_fix=True)
        ins = rand_matvec(rng)
        lit = Lit(sim)
        run1(sim, ins)
        literal(lit, ins, rnd_fix=True)
        lit.same(sim)


def test_quantise_bias_and_rnd_fix():
    sim = CplSim(gbuf_bytes=4096)
    sim.banks[0] = 0
    sim.banks[0, :, 0] = 1                      # acc[p] = x[0]
    for x, sh, gold, fixed in ((-1, 2, -1, 0), (-2, 2, -1, -1), (-6, 2, -2, -2), (5, 1, 3, 3),
                               (-5, 1, -3, -3), (127, 0, 127, 127)):
        for fix, exp in ((False, gold), (True, fixed)):
            sim.rnd_fix = fix
            sim.gbuf[0] = x & 0xFF
            run1(sim, isa.matvec(ntok=1, aaddr=0, out_mode=Out.Q8, oaddr=1024, ncols=1,
                                 scale=1, shift=sh))
            assert s8(int(sim.gbuf[1024])) == exp, (x, sh, fix)
            assert gm.quantise(x, 1, sh) == gold


def test_unsigned_activation_bit_serial():
    """act_unsigned: MSB cycle adds -> plain UQ8 dot product (bit-serial check)."""
    rng = np.random.default_rng(4)
    sim = new_sim(rng)
    run1(sim, isa.matvec(ntok=1, aaddr=0, act_unsigned=True, out_mode=Out.RAW32, oaddr=4096,
                         ncols=32, cstride=4))
    x = sim.gbuf[:64].astype(np.int64)
    W = sim.banks[0]
    raw = np.frombuffer(sim.gbuf[4096:4096 + 128].tobytes(), "<i4")
    # bit-serial, MSB first, the MSB cycle adds (ARCH.md §2)
    msb, lsb = gm.nibble_split(W)
    acc = np.zeros(32, np.int64)
    for k in range(7, -1, -1):
        xb = (x >> k) & 1
        comb = (msb @ xb) * 16 + lsb @ xb
        acc = comb if k == 7 else 2 * acc + comb
    assert np.array_equal(raw, acc) and np.array_equal(acc, W @ x)


def test_wload_and_lutload():
    rng = np.random.default_rng(5)
    sim = new_sim(rng)
    before = sim.banks.copy()
    st = run1(sim, isa.wload(1, 0x100, 80, p0=3, np=5))
    assert st["done"]
    for i in range(5):
        a = 0x100 + 80 * i
        assert np.array_equal(sim.banks[1, 3 + i], gm.s8(sim.gbuf[a:a + 64]))
    mask = np.ones((2, 32), bool)
    mask[1, 3:8] = False
    assert np.array_equal(sim.banks[mask], before[mask])
    run1(sim, isa.lutload(0x200))
    assert np.array_equal(sim.explut, sim.gbuf[0x200:0x300])
    # full array, np = 32
    run1(sim, isa.wload(0, 0, 64, 0, 32))
    assert np.array_equal(sim.banks[0], gm.s8(sim.gbuf[:2048]).reshape(32, 64))


def _exp_setup():
    sim = CplSim(gbuf_bytes=8192, acc_depth=64)
    sim.banks[0] = 0
    sim.banks[0, :, 0] = 1                       # acc[p] = x_t[0]
    sim.explut[:] = np.arange(256) % 200 + 1     # never 0
    return sim


def test_exp_causal_and_sum_orientation():
    sim = _exp_setup()
    for t in range(6):
        sim.gbuf[64 * t] = 10 + t                # z = x_t[0] with scale=1, shift=0
    # sum by column (encoder): (q, k) = (qbase + p, kbase + t), S[p]
    run1(sim, isa.matvec(ntok=6, aaddr=0, out_mode=Out.EXP, oaddr=4096, tstride=1, cstride=16,
                         ncols=4, scale=1, shift=0, causal=True, qbase=2, kbase=0, sum_clr=True))
    e = np.array([[sim.gbuf[4096 + 16 * p + t] for t in range(6)] for p in range(4)])
    lut = sim.explut
    exp = np.array([[0 if t > 2 + p else lut[10 + t] for t in range(6)] for p in range(4)])
    assert np.array_equal(e, exp)
    assert np.array_equal(sim.sums[:4], exp.sum(1)) and np.all(sim.sums[4:] == 0)
    # sum by t (decoder): (q, k) = (qbase + t, kbase + p), S[t mod 32]
    run1(sim, isa.matvec(ntok=6, aaddr=0, out_mode=Out.EXP, oaddr=4096, tstride=16, cstride=1,
                         ncols=4, scale=1, shift=0, causal=True, qbase=1, kbase=0, sum_clr=True,
                         sum_by_t=True))
    e = np.array([[sim.gbuf[4096 + 16 * t + p] for p in range(4)] for t in range(6)])
    exp = np.array([[0 if p > 1 + t else lut[10 + t] for p in range(4)] for t in range(6)])
    assert np.array_equal(e, exp)
    assert np.array_equal(sim.sums[:6], exp.sum(1))


def test_sum_wrap_24b_and_sum_by_t_mod32():
    sim = _exp_setup()
    sim.explut[:] = 255
    sim.sums[:] = (1 << 24) - 100
    # 40 passes, sum_by_t, 1 column: S[t mod 32] gets 255 once (t < 8: twice)
    st = run1(sim, isa.matvec(ntok=40, aaddr=0, astride=0, out_mode=Out.EXP, oaddr=4096,
                              tstride=1, ncols=1, scale=1, sum_by_t=True))
    assert st["done"]
    # START clears the sums, so they start from 0 here
    assert np.all(sim.sums[:8] == 510) and np.all(sim.sums[8:] == 255)
    # wrap: preset within the run via many passes onto one sum
    sim = CplSim(gbuf_bytes=128 * 1024)
    sim.explut[:] = 255
    n = (1 << 24) // 255 + 3                     # 65796 > 2^16 -> two instructions
    prog = [isa.matvec(ntok=60000, aaddr=0, astride=0, out_mode=Out.EXP, oaddr=4096, ncols=1,
                       tstride=1, scale=1, sum_by_t=False),
            isa.matvec(ntok=n - 60000, aaddr=0, astride=0, out_mode=Out.EXP, oaddr=4096,
                       tstride=1, ncols=1, scale=1, sum_by_t=False), isa.end()]
    sim.imem[:] = 0
    sim.load_program(prog)
    sim.run(0)
    assert sim.sums[0] == (n * 255) % (1 << 24)


def test_norm_matches_softmax_row():
    rng = np.random.default_rng(6)
    sim = CplSim(gbuf_bytes=8192)
    lut = gm.build_exp_lut(0.05)
    recip = gm.build_recip_lut()
    z = rng.integers(-128, 128, 64)
    vals = rng.integers(-128, 128, 64)
    row = gm.softmax_row(z, vals, lut, recip)
    # e via EXP: weights = identity row, activation = z, sum by column 0
    sim.gbuf[0x800:0x900] = lut
    sim.banks[0] = 0
    sim.banks[0, 0, :] = vals                    # partition 0 = V column
    sim.banks[1] = 0
    sim.gbuf[:64] = z & 0xFF
    for r in range(32):                          # bank 1 partition r picks x[r] / x[32 + r]
        sim.banks[1, r, r] = 1
    prog = [isa.lutload(0x800),
            # EXP of z[0..31] and z[32..63]: ntok=2 reading the two halves via 2 passes
            isa.matvec(bank=1, ntok=1, aaddr=0, out_mode=Out.EXP, oaddr=0x1000, cstride=1,
                       ncols=32, scale=1, sum_by_t=True, sum_clr=True),
            isa.matvec(bank=1, ntok=1, aaddr=0x40, out_mode=Out.EXP, oaddr=0x1020, cstride=1,
                       ncols=32, scale=1, sum_by_t=True),
            isa.matvec(bank=0, ntok=1, aaddr=0x1000, act_unsigned=True, out_mode=Out.NORM,
                       oaddr=0x1800, ncols=1, sum_by_t=True), isa.end()]
    # second half of z at 0x40 shifted down to rows 0..31
    sim.gbuf[0x40:0x60] = (z[32:] & 0xFF)
    sim.gbuf[0x60:0x80] = 0
    sim.imem[:] = 0
    sim.load_program(prog)
    st = sim.run(0)
    assert st["done"], st
    assert np.array_equal(sim.gbuf[0x1000:0x1040], row["e"])
    assert sim.sums[0] == row["sum"]
    assert s8(int(sim.gbuf[0x1800])) == row["score"]


def test_acc_wrap32_and_entry_wrap():
    sim = CplSim(gbuf_bytes=8192, acc_depth=16)
    sim.banks[0] = 0
    sim.banks[0, :, 0] = 127
    sim.gbuf[0] = 127                            # acc = 16129 per pass
    sim.acc[:] = (1 << 31) - 100
    # ntok=3 with depth 16: t=2 reuses entries of t=0 (16 mod 16), lane = p // 8
    run1(sim, isa.matvec(ntok=3, aaddr=0, astride=0, acc_mode=Acc.ADD, out_mode=Out.RAW32,
                         oaddr=4096, tstride=128, cstride=4, ncols=32, acc_base=0))
    raw = np.frombuffer(sim.gbuf[4096:4096 + 384].tobytes(), "<i4").reshape(3, 32)
    first = wrap32((1 << 31) - 100 + 16129)
    assert first < 0                             # wrapped
    assert np.all(raw[0] == first) and np.all(raw[1] == first)
    assert np.all(raw[2] == wrap32(first + 16129))
    assert sim.acc[0, 0] == wrap32(first + 16129) and sim.acc[8, 3] == first


def test_writes_visible_to_next_activation():
    """Output of step t feeds the activation read of step t+1 (no prefetch)."""
    for fast in (True, False):
        sim = CplSim(gbuf_bytes=8192, fast=fast)
        sim.banks[0] = 0
        sim.banks[0, 0, 0] = 1
        sim.banks[0, 0, 1] = 1                   # out = x[0] + x[1]
        sim.gbuf[0:2] = [1, 1]
        # write each result into x[1] of the next activation vector (astride 64)
        run1(sim, isa.matvec(ntok=6, aaddr=0, astride=64, out_mode=Out.Q8, oaddr=64 + 1,
                             tstride=64, ncols=1, scale=1, shift=0))
        got = [s8(int(sim.gbuf[64 * t + 1])) for t in range(1, 7)]
        assert got == [2, 2, 2, 2, 2, 2]
        sim = CplSim(gbuf_bytes=8192, fast=fast)
        sim.banks[0] = 0
        sim.banks[0, 0, 0] = 1
        sim.banks[0, 0, 1] = 1
        sim.gbuf[0:2] = [1, 1]
        sim.gbuf[64:64 * 8:64] = 1
        run1(sim, isa.matvec(ntok=6, aaddr=0, astride=64, out_mode=Out.Q8, oaddr=64 + 1,
                             tstride=64, ncols=1, scale=1, shift=0))
        got = [s8(int(sim.gbuf[64 * t + 1])) for t in range(1, 7)]
        assert got == [2, 3, 4, 5, 6, 7]         # x_{t+1}[1] = previous result


def test_last_write_wins():
    for fast in (True, False):
        sim = CplSim(gbuf_bytes=8192, fast=fast)
        sim.banks[0] = np.arange(32)[:, None] * (np.arange(64) == 0)
        sim.gbuf[0] = 1
        run1(sim, isa.matvec(ntok=2, aaddr=0, astride=0, out_mode=Out.Q8, oaddr=4096,
                             tstride=0, cstride=0, ncols=32, scale=1))
        assert sim.gbuf[4096] == 31


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
ERR_CASES = [
    (Instr(opcode=5), Err.OPCODE),
    (Instr(opcode=15), Err.OPCODE),
    (isa.matvec(ntok=1, aaddr=0, acc_mode=3), Err.OPCODE),
    (isa.matvec(ntok=1, aaddr=0, out_mode=5), Err.OPCODE),
    (isa.matvec(ntok=1, aaddr=0, out_mode=7), Err.OPCODE),
    (isa.matvec(ntok=1, aaddr=8), Err.ALIGN),
    (isa.matvec(ntok=1, aaddr=0, astride=24), Err.ALIGN),
    (isa.matvec(ntok=0, aaddr=4), Err.ALIGN),
    (isa.wload(0, 8, 64), Err.ALIGN),
    (isa.wload(0, 0, 72), Err.ALIGN),
    (isa.lutload(0x108), Err.ALIGN),
    (isa.matvec(ntok=1, aaddr=0, ncols=33), Err.PART),
    (isa.wload(0, 0, 64, p0=0, np=0), Err.PART),
    (isa.wload(0, 0, 64, p0=31, np=2), Err.PART),
    (isa.wload(0, 0, 64, p0=0, np=33), Err.PART),
    (isa.matvec(ntok=1, aaddr=8192 - 48), Err.RANGE),
    (isa.matvec(ntok=3, aaddr=8192 - 160, astride=64), Err.RANGE),
    (isa.matvec(ntok=2, aaddr=0, out_mode=Out.Q8, oaddr=8191, tstride=1, ncols=1), Err.RANGE),
    (isa.matvec(ntok=1, aaddr=0, out_mode=Out.RAW32, oaddr=8190, ncols=1), Err.RANGE),
    (isa.matvec(ntok=1, aaddr=0, out_mode=Out.EXP, oaddr=8000, cstride=7, ncols=32), Err.RANGE),
    (isa.wload(0, 8192 - 64, 64, np=2), Err.RANGE),
    (isa.lutload(8192 - 128), Err.RANGE),
    # error priority [A]: encoding > alignment > partition > range
    (isa.matvec(ntok=1, aaddr=8, acc_mode=3, ncols=40), Err.OPCODE),
    (isa.matvec(ntok=1, aaddr=8, ncols=40), Err.ALIGN),
    (isa.matvec(ntok=1, aaddr=1 << 20, ncols=40), Err.PART),
    (isa.wload(0, 8, 64, np=0), Err.ALIGN),
]


@pytest.mark.parametrize("ins,code", ERR_CASES)
def test_error_codes(ins, code):
    rng = np.random.default_rng(7)
    sim = new_sim(rng)
    snap = (sim.gbuf.copy(), sim.banks.copy(), sim.acc.copy(), sim.explut.copy())
    sim.imem[:] = 0
    sim.load_program([isa.nop(), isa.nop(), ins, isa.end()])
    st = sim.run(0)
    assert st["error"] and not st["done"] and not st["busy"]
    assert st["err_code"] == code and st["pc"] == 2 and st["instret"] == 2
    assert sim.read32(REG_STATUS) == (4 | code << 8)
    for a, b in zip(snap, (sim.gbuf, sim.banks, sim.acc, sim.explut)):
        assert np.array_equal(a, b)            # detected before any state change


def test_no_error_for_unused_out_of_range():
    sim = CplSim(gbuf_bytes=8192)
    # ntok = 0: nothing touched; ncols = 0 / out NONE: oaddr unused
    for ins in (isa.matvec(ntok=0, aaddr=1 << 30), isa.matvec(ntok=1, aaddr=0, oaddr=1 << 30,
                                                                out_mode=Out.Q8, ncols=0),
                isa.matvec(ntok=1, aaddr=0, oaddr=1 << 30, out_mode=Out.NONE, ncols=32)):
        assert run1(sim, ins)["done"]


def test_sum_clr_with_zero_tokens():
    sim = CplSim(gbuf_bytes=8192)
    sim.imem[:] = 0
    sim.explut[:] = 9
    sim.banks[:] = 0
    sim.load_program([isa.matvec(ntok=1, aaddr=0, out_mode=Out.EXP, oaddr=4096, ncols=3),
                      isa.matvec(ntok=0, aaddr=0, sum_clr=True), isa.end()])
    sim.run(0)
    assert np.all(sim.sums == 0)
    sim.load_program([isa.matvec(ntok=1, aaddr=0, out_mode=Out.EXP, oaddr=4096, ncols=3),
                      isa.end()])
    sim.run(0)
    assert list(sim.sums[:4]) == [9, 9, 9, 0]


def test_pc_overflow_and_pc_start():
    sim = CplSim(gbuf_bytes=8192)
    sim.imem[:] = isa.nop().encode()
    st = sim.run(250)
    assert st["error"] and st["err_code"] == Err.OPCODE and st["pc"] == 255 and st["instret"] == 6
    sim.imem[255] = isa.end().encode()
    st = sim.run(253)
    assert st["done"] and st["pc"] == 255 and st["instret"] == 3


# ---------------------------------------------------------------------------
# AXI4-Lite map
# ---------------------------------------------------------------------------
def test_regs_and_cfg():
    sim = CplSim()
    assert sim.read32(REG_ID) == 0x43504C31
    assert sim.read32(REG_CFG) == (512 | 256 << 16)
    assert CplSim(gbuf_bytes=64 * 1024).read32(REG_CFG) == (64 | 256 << 16)
    sim.write32(REG_PC_START, 0x1234)
    assert sim.read32(REG_PC_START) == 0x34
    assert sim.axi_write(REG_ID, 5) == OKAY and sim.read32(REG_ID) == 0x43504C31   # RO dropped [A]
    assert sim.axi_read(REG_CTRL) == (0, OKAY)                                       # WO reads 0 [A]


@pytest.mark.parametrize("addr", [0x24, 0x100, 0xFFFC, 0x012000, 0x0FFFFC, 0x100000 + 512 * 1024,
                                  0x1FFFFC])
def test_unmapped_slverr(addr):
    sim = CplSim()
    assert sim.axi_read(addr) == (0, SLVERR)
    assert sim.axi_write(addr, 1) == SLVERR
    with pytest.raises(SlvErr):
        sim.read32(addr)


@pytest.mark.parametrize("addr", [0x2, 0x5, 0x010001, 0x100003])
def test_unaligned_slverr(addr):
    sim = CplSim()
    assert sim.axi_read(addr)[1] == SLVERR and sim.axi_write(addr, 1) == SLVERR


def test_gbuf_imem_window_and_strobes():
    sim = CplSim(gbuf_bytes=64 * 1024)
    sim.write32(GBUF_BASE + 0x10, 0x11223344)
    assert list(sim.gbuf[0x10:0x14]) == [0x44, 0x33, 0x22, 0x11]
    assert sim.axi_write(GBUF_BASE + 0x10, 0xAABBCCDD, strb=0b0101) == OKAY
    assert sim.read32(GBUF_BASE + 0x10) == 0x11BB33DD
    sim.write32(GBUF_BASE + 64 * 1024 - 4, 7)
    assert sim.gbuf[-4] == 7
    sim.write32(IMEM_BASE + 32 * 3 + 4 * 5, 0xDEADBEEF)
    assert sim.imem[3, 5] == 0xDEADBEEF and sim.read32(IMEM_BASE + 32 * 3 + 20) == 0xDEADBEEF
    assert sim.axi_write(IMEM_BASE + 32 * 255 + 28, 0xFFFFFFFF, strb=0b1000) == OKAY
    assert sim.imem[255, 7] == 0xFF000000
    sim.write32(REG_PC_START, 0xAB, )
    assert sim.axi_write(REG_PC_START, 0xCD, strb=0b0010) == OKAY
    assert sim.read32(REG_PC_START) == 0xAB


def test_start_clear_busy_slverr():
    sim = CplSim(gbuf_bytes=8192, autorun=False)
    sim.load_program([isa.nop(), isa.matvec(ntok=3, aaddr=0), isa.end()], 0)
    sim.sums[:] = 5
    sim.write32(REG_CTRL, 1)
    assert sim.read32(REG_STATUS) == 1 and np.all(sim.sums == 0)
    # busy: IMEM / GBUF SLVERR (writes dropped, reads 0), registers fine
    assert sim.axi_read(GBUF_BASE) == (0, SLVERR)
    assert sim.axi_write(GBUF_BASE, 0x55) == SLVERR and sim.gbuf[0] == 0
    assert sim.axi_read(IMEM_BASE) == (0, SLVERR)
    assert sim.axi_write(IMEM_BASE, 0) == SLVERR and sim.imem[0, 0] == 4
    assert sim.read32(REG_PC) == 0
    sim.write32(REG_PC_START, 1)
    sim.write32(REG_CTRL, 1)                       # ignored while busy
    assert sim.read32(REG_PC) == 0
    sim.step()
    assert sim.read32(REG_PC) == 1 and sim.read32(REG_INSTRET) == 1
    sim.finish()
    assert sim.read32(REG_STATUS) == 2 and sim.read32(REG_PC) == 2
    assert sim.read32(REG_INSTRET) == 3 and sim.read32(REG_PASSES) == 3
    assert sim.read32(REG_CYCLES) > 0
    sim.write32(REG_CTRL, 2)                       # CLEAR
    assert sim.read32(REG_STATUS) == 0
    assert sim.read32(REG_INSTRET) == 3            # counters kept until the next START
    # error then CLEAR
    sim.imem[1] = Instr(opcode=9).encode()
    sim.write32(REG_CTRL, 1)
    sim.finish()
    assert sim.read32(REG_STATUS) == (4 | 1 << 8) and sim.read32(REG_PC) == 1
    sim.write32(REG_CTRL, 2)
    assert sim.read32(REG_STATUS) == 0
    # START clears done and counters; ACC / banks / LUT / GBUF persist
    sim.acc[3, 2] = 77
    sim.explut[9] = 4
    sim.imem[1] = isa.nop().encode()
    sim.write32(REG_CTRL, 1)
    assert sim.read32(REG_INSTRET) == 0 and sim.read32(REG_PASSES) == 0
    sim.finish()
    assert sim.acc[3, 2] == 77 and sim.explut[9] == 4
    assert sim.read32(REG_CYCLES) > 0 and sim.read32(REG_PC) == 2


def test_autorun_start():
    sim = CplSim(gbuf_bytes=8192)
    sim.load_program([isa.nop(), isa.end()], 10)
    sim.write32(REG_PC_START, 10)
    sim.write32(REG_CTRL, 1)
    assert sim.read32(REG_STATUS) == 2 and sim.read32(REG_PC) == 11
