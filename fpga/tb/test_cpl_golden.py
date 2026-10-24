"""cocotb end-to-end check of cpl_top against the USER's golden model.

    cd <repo> && unset VIRTUAL_ENV
    uv run make -C fpga/tb MODULE=test_cpl_golden TOPLEVEL=cpl_top PARAMS="-GGBUF_BYTES=65536"

Ground truth: src/cimple_rtl/model/cimple_model.py (``gm``), and nothing else.
Every expected value below is produced by gm primitives (core_mac / cim_matvec /
partition_mac, quantise, ratio_to_scale_shift, to_int8, build_exp_lut,
build_recip_lut, softmax_row, mantissa8, msb_pos, attention_head_cim). The ISA
simulator fpga/host/cpl/sim.py and the compiler's ``Build.ref`` are NOT used.

Transport only: the RTL is driven through its AXI4-Lite port by the host
driver (cpl.driver.Device over cpl.cocotb_backend.CocotbBackend over
axil_bfm.AxilMaster). Programs are hand-assembled with cpl.isa (tests 1-3) or
emitted by cpl.compiler's program builders (tests 2, 4, 5); the compiler is
used for its instruction stream and GBUF layout only.

Tests
  1. test_raw32_core_mac        RAW32 matvec == gm.core_mac, random + corner
                                weights / activations, both banks, partial WLOAD
  2. test_q8_projection         Q8(X@W) == gm.quantise(exact int dot), ties,
                                saturation, K>64 through the ACC buffer
  3. test_softmax_row_stream    EXP then NORM of one query == gm.softmax_row
                                (e, partial, score; M/shift identified from probes)
  4. test_head_vs_attention_head_cim
                                whole N<=32 encoder head == gm.attention_head_cim
                                (Q, K, V, e, out), scales captured from gm itself
  5. test_multitile_head        N=128, d=64 head == gm-primitive composition
"""
from __future__ import annotations

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

from axil_bfm import AxilMaster
from cimple_rtl.model import cimple_model as gm
from cpl import compiler as C
from cpl import isa
from cpl.cocotb_backend import CocotbBackend
from cpl.driver import Device
from cpl.isa import Acc, Out

CLK_NS = 5
RECIP = gm.build_recip_lut()


# ---------------------------------------------------------------------------
# transport helpers
# ---------------------------------------------------------------------------
async def start(dut):
    cocotb.start_soon(Clock(dut.clk_i, CLK_NS, unit="ns").start())
    dut.rst_ni.value = 0
    bfm = AxilMaster(dut, dut.clk_i)
    await ClockCycles(dut.clk_i, 5)
    dut.rst_ni.value = 1
    await ClockCycles(dut.clk_i, 2)
    be = CocotbBackend(bfm)
    dev = Device(be)
    await be.run(dev.check_id)
    G = (await be.run(dev.cfg))["gbuf_bytes"]
    return be, dev, G


def run_prog(dev: Device, prog):
    """Load + run one program (END appended); raise on error."""
    prog = list(prog) + [isa.end()]
    assert len(prog) <= isa.IMEM_DEPTH
    dev.load_program(prog, 0)
    st = dev.run(timeout=1e9)
    assert st["done"] and not st["error"], f"program failed: {st}\n{isa.disassemble(prog)}"
    return st


def i8bytes(a) -> bytes:
    return (np.asarray(a, np.int64) & 0xFF).astype(np.uint8).tobytes()


def read_i8(dev, addr, n):
    return gm.s8(np.frombuffer(dev.read_gbuf(addr, n), np.uint8))


def read_u8(dev, addr, n):
    return np.frombuffer(dev.read_gbuf(addr, n), np.uint8).astype(np.int64)


def read_i32(dev, addr, n):
    return np.frombuffer(dev.read_gbuf(addr, 4 * n), "<i4").astype(np.int64)


class Bump:
    def __init__(self, G):
        self.G, self.top = G, 0

    def __call__(self, nbytes, align=64):
        a = -(-self.top // align) * align
        self.top = a + nbytes
        assert self.top <= self.G, f"GBUF too small: need {self.top} > {self.G}"
        return a


def eq(name, got, exp, log):
    got, exp = np.asarray(got, np.int64), np.asarray(exp, np.int64)
    assert got.shape == exp.shape, f"{name}: shape {got.shape} != {exp.shape}"
    if not np.array_equal(got, exp):
        bad = np.argwhere(got != exp)
        i = tuple(bad[0])
        raise AssertionError(f"{name}: {len(bad)}/{got.size} mismatches vs golden model, "
                             f"first at {i}: RTL {got[i]} gm {exp[i]}")
    log.info("PASS %-48s %6d values bit-exact vs gm", name, got.size)


# gm.cim_matvec handles one 32-partition x 64-row tile; tile over partitions.
def gm_matvec(W, x):
    """W (P, r<=64) INT8, x (r,) INT8 -> (P,) via gm.cim_matvec on 32-partition tiles."""
    W = np.asarray(W, np.int64)
    return np.concatenate([gm.cim_matvec(W[p0:p0 + gm.N_PART], x)
                           for p0 in range(0, len(W), gm.N_PART)])


def gm_matvec_k(W, x):
    """W (P, R) with R > 64: sum of 64-row tiles (what the ACC buffer does)."""
    W = np.asarray(W, np.int64)
    return sum(gm_matvec(W[:, k0:k0 + gm.N_ROWS], np.asarray(x)[k0:k0 + gm.N_ROWS])
               for k0 in range(0, W.shape[1], gm.N_ROWS))


# ---------------------------------------------------------------------------
# 1. raw CIM matvec vs gm.core_mac
# ---------------------------------------------------------------------------
def corner_weights(rng):
    rows = [np.full(64, -128), np.full(64, 127), np.zeros(64), np.full(64, -1),
            np.tile([-128, 127], 32), np.arange(-32, 32), np.full(64, 15), np.full(64, -16),
            np.full(64, 0x70), np.full(64, 0x8F - 256),
            np.tile([-128, 0x0F, 0x7F, 0x8F - 256], 16), np.full(64, 1),
            np.tile([-128, -128, 127, 127], 16), np.where(np.arange(64) < 32, -128, 127),
            np.eye(64, dtype=np.int64)[0] * -128, np.eye(64, dtype=np.int64)[63] * 127]
    W = np.array(rows, np.int64)
    return np.concatenate([W, rng.integers(-128, 128, (32 - len(W), 64))])


def corner_acts(rng):
    rows = [np.full(64, -128), np.full(64, 127), np.zeros(64), np.full(64, -1), np.full(64, 1),
            np.tile([-128, 127], 32), np.arange(-32, 32), np.eye(64, dtype=np.int64)[0] * -128,
            np.eye(64, dtype=np.int64)[63] * -128, np.full(64, 64), np.full(64, -64),
            np.where(np.arange(64) < 32, 127, -128)]
    return np.concatenate([np.array(rows, np.int64), rng.integers(-128, 128, (12, 64))])


@cocotb.test()
async def test_raw32_core_mac(dut):
    be, dev, G = await start(dut)
    log = dut._log
    rng = np.random.default_rng(101)
    W0 = corner_weights(rng)                       # bank 0: corners + random
    W1 = rng.integers(-128, 128, (32, 64))         # bank 1: random
    Wp = rng.integers(-128, 128, (3, 64))          # partial reload of bank 1, partitions 5..7
    X = corner_acts(rng)
    T = len(X)
    a = Bump(G)
    w0a, w1a, wpa, xa = a(2048), a(2048), a(3 * 64), a(T * 64)
    o = [a(T * 128) for _ in range(4)]

    def job():
        dev.write_gbuf(w0a, i8bytes(W0))
        dev.write_gbuf(w1a, i8bytes(W1))
        dev.write_gbuf(wpa, i8bytes(Wp))
        dev.write_gbuf(xa, i8bytes(X))
        mv = lambda b, oa: isa.matvec(bank=b, ntok=T, aaddr=xa, astride=64, out_mode=Out.RAW32,  # noqa: E731
                                      oaddr=oa, tstride=128, cstride=4, ncols=32)
        run_prog(dev, [isa.wload(0, w0a, 64), isa.wload(1, w1a, 64),
                       mv(0, o[0]), mv(1, o[1]),
                       isa.wload(1, wpa, 64, p0=5, np=3),       # bank 1 partitions 5..7 only
                       mv(1, o[2]), mv(0, o[3])])
        return [read_i32(dev, oa, T * 32).reshape(T, 32) for oa in o]

    got = await be.run(job)
    W1p = W1.copy()
    W1p[5:8] = Wp
    for name, Wb, g in (("bank0 corners", W0, got[0]), ("bank1 random", W1, got[1]),
                        ("bank1 after partial WLOAD p5..7", W1p, got[2]),
                        ("bank0 untouched by bank1 WLOAD", W0, got[3])):
        exp = np.array([gm.core_mac(Wb, X[t]) for t in range(T)])
        eq(f"RAW32 vs gm.core_mac {name}", g, exp, log)
    exp = np.array([gm.core_mac(W0, X[t]) for t in range(T)])
    log.info("RAW32 extremes observed: min %d max %d (23b MAC range +-%d)",
             exp.min(), exp.max(), 1 << 22)


# ---------------------------------------------------------------------------
# 2. requantised projection X @ W (Q8) vs gm.quantise of exact integer dots
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_q8_projection(dut):
    be, dev, G = await start(dut)
    log = dut._log
    rng = np.random.default_rng(202)
    N, M = 40, 48
    for K in (64, 160):
        X = rng.integers(-128, 128, (N, K))
        W = rng.integers(-128, 128, (K, M))
        X[0], X[1], X[2] = -128, 127, 0
        W[:, 0], W[:, 1] = -128, 127
        acc = np.array([gm_matvec_k(W.T, X[i]) for i in range(N)])    # gm CIM, tiled
        assert np.array_equal(acc, X @ W)                               # == exact int dot
        amax = int(np.abs(acc).max())
        configs = [("calibrated absmax", gm.ratio_to_scale_shift(127 / amax)),
                   ("ties shift=1", (1, 1)), ("negative non-ties", (3, 3)),
                   ("saturating", (1, 0)), ("max scale/shift", (65535, 31)),
                   ("x4 overdrive", gm.ratio_to_scale_shift(4 * 127 / amax))]
        for what, (scale, shift) in configs:
            b = C.matmul(X, W, scale, shift, gbuf_bytes=G)
            outs, tot = await be.run(dev.execute, b, timeout=1e9)
            exp = np.vectorize(gm.quantise, otypes=[np.int64])(acc, scale, shift)
            eq(f"Q8 K={K} {what} (scale={scale} shift={shift})", outs["Y"], exp, log)
            if what == "ties shift=1":
                log.info("      %d exact ties (odd acc) in this case", int(np.sum(acc % 2 != 0)))


# ---------------------------------------------------------------------------
# 3. split-softmax row stream: EXP then NORM over one query vs gm.softmax_row
# ---------------------------------------------------------------------------
PROBE_K = np.concatenate([np.arange(-128, -96), np.arange(96, 128)])   # constant V columns


def softmax_case(dev, G, *, Kmat, q, scale, shift, exp_lut, V):
    """K (Nk, 64) in CIM (partition = key), q streamed (one query, t = 0).

    EXP writes e (UQ8) for every key; then A'V with e as UQ8 activations
    (act_unsigned) over 64-key chunks, V^T tiles (partition = value column).
    Column tiles: random V (32 cols) three times -- RAW32 unsigned (partial),
    NORM (score), RAW32 signed (to show what INT8 e would give) -- then the
    constant probe columns (NORM).
    """
    Nk = len(Kmat)
    pitch = -(-Nk // 64) * 64
    nch = pitch // 64
    d = V.shape[1]
    assert d == 32
    a = Bump(G)
    ka, qa, la, ea = a(Nk * 64), a(64), a(256), a(pitch)
    vta = a(32 * pitch)
    pra = [a(32 * 64) for _ in range(len(PROBE_K) // 32)]
    osz = dict(partial=128, score=32, signed=128, probe=len(PROBE_K))
    oa = {k: a(n) for k, n in osz.items()}

    dev.write_gbuf(ka, i8bytes(Kmat))
    dev.write_gbuf(qa, i8bytes(q))
    dev.write_gbuf(la, bytes(np.asarray(exp_lut, np.uint8)))
    dev.write_gbuf(ea, bytes(pitch))
    VT = np.zeros((32, pitch), np.int64)
    VT[:, :Nk] = V.T
    dev.write_gbuf(vta, i8bytes(VT))
    for j, pa in enumerate(pra):          # one 64-key chunk of constant rows, reused per chunk
        dev.write_gbuf(pa, i8bytes(np.repeat(PROBE_K[32 * j:32 * j + 32, None], 64, axis=1)))
    for k, v in oa.items():
        dev.write_gbuf(v, bytes(osz[k]))

    prog = [isa.lutload(la)]
    for k0 in range(0, Nk, 32):
        pk = min(32, Nk - k0)
        prog += [isa.wload(0, ka + 64 * k0, 64, 0, pk),
                 isa.matvec(bank=0, ntok=1, aaddr=qa, out_mode=Out.EXP, oaddr=ea + k0, cstride=1,
                            ncols=pk, scale=scale, shift=shift, sum_clr=(k0 == 0),
                            sum_by_t=True, qbase=0, kbase=k0)]

    def av(wsrc, wpitch, out_mode, oaddr, cstride, unsigned):
        p = []
        for c in range(nch):
            last = c == nch - 1
            p += [isa.wload(1, wsrc(c), wpitch, 0, 32),
                  isa.matvec(bank=1, ntok=1, aaddr=ea + 64 * c, act_unsigned=unsigned,
                             acc_mode=Acc.NONE if nch == 1 else (Acc.FIRST if c == 0 else Acc.ADD),
                             acc_base=0, out_mode=out_mode if last else Out.NONE, sum_by_t=True,
                             oaddr=oaddr if last else 0, cstride=cstride, ncols=32)]
        return p

    prog += av(lambda c: vta + 64 * c, pitch, Out.RAW32, oa["partial"], 4, True)
    prog += av(lambda c: vta + 64 * c, pitch, Out.NORM, oa["score"], 1, True)
    prog += av(lambda c: vta + 64 * c, pitch, Out.RAW32, oa["signed"], 4, False)
    for j, pa in enumerate(pra):
        prog += av(lambda c, pa=pa: pa, 64, Out.NORM, oa["probe"] + 32 * j, 1, True)
    run_prog(dev, prog)
    return dict(e=read_u8(dev, ea, Nk), partial=read_i32(dev, oa["partial"], 32),
                score=read_i8(dev, oa["score"], 32), signed=read_i32(dev, oa["signed"], 32),
                probe=read_i8(dev, oa["probe"], len(PROBE_K)))


def identify_M_shift(probe_out, s, shift_gm):
    """All (M', shift') with quantise(k*s, M', shift') == RTL for every probe k."""
    cands = []
    for sh in range(max(0, shift_gm - 3), min(31, shift_gm + 3) + 1):
        for M in range(256):
            if all(gm.quantise(int(k) * s, M, sh) == int(o) for k, o in zip(PROBE_K, probe_out)):
                cands.append((M, sh))
    return cands


@cocotb.test()
async def test_softmax_row_stream(dut):
    be, dev, G = await start(dut)
    log = dut._log
    rng = np.random.default_rng(303)
    unit_q = np.eye(64, dtype=np.int64)[0]

    def direct(z):              # K_p[0] = z_p, q = e_0, scale 1 shift 0 -> score = z exactly
        Km = np.zeros((len(z), 64), np.int64)
        Km[:, 0] = z
        return Km, unit_q, 1, 0

    cases = []
    # (a) direct random scores incl. corners, 160 keys (3 chunks, last partial)
    z = rng.integers(-128, 128, 160)
    z[:6] = [-128, 127, 0, -1, 1, 126]
    cases.append(("a: direct scores Nk=160", *direct(z), 0.04))
    # (b) real q.K with a calibrated requant (gm.ratio_to_scale_shift), 96 keys
    Km = rng.integers(-128, 128, (96, 64))
    q = rng.integers(-128, 128, 64)
    amax = int(np.abs(gm_matvec(Km, q)).max())
    cases.append(("b: q.K requantised Nk=96", Km, q, *gm.ratio_to_scale_shift(127 / amax), 0.03))
    # (c) every score at z_quant_max: e = 255 everywhere, long row (sum 65280)
    cases.append(("c: all scores 127 Nk=256", *direct(np.full(256, 127)), 0.04))
    # (d) sparse: e in {0,1,2,3} only, sum < 256 (mantissa8 left-shift path)
    lut_d = gm.build_exp_lut(0.5)
    small = [int(np.flatnonzero(lut_d == v)[0]) for v in (1, 2, 3)]
    z = np.full(64, -128)
    z[[3, 17, 40, 41, 63]] = gm.s8([small[0], small[1], small[2], small[2], small[0]])
    cases.append(("d: sparse e, sum<256 Nk=64", *direct(z), 0.5))
    # (e) all e = 0: sum = 0 (msb_pos(0) = 0, mantissa8(0) = 0)
    cases.append(("e: all e=0, sum=0 Nk=64", *direct(np.full(64, -128)), 0.5))

    for what, Km, q, scale, shift, score_scale in cases:
        Nk = len(Km)
        V = rng.integers(-128, 128, (Nk, 32))
        V[:, 0], V[:, 1], V[:, 2] = 127, -128, np.where(np.arange(Nk) % 2, 127, -128)
        exp_lut = gm.build_exp_lut(score_scale)
        got = await be.run(softmax_case, dev, G, Kmat=Km, q=q, scale=scale, shift=shift,
                           exp_lut=exp_lut, V=V)
        # golden: scores from gm CIM + gm.quantise, then gm.softmax_row per column
        zg = np.array([gm.quantise(a, scale, shift) for a in gm_matvec(Km, q)])
        rows = [gm.softmax_row(zg, V[:, c], exp_lut, RECIP) for c in range(32)]
        prow = [gm.softmax_row(zg, np.full(Nk, k), exp_lut, RECIP) for k in PROBE_K]
        r0 = rows[0]
        log.info("%s: sum=%d M=%d shift=%d (msb=%d mantissa=%d)", what, r0["sum"], r0["M"],
                 r0["shift"], gm.msb_pos(r0["sum"]), gm.mantissa8(r0["sum"]))
        eq(f"[{what}] EXP e", got["e"], r0["e"], log)
        eq(f"[{what}] A'V partial (unsigned e)", got["partial"], [r["partial"] for r in rows], log)
        eq(f"[{what}] NORM score", got["score"], [r["score"] for r in rows], log)
        eq(f"[{what}] NORM probe columns", got["probe"], [r["score"] for r in prow], log)
        # signed-INT8 e (act_unsigned=0) equals gm.core_mac on s8(e) -- and is NOT gm's partial
        es = gm.s8(r0["e"])
        signed_gm = sum(gm.cim_matvec(V[k0:k0 + 64].T, es[k0:k0 + 64]) for k0 in range(0, Nk, 64))
        eq(f"[{what}] RAW32 signed-e == gm.core_mac(s8(e))", got["signed"], signed_gm, log)
        n_wrap = int(np.sum(r0["e"] > 127))
        log.info("      e>127 (wrap negative as INT8): %d of %d; signed partial differs from gm "
                 "partial in %d/32 columns", n_wrap, Nk,
                 int(np.sum(signed_gm != np.array([r["partial"] for r in rows]))))
        if r0["sum"] > 0:
            cands = identify_M_shift(got["probe"], r0["sum"], r0["shift"])
            assert (r0["M"], r0["shift"]) in cands, (cands, r0["M"], r0["shift"])
            log.info("      (M, shift) consistent with the RTL probe outputs: %s; gm (%d, %d)",
                     cands, r0["M"], r0["shift"])


# ---------------------------------------------------------------------------
# 4. whole N<=32 head vs gm.attention_head_cim
# ---------------------------------------------------------------------------
def capture_gm_head(X, Wq, Wk, Wv):
    """Run gm.attention_head_cim and record the INT8 tensors and the exact integer
    (scale, shift) pairs it chose (by wrapping gm.to_int8 / gm.ratio_to_scale_shift
    for the duration of the call -- the model file is not modified)."""
    rec_q, rec_rq = [], []
    orig_q, orig_rq = gm.to_int8, gm.ratio_to_scale_shift

    def to_int8(x, scale):
        r = orig_q(x, scale)
        rec_q.append(r)
        return r

    def rq(r, *a, **k):
        v = orig_rq(r, *a, **k)
        rec_rq.append(v)
        return v

    gm.to_int8, gm.ratio_to_scale_shift = to_int8, rq
    try:
        res = gm.attention_head_cim(X, Wq, Wk, Wv)
    finally:
        gm.to_int8, gm.ratio_to_scale_shift = orig_q, orig_rq
    assert len(rec_q) == 4 and len(rec_rq) == 4, (len(rec_q), len(rec_rq))
    qX, qWq, qWk, qWv = rec_q
    scales = C.AttnScales(q=rec_rq[0], k=rec_rq[1], v=rec_rq[2], score=rec_rq[3],
                          exp_lut=gm.build_exp_lut(res["score_scale"]), out_scale=res["out_scale"])
    return res, (qX, qWq, qWk, qWv), scales


@cocotb.test()
async def test_head_vs_attention_head_cim(dut):
    be, dev, G = await start(dut)
    log = dut._log
    for seed, (N, dm, d) in ((404, (32, 64, 32)), (405, (20, 64, 24)), (406, (32, 64, 16))):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((N, dm))
        Ws = [rng.standard_normal((dm, d)) / np.sqrt(dm) for _ in range(3)]
        res, ints, sc = capture_gm_head(X, *Ws)
        b = C.encoder_attention_head(*ints, sc, gbuf_bytes=G)
        b.layout.outputs.append("E")
        outs, tot = await be.run(dev.execute, b, timeout=1e9)
        tag = f"head N={N} dm={dm} d={d}"
        log.info("%s: gm scales q=%s k=%s v=%s score=%s score_scale=%.5g; %d instr, %d passes, "
                 "%d cycles", tag, sc.q, sc.k, sc.v, sc.score, res["score_scale"], tot["instret"],
                 tot["passes"], tot["cycles"])
        eq(f"[{tag}] Q == gm Q", outs["Q"], res["Q"], log)
        eq(f"[{tag}] K == gm K", outs["K"], res["K"], log)
        eq(f"[{tag}] V^T == gm V^T", outs["VT"], res["V"].T, log)
        eq(f"[{tag}] e == gm e", outs["E"], res["e"], log)
        eq(f"[{tag}] out == gm.attention_head_cim out", outs["O"], res["out"], log)
        ref, _ = gm.attention_head_float(X, *Ws)
        cos = C.cosine(outs["O"] * res["out_scale"], ref)
        err = np.abs(outs["O"] * res["out_scale"] - ref)
        log.info("      vs gm.attention_head_float: cosine %.5f, max abs err %.4f "
                 "(= %.2f output LSB), mean abs err %.4f", cos, err.max(),
                 err.max() / res["out_scale"], err.mean())


# ---------------------------------------------------------------------------
# 5. multi-tile head N=128, d=64: reference from gm primitives only
# ---------------------------------------------------------------------------
def gm_head_multitile(X, Wq, Wk, Wv):
    """gm.attention_head_cim's recipe (per-tensor absmax calibration, same float
    expressions), with every CIM product tiled through gm.cim_matvec."""
    N, dm = X.shape
    d = Wq.shape[1]
    sx = np.abs(X).max() / 127
    sw = [np.abs(W).max() / 127 for W in (Wq, Wk, Wv)]
    qX = gm.to_int8(X, sx)
    qW = [gm.to_int8(W, s) for W, s in zip((Wq, Wk, Wv), sw)]
    proj, rqs = [], []
    for W, s in zip(qW, sw):
        acc = np.array([gm_matvec(W.T, qX[i]) for i in range(N)])      # d > 32: 2 tiles
        real = acc * sx * s
        sc = np.abs(real).max() / 127
        scl, sh = gm.ratio_to_scale_shift(sx * s / sc)
        proj.append((np.array([[gm.quantise(a, scl, sh) for a in row] for row in acc]), sc))
        rqs.append((scl, sh))
    (Qi, sq), (Ki, sk), (Vi, sv) = proj
    acc_s = np.array([gm_matvec(Ki, Qi[i]) for i in range(N)])         # N keys: 4 tiles
    real_s = acc_s * sq * sk / np.sqrt(d)
    score_scale = np.abs(real_s).max() / 127
    scl, sh = gm.ratio_to_scale_shift(sq * sk / np.sqrt(d) / score_scale)
    z = np.array([[gm.quantise(a, scl, sh) for a in row] for row in acc_s])
    exp_lut = gm.build_exp_lut(score_scale)
    out = np.zeros((N, d), np.int64)
    e = []
    for i in range(N):
        for c in range(d):
            row = gm.softmax_row(z[i], Vi[:, c], exp_lut, RECIP)
            out[i, c] = row["score"]
        e.append(row["e"])
    scales = C.AttnScales(q=rqs[0], k=rqs[1], v=rqs[2], score=(scl, sh), exp_lut=exp_lut,
                          out_scale=sv)
    return dict(out=out, Q=Qi, K=Ki, V=Vi, z=z, e=np.array(e), score_scale=score_scale,
                out_scale=sv), (qX, *qW), scales


@cocotb.test()
async def test_multitile_head(dut):
    be, dev, G = await start(dut)
    log = dut._log
    rng = np.random.default_rng(505)
    N, dm, d = 128, 64, 64
    X = rng.standard_normal((N, dm))
    Ws = [rng.standard_normal((dm, d)) / np.sqrt(dm) for _ in range(3)]
    res, ints, sc = gm_head_multitile(X, *Ws)
    b = C.encoder_attention_head(*ints, sc, gbuf_bytes=G)
    b.layout.outputs.append("E")
    outs, tot = await be.run(dev.execute, b, timeout=1e9)
    tag = f"multitile head N={N} d={d}"
    log.info("%s: scales q=%s k=%s v=%s score=%s; %d instr in %d run(s), %d passes, %d cycles",
             tag, sc.q, sc.k, sc.v, sc.score, tot["instret"], tot["runs"], tot["passes"],
             tot["cycles"])
    eq(f"[{tag}] Q", outs["Q"], res["Q"], log)
    eq(f"[{tag}] K", outs["K"], res["K"], log)
    eq(f"[{tag}] V^T", outs["VT"], res["V"].T, log)
    eq(f"[{tag}] e (last query tile 96..127)", outs["E"], res["e"][96:], log)
    eq(f"[{tag}] out", outs["O"], res["out"], log)
    ref, _ = gm.attention_head_float(X, *Ws)
    got = outs["O"] * res["out_scale"]
    log.info("      vs gm.attention_head_float: cosine %.5f, max abs err %.4f (%.2f LSB), "
             "mean abs err %.4f; max e-sum %d (< 2^24)", C.cosine(got, ref),
             np.abs(got - ref).max(), np.abs(got - ref).max() / res["out_scale"],
             np.abs(got - ref).mean(), int(res["e"].sum(axis=1).max()))
