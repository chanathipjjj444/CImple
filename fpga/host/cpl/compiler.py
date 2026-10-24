"""
Program builders for the CIMple-FPGA ISA: memory planner, tiled INT8 matmul
(weight projection) and the three attention mappings of the paper.

Paper: Ahn et al., "CIMple", arXiv:2604.15944v2, Sec. IV-C (encoder-only:
Q and V buffered in the CIM, K^T streamed as input), Sec. IV-D (decoder-only:
the CIM stores K^T and V, the input buffer holds q and QK^T; Eq. 3),
Sec. IV-E (encoder-decoder: decoder queries against the encoder's K and V).
ISA: fpga/docs/ARCH.md §4/§5.

Every builder returns a ``Build`` = (program, layout, ref): the instruction
list (no END; ``Build.segments()`` splits it into IMEM-sized programs), the
GBUF layout with the initial image, and a plain-numpy *integer* reference of
the same quantised maths (written with the golden model's scalar functions,
never through the simulator), so tests can assert ``device == ref`` exactly.

Data layouts in GBUF (all regions 16-byte aligned, zero padded):
  * activation matrices (X, K, Q, decoder q): row-major, pitch = round_up(cols, 64)
    so that every 64-byte activation read stays inside the row's zero padding.
  * weight sources for WLOAD (W^T, Q tile, K cache): row = partition, pitch
    round_up(cols, 64); a 64-row K tile is the 64 bytes at column k0.
  * V^T (d x keys), pitch round_up(keys, 64): a V^T tile of one 64-key chunk
    is (partition = head dim, rows = keys) -- [D] from Sec. IV-C "storing V ... in the CIM".
  * e buffer (queries x keys, UQ8), pitch round_up(keys, 64): one query's e of one
    64-key chunk is contiguous, so it is one A'V activation vector. The
    encoder QK^T pass yields one key for 32 queries (a column), so its EXP
    output is written transposed (tstride=1, cstride=pitch) -- the corner turn
    the paper does not describe [A] (sw/src/cimple_sw/mapping.py "E buffer").

Choices:
  [A] e is fed to A'V as UQ8 (``act_unsigned=1``), the denominator is the exp-sum
      of the same e: identical to cimple_sw.mapping with e_mode="unsigned".
  [A] Bank use: projections alternate banks; encoder: Q tile in bank 0, V^T
      tiles in bank 1; decoder: K tiles in bank 0, V^T tiles in bank 1.
  [A] Encoder QK^T runs over all keys of a query tile in one MATVEC before the
      A'V chunks (the sums are order independent); with ``causal`` keys past
      the tile's last 64-key chunk are skipped, masked e inside it are 0.
  [A] Matmul with K > 64 accumulates K tiles in the ACC buffer (FIRST, ADD..,
      last ADD + Q8) in token blocks of ACC_DEPTH/8 tokens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cimple_rtl.model import cimple_model as gm

from . import isa
from .isa import Acc, Instr, Out

N_ROWS, N_PART = 64, 32
LUT_BYTES = 256
MAX_PROG = 255          # IMEM entries per segment, + END


def round_up(x: int, m: int) -> int:
    return -(-int(x) // m) * m


# ---------------------------------------------------------------------------
# Memory planner
# ---------------------------------------------------------------------------
@dataclass
class Region:
    name: str
    addr: int
    rows: int
    cols: int
    pitch: int            # bytes between rows
    elem: int = 1         # 1: INT8/UQ8, 4: int32 (RAW32)
    signed: bool = True

    @property
    def nbytes(self) -> int:
        return self.rows * self.pitch

    def at(self, r: int, c: int) -> int:
        return self.addr + r * self.pitch + c * self.elem

    def pack(self, a) -> bytes:
        """numpy (rows, cols) -> the region's bytes (padding = 0)."""
        a = np.asarray(a, dtype=np.int64)
        assert a.shape == (self.rows, self.cols), (self.name, a.shape)
        buf = np.zeros((self.rows, self.pitch), dtype=np.uint8)
        if self.elem == 1:
            buf[:, :self.cols] = (a & 0xFF).astype(np.uint8)
        else:
            b = (a & 0xFFFFFFFF).astype("<u4").view(np.uint8).reshape(self.rows, -1)
            buf[:, :4 * self.cols] = b
        return buf.tobytes()

    def unpack(self, data: bytes) -> np.ndarray:
        """the region's bytes -> numpy (rows, cols)."""
        b = np.frombuffer(bytes(data), dtype=np.uint8)[:self.nbytes].reshape(self.rows, self.pitch)
        if self.elem == 1:
            v = b[:, :self.cols].astype(np.int64)
            return gm.s8(v) if self.signed else v
        w = np.ascontiguousarray(b[:, :4 * self.cols]).view("<u4").astype(np.int64)
        return ((w + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)


class Planner:
    """Bump allocator over GBUF, 16-byte (128b line) aligned by default."""

    def __init__(self, size: int = 512 * 1024, base: int = 0):
        self.size = size
        self.top = base
        self.regions: dict[str, Region] = {}

    def alloc(self, name: str, rows: int, cols: int, pitch: int | None = None,
              elem: int = 1, signed: bool = True, align: int = 16) -> Region:
        assert name not in self.regions, name
        pitch = round_up(cols * elem, 16) if pitch is None else pitch
        assert pitch % 16 == 0 and pitch >= cols * elem
        addr = round_up(self.top, align)
        r = Region(name, addr, rows, cols, pitch, elem, signed)
        if addr + r.nbytes > self.size:
            raise MemoryError(f"GBUF overflow allocating {name}: needs {addr + r.nbytes} B "
                              f"> {self.size} B")
        self.top = addr + r.nbytes
        self.regions[name] = r
        return r

    def act(self, name: str, rows: int, cols: int, **kw) -> Region:
        """A matrix whose rows are read as 64-byte activations / WLOAD rows."""
        return self.alloc(name, rows, cols, pitch=round_up(max(cols, 1), N_ROWS), **kw)


@dataclass
class Layout:
    gbuf_bytes: int
    regions: dict
    inputs: dict = field(default_factory=dict)       # name -> numpy array
    outputs: list = field(default_factory=list)      # names to read back
    top: int = 0

    def image(self) -> bytes:
        """GBUF bytes [0, top): every input placed, everything else zero."""
        img = bytearray(self.top)
        for name, a in self.inputs.items():
            r = self.regions[name]
            img[r.addr:r.addr + r.nbytes] = r.pack(a)
        return bytes(img)

    def extract(self, name: str, read) -> np.ndarray:
        """``read(addr, n) -> bytes`` (a device / sim reader)."""
        r = self.regions[name]
        return r.unpack(read(r.addr, r.nbytes))


@dataclass
class Build:
    program: list
    layout: Layout
    ref: dict
    splits: set = field(default_factory=set)

    def __iter__(self):
        yield self.program
        yield self.layout
        yield self.ref

    def segments(self, max_len: int = MAX_PROG) -> list[list[Instr]]:
        """Split at allowed points into programs of <= max_len + END.

        The 32 exp-sums are cleared by START, so a split never falls between
        an EXP and its NORM; ACC, banks, LUT and GBUF persist across runs.
        """
        cuts = sorted(s for s in self.splits | {len(self.program)} if 0 < s <= len(self.program))
        segs, start = [], 0
        last_ok = 0
        for c in cuts:
            if c - start > max_len:
                if last_ok == start:
                    raise ValueError("an unsplittable instruction run exceeds IMEM")
                segs.append(self.program[start:last_ok])
                start = last_ok
                if c - start > max_len:
                    raise ValueError("an unsplittable instruction run exceeds IMEM")
            last_ok = c
        segs.append(self.program[start:])
        return [s + [isa.end()] for s in segs if s] or [[isa.end()]]


class _Emitter:
    def __init__(self, gbuf_bytes: int, acc_depth: int):
        self.plan = Planner(gbuf_bytes)
        self.acc_depth = acc_depth
        self.prog: list[Instr] = []
        self.splits: set[int] = set()
        self.sums_live = False
        self._bank = 1
        self.layout = Layout(gbuf_bytes, self.plan.regions)

    def emit(self, ins: Instr):
        if not self.sums_live:
            self.splits.add(len(self.prog))
        self.prog.append(ins)

    def next_bank(self) -> int:
        self._bank ^= 1
        return self._bank

    def input(self, region: Region, a):
        self.layout.inputs[region.name] = np.asarray(a, dtype=np.int64)

    def build(self, ref: dict, outputs) -> Build:
        self.layout.top = self.plan.top
        self.layout.outputs = list(outputs)
        if not self.sums_live:
            self.splits.add(len(self.prog))
        return Build(self.prog, self.layout, ref, set(self.splits))

    # -- weight projection (Sec. IV-C, red flow) ------------------------------
    def matmul(self, src: Region, wt: Region, dst: Region, rq, *, n: int | None = None,
               row0: int = 0, col0: int = 0, transposed: bool = False, raw: bool = False):
        """dst[(row0+i), (col0+j)] = Q8((src @ wt.T)[i, j]) (or Y^T if ``transposed``).

        src: (N, K) activations; wt: (M, K) = W^T (row = output column).
        """
        N = src.rows if n is None else n
        K, M = src.cols, wt.rows
        assert wt.cols == K
        scale, shift = rq if not raw else (0, 0)
        out = Out.RAW32 if raw else Out.Q8
        ktiles = -(-K // N_ROWS)
        nblk = N if ktiles == 1 else self.acc_depth // 8

        def oparams(n0, m0):
            if transposed:
                return dict(oaddr=dst.at(col0 + m0, row0 + n0), tstride=dst.elem, cstride=dst.pitch)
            return dict(oaddr=dst.at(row0 + n0, col0 + m0), tstride=dst.pitch, cstride=dst.elem)

        for m0 in range(0, M, N_PART):
            pm = min(N_PART, M - m0)
            for n0 in range(0, N, max(nblk, 1)):
                nn = min(nblk, N - n0)
                for ki in range(ktiles):
                    k0 = ki * N_ROWS
                    b = self.next_bank()
                    self.emit(isa.wload(b, wt.at(m0, k0), wt.pitch, 0, pm))
                    last = ki == ktiles - 1
                    acc = Acc.NONE if ktiles == 1 else (Acc.FIRST if ki == 0 else Acc.ADD)
                    self.emit(isa.matvec(
                        bank=b, ntok=nn, aaddr=src.at(n0, k0), astride=src.pitch, acc_mode=acc,
                        out_mode=out if last else Out.NONE, ncols=pm, scale=scale, shift=shift,
                        acc_base=0, **(oparams(n0, m0) if last else dict(oaddr=0))))

    # -- split-softmax attention ---------------------------------------------
    def encoder_attn(self, Qr: Region, Kr: Region, VTr: Region, Er: Region, Or: Region,
                     N: int, Nk: int, d: int, rq_s, lut: Region, causal: bool):
        """Sec. IV-C: Q tile (32 queries) and V^T tiles in the CIM, K streamed."""
        scale, shift = rq_s
        bq, bv = 0, 1
        self.emit(isa.lutload(lut.addr))
        for q0 in range(0, N, N_PART):
            nq = min(N_PART, N - q0)
            nkeys = min(Nk, round_up(q0 + nq, N_ROWS)) if causal else Nk
            self.emit(isa.wload(bq, Qr.at(q0, 0), Qr.pitch, 0, nq))
            # one pass per key: column of scores for nq queries -> e^T written
            self.emit(isa.matvec(
                bank=bq, ntok=nkeys, aaddr=Kr.addr, astride=Kr.pitch, out_mode=Out.EXP,
                oaddr=Er.addr, tstride=1, cstride=Er.pitch, ncols=nq, scale=scale, shift=shift,
                sum_clr=True, sum_by_t=False, causal=causal, qbase=q0, kbase=0))
            self.sums_live = True
            self._av(Er, VTr, Or, ng=nq, out_row=q0, nkeys=nkeys, d=d, bank=bv)
            self.sums_live = False

    def decoder_attn(self, Qr: Region, Kr: Region, VTr: Region, Er: Region, Or: Region,
                     nq_total: int, L: int, d: int, rq_s, lut: Region, causal: bool, qpos0: int):
        """Sec. IV-D: K tiles (partition = key) and V^T tiles in the CIM, q streamed."""
        scale, shift = rq_s
        bk, bv = 0, 1
        self.emit(isa.lutload(lut.addr))
        for g0 in range(0, nq_total, N_PART):
            ng = min(N_PART, nq_total - g0)
            for k0 in range(0, L, N_PART):
                pk = min(N_PART, L - k0)
                self.emit(isa.wload(bk, Kr.at(k0, 0), Kr.pitch, 0, pk))
                self.emit(isa.matvec(
                    bank=bk, ntok=ng, aaddr=Qr.at(g0, 0), astride=Qr.pitch, out_mode=Out.EXP,
                    oaddr=Er.addr + k0, tstride=Er.pitch, cstride=1, ncols=pk, scale=scale,
                    shift=shift, sum_clr=(k0 == 0), sum_by_t=True, causal=causal,
                    qbase=qpos0 + g0, kbase=k0))
                self.sums_live = True
            self._av(Er, VTr, Or, ng=ng, out_row=g0, nkeys=L, d=d, bank=bv)
            self.sums_live = False

    def _av(self, Er, VTr, Or, *, ng, out_row, nkeys, d, bank):
        """A'V over 64-key chunks, e as UQ8 activations, NORM by t on the last chunk."""
        nch = -(-nkeys // N_ROWS)
        for d0 in range(0, d, N_PART):
            pd = min(N_PART, d - d0)
            abase = (d0 // N_PART) * 8 * N_PART
            assert abase + 8 * ng <= self.acc_depth
            for c in range(nch):
                self.emit(isa.wload(bank, VTr.at(d0, c * N_ROWS), VTr.pitch, 0, pd))
                last = c == nch - 1
                acc = Acc.NONE if nch == 1 else (Acc.FIRST if c == 0 else Acc.ADD)
                self.emit(isa.matvec(
                    bank=bank, ntok=ng, aaddr=Er.at(0, c * N_ROWS), astride=Er.pitch,
                    act_unsigned=True, acc_mode=acc, acc_base=abase,
                    out_mode=Out.NORM if last else Out.NONE, sum_by_t=True,
                    oaddr=Or.at(out_row, d0) if last else 0, tstride=Or.pitch, cstride=1,
                    ncols=pd))


# ---------------------------------------------------------------------------
# Plain integer reference (golden-model scalar functions only)
# ---------------------------------------------------------------------------
_gq = np.vectorize(gm.quantise, otypes=[np.int64])
_RECIP = gm.build_recip_lut()


def ref_quantise(acc, rq):
    return _gq(np.asarray(acc, dtype=np.int64), int(rq[0]), int(rq[1]))


def ref_matmul(X, W, rq):
    return ref_quantise(np.asarray(X, np.int64) @ np.asarray(W, np.int64), rq)


def ref_attention(Q, K, V, rq_s, exp_lut, causal: bool = False, qpos0: int = 0):
    """Split softmax with e as UQ8: z, e, den, partial, out (all integer)."""
    Q, K, V = (np.asarray(a, np.int64) for a in (Q, K, V))
    z = ref_quantise(Q @ K.T, rq_s)
    e = np.asarray(exp_lut, np.int64)[z & 0xFF]
    if causal:
        mask = np.arange(len(K))[None, :] > (np.arange(len(Q))[:, None] + qpos0)
        e = np.where(mask, 0, e)
    den = e.sum(axis=1)
    assert np.all(den < (1 << gm.SUM_W)), "exp-sum overflow (24b)"
    partial = e @ V
    out = np.zeros_like(partial)
    for i, s in enumerate(den):
        s = int(s)
        M, sh = int(_RECIP[gm.mantissa8(s)]), gm.RECIP_FRAC - gm.LUT_W + gm.msb_pos(s)
        out[i] = _gq(partial[i], M, sh)
    return dict(z=z, e=e, den=den, partial=partial, out=out)


# ---------------------------------------------------------------------------
# Scales / calibration
# ---------------------------------------------------------------------------
@dataclass
class AttnScales:
    """Requant (scale, shift) pairs of one head + its exp LUT (static scales)."""

    q: tuple
    k: tuple
    v: tuple
    score: tuple
    exp_lut: np.ndarray
    out_scale: float = 1.0        # real value of one output LSB (= V's scale)
    meta: dict = field(default_factory=dict)


@dataclass
class Calibration:
    xq: np.ndarray            # INT8 query-side input
    xkv: np.ndarray           # INT8 key/value-side input
    wq: np.ndarray            # INT8 weights (d_in, d)
    wk: np.ndarray
    wv: np.ndarray
    scales: AttnScales


def _absmax_scale(a, unit: float) -> float:
    m = float(np.abs(np.asarray(a)).max()) if np.size(a) else 0.0
    return m * unit / 127 if m > 0 else unit


def calibrate(X, Wq, Wk, Wv, *, X_kv=None, causal: bool = False, qpos0: int = 0,
              z_quant_max: int = 127) -> Calibration:
    """Per-tensor absmax/127 scales, as gm.attention_head_cim does, from float data.

    X (Nq, d_in), X_kv (Nk, d_kv) (default X), W* float (d_in, d) / (d_kv, d).
    The score scale is the absmax of the (unmasked) real scores; the exp LUT is
    gm.build_exp_lut(score_scale). Returns INT8 tensors + AttnScales.
    """
    X = np.asarray(X, np.float64)
    Xkv = X if X_kv is None else np.asarray(X_kv, np.float64)
    sx, sxkv = (_absmax_scale(a, 1.0) for a in (X, Xkv))
    xq, xkv = gm.to_int8(X, sx), gm.to_int8(Xkv, sxkv)
    sw = [_absmax_scale(W, 1.0) for W in (Wq, Wk, Wv)]
    wq, wk, wv = (gm.to_int8(W, s) for W, s in zip((Wq, Wk, Wv), sw))
    rq, so, ints = {}, {}, {}
    for name, xin, s_in, w, s_w in (("q", xq, sx, wq, sw[0]), ("k", xkv, sxkv, wk, sw[1]),
                                    ("v", xkv, sxkv, wv, sw[2])):
        acc = xin @ w
        so[name] = _absmax_scale(acc, s_in * s_w)
        rq[name] = gm.ratio_to_scale_shift(s_in * s_w / so[name])
        ints[name] = ref_quantise(acc, rq[name])
    d = wq.shape[1]
    acc_s = ints["q"] @ ints["k"].T
    unit = so["q"] * so["k"] / math.sqrt(d)
    if causal:
        mask = np.arange(len(xkv))[None, :] > (np.arange(len(xq))[:, None] + qpos0)
        acc_s = acc_s[~mask]
    score_scale = _absmax_scale(acc_s, unit)
    rq_s = gm.ratio_to_scale_shift(unit / score_scale)
    lut = gm.build_exp_lut(score_scale, z_quant_max)
    sc = AttnScales(rq["q"], rq["k"], rq["v"], rq_s, lut, out_scale=so["v"],
                    meta=dict(sx=sx, sx_kv=sxkv, sw=tuple(sw), sq=so["q"], sk=so["k"],
                              sv=so["v"], score_scale=score_scale))
    return Calibration(xq, xkv, wq, wk, wv, sc)


def float_attention(X, Wq, Wk, Wv, *, X_kv=None, causal: bool = False, qpos0: int = 0):
    """softmax(Q K^T / sqrt(d)) V in float64."""
    X = np.asarray(X, np.float64)
    Xkv = X if X_kv is None else np.asarray(X_kv, np.float64)
    Q, K, V = X @ Wq, Xkv @ Wk, Xkv @ Wv
    s = Q @ K.T / np.sqrt(Q.shape[1])
    if causal:
        s = np.where(np.arange(len(K))[None, :] > (np.arange(len(Q))[:, None] + qpos0), -np.inf, s)
    s -= s.max(axis=1, keepdims=True)
    p = np.exp(s)
    return (p / p.sum(axis=1, keepdims=True)) @ V


def cosine(a, b) -> float:
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------
def _check_int8(*arrs):
    for a in arrs:
        a = np.asarray(a)
        assert np.issubdtype(a.dtype, np.integer) and a.min(initial=0) >= -128 \
            and a.max(initial=0) <= 127, "expected INT8 data"


def matmul(X, W, scale: int | None = None, shift: int | None = None, *, ratio: float | None = None,
           raw32: bool = False, gbuf_bytes: int = 512 * 1024, acc_depth: int = 1024) -> Build:
    """Y = Q8(X @ W) (or the int32 accumulators with ``raw32``).

    X INT8 (N, K), W INT8 (K, M); (scale, shift) or a float ``ratio``
    (-> gm.ratio_to_scale_shift). K tiled in 64-row chunks through the ACC
    buffer, M in 32-column tiles.
    """
    X, W = np.asarray(X, np.int64), np.asarray(W, np.int64)
    _check_int8(X, W)
    N, K = X.shape
    M = W.shape[1]
    if not raw32:
        if ratio is not None:
            scale, shift = gm.ratio_to_scale_shift(ratio)
        assert scale is not None and shift is not None
    em = _Emitter(gbuf_bytes, acc_depth)
    xr = em.plan.act("X", N, K)
    wr = em.plan.act("WT", M, K)
    yr = em.plan.alloc("Y", N, M, elem=4 if raw32 else 1)
    em.input(xr, X)
    em.input(wr, W.T)
    em.matmul(xr, wr, yr, (scale, shift), raw=raw32)
    acc = X @ W
    ref = dict(Y=acc if raw32 else ref_matmul(X, W, (scale, shift)), acc=acc)
    return em.build(ref, ["Y"])


def _check_head(d: int):
    assert 1 <= d <= N_ROWS, "head dim must be <= 64 (one pass per q.k)"


def encoder_attention_head(Xq, Wq, Wk, Wv, scales: AttnScales, *, causal: bool = False,
                           gbuf_bytes: int = 512 * 1024, acc_depth: int = 1024) -> Build:
    """Encoder-only self-attention head (Sec. IV-C) incl. the Q/K/V projections.

    Xq INT8 (N, d_model), W* INT8 (d_model, d). Output "O": INT8 (N, d),
    real value = O * scales.out_scale.
    """
    Xq, Wq, Wk, Wv = (np.asarray(a, np.int64) for a in (Xq, Wq, Wk, Wv))
    _check_int8(Xq, Wq, Wk, Wv)
    N, dm = Xq.shape
    d = Wq.shape[1]
    _check_head(d)
    em = _Emitter(gbuf_bytes, acc_depth)
    p = em.plan
    xr = p.act("X", N, dm)
    wq, wk, wv = (p.act(n, d, dm) for n in ("WqT", "WkT", "WvT"))
    lut = p.alloc("LUT", 1, LUT_BYTES, signed=False)
    qr = p.act("Q", N, d)
    kr = p.act("K", N, d)
    vtr = p.act("VT", d, N)
    er = p.act("E", min(N, N_PART), N, signed=False)
    orr = p.alloc("O", N, d)
    em.input(xr, Xq)
    for r, W in ((wq, Wq), (wk, Wk), (wv, Wv)):
        em.input(r, W.T)
    em.input(lut, np.asarray(scales.exp_lut, np.int64)[None, :])
    em.matmul(xr, wq, qr, scales.q)
    em.matmul(xr, wk, kr, scales.k)
    em.matmul(xr, wv, vtr, scales.v, transposed=True)
    em.encoder_attn(qr, kr, vtr, er, orr, N, N, d, scales.score, lut, causal)
    Q, K, V = (ref_matmul(Xq, W, rq) for W, rq in ((Wq, scales.q), (Wk, scales.k), (Wv, scales.v)))
    att = ref_attention(Q, K, V, scales.score, scales.exp_lut, causal)
    ref = dict(Q=Q, K=K, V=V, VT=V.T, **att, O=att["out"])
    return em.build(ref, ["Q", "K", "VT", "O"])


def decoder_attention_step(x, Wq, Wk, Wv, k_cache, v_cache, scales: AttnScales, *,
                           causal: bool = True, gbuf_bytes: int = 512 * 1024,
                           acc_depth: int = 1024) -> Build:
    """Decoder-only step (Sec. IV-D, Eq. 3) for a KV cache of length L.

    x INT8 (d_model,) or (nt, d_model) new token(s) at positions L..L+nt-1;
    k_cache / v_cache INT8 (L, d). The step projects q, k, v of the new
    token(s), appends k and v to the cache *in GBUF* (K rows / V^T columns
    L..), then attends over the L+nt keys with K tiles (partition = key) and
    V^T tiles in the CIM and q as the activation. With ``causal`` (default)
    a new token does not see the ones after it (only matters for nt > 1).
    Outputs: "O" (nt, d), the extended caches "K" (L+nt, d) and "VT" (d, L+nt).
    """
    x = np.atleast_2d(np.asarray(x, np.int64))
    Wq, Wk, Wv = (np.asarray(a, np.int64) for a in (Wq, Wk, Wv))
    d = Wq.shape[1]
    kc = np.asarray(k_cache, np.int64).reshape(-1, d)
    vc = np.asarray(v_cache, np.int64).reshape(-1, d)
    _check_int8(x, Wq, Wk, Wv, kc, vc)
    _check_head(d)
    nt, dm = x.shape
    L = len(kc)
    Lt = L + nt
    em = _Emitter(gbuf_bytes, acc_depth)
    p = em.plan
    xr = p.act("X", nt, dm)
    wq, wk, wv = (p.act(n, d, dm) for n in ("WqT", "WkT", "WvT"))
    lut = p.alloc("LUT", 1, LUT_BYTES, signed=False)
    qr = p.act("Q", nt, d)
    kr = p.act("K", Lt, d)
    vtr = p.act("VT", d, Lt)
    er = p.act("E", min(nt, N_PART), Lt, signed=False)
    orr = p.alloc("O", nt, d)
    em.input(xr, x)
    for r, W in ((wq, Wq), (wk, Wk), (wv, Wv)):
        em.input(r, W.T)
    em.input(lut, np.asarray(scales.exp_lut, np.int64)[None, :])
    em.input(kr, np.concatenate([kc, np.zeros((nt, d), np.int64)]))
    em.input(vtr, np.concatenate([vc, np.zeros((nt, d), np.int64)]).T)
    em.matmul(xr, wq, qr, scales.q)
    em.matmul(xr, wk, kr, scales.k, row0=L)
    em.matmul(xr, wv, vtr, scales.v, row0=L, transposed=True)
    em.decoder_attn(qr, kr, vtr, er, orr, nt, Lt, d, scales.score, lut, causal, qpos0=L)
    q = ref_matmul(x, Wq, scales.q)
    K = np.concatenate([kc, ref_matmul(x, Wk, scales.k)])
    V = np.concatenate([vc, ref_matmul(x, Wv, scales.v)])
    att = ref_attention(q, K, V, scales.score, scales.exp_lut, causal, qpos0=L)
    ref = dict(Q=q, K=K, V=V, VT=V.T, **att, O=att["out"])
    return em.build(ref, ["Q", "K", "VT", "O"])


def cross_attention(Xd, Henc, Wq, Wk, Wv, scales: AttnScales, *,
                    gbuf_bytes: int = 512 * 1024, acc_depth: int = 1024) -> Build:
    """Encoder-decoder cross attention (Sec. IV-E): decoder queries, encoder K/V.

    Xd INT8 (Nd, d_model) decoder-side input (the previous layer's output),
    Henc INT8 (Ne, d_enc) encoder output; W* INT8. K, V are projected once
    from Henc; the attention uses the decoder mapping (K and V^T tiles in the
    CIM, up to 32 decoder queries streamed per group), non-causal.
    """
    Xd, He = np.atleast_2d(np.asarray(Xd, np.int64)), np.asarray(Henc, np.int64)
    Wq, Wk, Wv = (np.asarray(a, np.int64) for a in (Wq, Wk, Wv))
    _check_int8(Xd, He, Wq, Wk, Wv)
    d = Wq.shape[1]
    _check_head(d)
    Nd, dm = Xd.shape
    Ne, de = He.shape
    em = _Emitter(gbuf_bytes, acc_depth)
    p = em.plan
    xr = p.act("Xd", Nd, dm)
    hr = p.act("H", Ne, de)
    wq = p.act("WqT", d, dm)
    wk, wv = (p.act(n, d, de) for n in ("WkT", "WvT"))
    lut = p.alloc("LUT", 1, LUT_BYTES, signed=False)
    qr = p.act("Q", Nd, d)
    kr = p.act("K", Ne, d)
    vtr = p.act("VT", d, Ne)
    er = p.act("E", min(Nd, N_PART), Ne, signed=False)
    orr = p.alloc("O", Nd, d)
    em.input(xr, Xd)
    em.input(hr, He)
    for r, W in ((wq, Wq), (wk, Wk), (wv, Wv)):
        em.input(r, W.T)
    em.input(lut, np.asarray(scales.exp_lut, np.int64)[None, :])
    em.matmul(xr, wq, qr, scales.q)
    em.matmul(hr, wk, kr, scales.k)
    em.matmul(hr, wv, vtr, scales.v, transposed=True)
    em.decoder_attn(qr, kr, vtr, er, orr, Nd, Ne, d, scales.score, lut, False, qpos0=0)
    q = ref_matmul(Xd, Wq, scales.q)
    K, V = ref_matmul(He, Wk, scales.k), ref_matmul(He, Wv, scales.v)
    att = ref_attention(q, K, V, scales.score, scales.exp_lut)
    ref = dict(Q=q, K=K, V=V, VT=V.T, **att, O=att["out"])
    return em.build(ref, ["Q", "K", "VT", "O"])


# ---------------------------------------------------------------------------
# Running a Build on the ISA simulator directly (fast path for tests)
# ---------------------------------------------------------------------------
def run_sim(build: Build, sim=None, max_len: int = MAX_PROG, **simkw):
    """-> (outputs dict, summed status dict, sim). Uses the simulator backdoor."""
    from .sim import CplSim
    sim = CplSim(gbuf_bytes=build.layout.gbuf_bytes, **simkw) if sim is None else sim
    sim.gbuf_write(0, build.layout.image())
    total = dict(cycles=0, instret=0, passes=0, runs=0)
    st = None
    for seg in build.segments(max_len):
        sim.imem[:] = 0
        sim.load_program(seg, 0)
        st = sim.run(0)
        if st["error"] or not st["done"]:
            raise RuntimeError(f"program failed: {st}")
        for k in ("cycles", "instret", "passes"):
            total[k] += st[k]
        total["runs"] += 1
    outs = {n: build.layout.extract(n, sim.gbuf_read) for n in build.layout.outputs}
    return outs, total, sim
