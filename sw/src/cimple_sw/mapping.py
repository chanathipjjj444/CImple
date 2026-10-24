"""
Mapping simulator: whole multi-head attention layers on a cycle-annotated,
bit-exact model of the CIMple accelerator.

Reference: Ahn et al., "CIMple: Standard-cell SRAM-based CIM with LUT-based
split softmax for attention acceleration", arXiv:2604.15944v2, Sec. IV-A..E.

What is simulated
-----------------
* ``CimpleCore`` -- 2 banks x 32 partitions x 64 rows of INT8 weights, one bank
  active [P]. ``mac`` is one 8-cycle bit-serial pass (MSB first, sign cycle
  subtracts, MSB nibble signed / LSB nibble unsigned -- the golden model's
  scheme, docs/01 Sec. 2), vectorised over partitions and over a batch of
  activation vectors. Writes go through the 128b port, 16 weights per write
  [P]; one write addresses 16 weights of one partition row of one bank, so a
  (p x r) tile costs p * ceil(r / 16) cycles [D: WA is 8b = 2 banks x 32
  partitions x 4 segments of 128b]. Writing the active bank while it computes
  is an assertion failure.
* The quantisation unit is ``gm.quantise`` (round-half-away, saturating) with
  (scale, shift) from ``gm.ratio_to_scale_shift``; the split softmax uses the
  golden 8b exp / reciprocal LUTs, ``mantissa8`` and ``msb_pos``.

The three mappings
------------------
1. Weight projection (Sec. IV-C, red flow): Q, K, V = X W with W stationary,
   tiled 64 rows x 32 columns; partial sums over 64-row K tiles accumulate in
   the 32b intermediate buffer, then the quantisation unit makes INT8. The
   optional W_O projection after head concatenation uses the same flow.
2. Encoder-only (Sec. IV-C, blue): per head, a 32-query tile of Q is written
   into one bank (Q "buffered in SRAM"); K^T is streamed as activations, so one
   pass gives the scores of 32 queries against one key. Scores are quantised,
   looked up in the exp LUT, and the e values of 64 keys form the activation
   of an A'V pass against a V tile (32 head-dim partitions x 64 key rows) in
   the other bank. Denominators accumulate across 64-key chunks (no row-max
   pass: z_quant_max is constant, Sec. IV-B); the reciprocal LUT and the
   quantiser normalise at the end. Sec. IV-A also says the softmax output goes
   "to the write data for encoder-only computations"; that contradicts Sec.
   IV-C's "storing V and Q in the CIM and feeding K^T as input", which is
   followed here: e goes to the input driver in every mapping, and the
   write-data path is not modelled.
3. Decoder-only (Sec. IV-D, green; Eq. 3): one token at a time. K^T tiles (32
   keys x head dim) and V tiles are written into the CIM from the KV cache, the
   current q is the activation, QK^T stays in the input buffer.
4. Encoder-decoder (Sec. IV-E): encoder layer (mapping 2), then per decoder
   token a causal self-attention step (mapping 3) whose output is the input of
   the cross-attention Q projection; the cross K, V are projected once from the
   encoder output and written into the CIM per step like a KV cache.

Exp values as activations  [A]
-------------------------------
The exp LUT gives UQ0.8 (0..255) but the core reads activations as signed
INT8. ``e_mode="shift1"`` (default) feeds e >> 1 (UQ0.7) and accumulates the
denominator over the same e >> 1, so the ratio is unchanged and one LSB of
numerator precision is lost (same decision as the user's nearmem_model.py; no
RTL change). ``e_mode="unsigned"`` models a 1-bit unsigned-activation mode in
which the sign cycle adds instead of subtracting (an RTL change) and feeds e.

E buffer (corner turn)  [A]
---------------------------
A QK^T pass yields one key's scores for 32 queries (a column of S), but an
A'V pass needs one query's e values for 64 keys (a row). The encoder therefore
needs an 8b e buffer of 32 queries x 64 keys (2 kB) between the exp LUT and the
input driver, written column-wise and read row-wise; the decoder needs 64
entries. The paper does not describe it. Its peak use is ``buf_e_max``. One
buffer suffices with the ping-pong schedule below (latency.py's
``interleave=True`` needs two).

Exp underflow  [P]-inherited
----------------------------
The exp LUT holds round(255 * exp((z - z_quant_max) * s_score)) with a
constant z_quant_max and a per-head absmax score scale, so a row whose max
score is far below the head's absmax gets e = 0 for every key (more so after
the shift1 ``>> 1``). Its exp-sum is 0 and, as in ``gm.softmax_row``, the row
is output as 0 with no flag; ``Stats.zero_rows`` counts these rows. Measured
(tests/test_mapping.py): with unit-variance inputs no rows are lost and the
cosine to float attention is > 0.95; scaling the inputs by 2 zeroes most rows
and the cosine falls below 0.6. The accuracy claim holds only in that regime
(cf. README "Known limitations", cimple_hw8).

Schedule assumptions  [A]
-------------------------
* Bank timing is event-driven: a write to bank b starts when the write port is
  free and b has been deselected; a pass starts when the core is free and the
  active bank's last write has finished. So the next tile's write overlaps the
  current tile's passes (double buffering) and only the excess shows as
  ``stall``. Bank select is free.
* Quantiser and LUT process 4 lanes per cycle (hw.Timing) and run concurrently
  with the core, pipelined behind the 4-lane output mux; fill latency ignored.
  A phase (projection / attention / W_O) takes max(core span incl. stalls,
  quantiser cycles, LUT cycles). Phases are serialised (data dependence), no
  prefetch across a phase boundary.
* The exp LUT is reloaded per head (256 x 8b over a 128b port = 16 cycles),
  counted as LUT cycles. Exp lookups are counted for every computed score
  (masked ones too); one reciprocal lookup per output row.
* The intermediate buffer is assumed deep enough for one 32-column output tile
  of the whole sequence; its peak occupancy is reported (``buf_acc_max``).
  KV-cache / input-buffer bandwidth is not modelled (free).
* Causal encoder: key chunks entirely in the future of a query tile are
  skipped; masked scores inside a chunk are computed and their e gated to 0.
* Head dim <= 64 (one pass gives a full q.k dot product). A head dim > 32 needs
  two V tiles per chunk. Encoder tiles use ping-pong placement (``load_next``,
  latency.py ``bank_policy="pingpong"``): a non-resident tile goes to the bank
  the previous pass did not use, so with two V tiles the second V tile
  overwrites the Q tile and Q is re-written from the projection buffer behind
  the next A'V pass. Every write then hides behind a pass (at N=1024, d=64 the
  attention phase is 1 048 704 cycles vs latency.py's default 1 049 730, the
  difference being its pipeline drain) at the cost of one extra Q tile write
  per chunk. Keeping Q pinned in one bank with this loop
  order would expose the second V write every chunk (~6% at N=1024, d=64,
  latency.py ``bank_policy="fixed", interleave=False``). With head dim <= 32
  Q stays resident in one bank and V tiles stream through the other.
* Scales are per-tensor (absmax / 127) for X, W_*, Q, K, V, W_O output and per
  head for the score (one exp LUT per head) -- like gm.attention_head_cim. The
  hardware holds static scales; ``calibrate`` computes them from the data
  with plain integer maths, or callers pass precomputed ``Scales``.
* Attention block only: no residual, LayerNorm, FFN or RoPE.
"""
from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass, field, fields
from typing import Literal

import numpy as np

from cimple_rtl.model import cimple_model as gm

from .hw import ACC_W, N_BANKS, N_LANES, N_PART, N_ROWS, PEAK_OPS_PER_CYCLE, Timing

EMode = Literal["shift1", "unsigned"]

ACC_MIN, ACC_MAX = -(1 << (ACC_W - 1)), (1 << (ACC_W - 1)) - 1
MAC_MIN, MAC_MAX = -(1 << (gm.MAC_W - 1)), (1 << (gm.MAC_W - 1)) - 1
LUT_LOAD_CYCLES = gm.LUT_DEPTH * gm.LUT_W // 128     # [A] 16: LUT refill per head
KEY_CHUNK = N_ROWS      # [D] 64 keys per A'V activation vector
Q_TILE = N_PART         # [D] 32 queries (or keys) per score tile

STAGES = ("proj", "qkt", "softmax", "av", "norm", "out_proj")
PHASES = ("proj", "attn", "out_proj")


# ---------------------------------------------------------------------------
# Integer helpers
# ---------------------------------------------------------------------------
def quantise_vec(acc, scale: int, shift: int):
    """Vectorised ``gm.quantise``: sat8(round_half_away(acc * scale / 2**shift))."""
    prod = np.asarray(acc, dtype=np.int64) * int(scale)
    assert np.all(np.abs(prod) < (1 << 62))
    if shift:
        half = 1 << (shift - 1)
        prod = prod + np.where(prod >= 0, half, -half)
    return np.clip(prod >> shift, -128, 127)


def acc32(x):
    """The 32b intermediate accumulator. Overflow would wrap in hardware; here
    it is an error, because the attention maths assumes it never happens."""
    x = np.asarray(x, dtype=np.int64)
    assert np.all((x >= ACC_MIN) & (x <= ACC_MAX)), "32b intermediate buffer overflow"
    return x


def normalise_rows(partial, den):
    """Split-softmax denominator: partial (n, d) 32b, den (n,) exp-sums -> INT8.

    M = RECIP_LUT[mantissa8(s)], shift = RECIP_FRAC - LUT_W + msb_pos(s), as in
    ``gm.softmax_row``. Valid for UQ0.8 or UQ0.7 numerators alike: only the
    ratio partial / s is formed. A row with s = 0 (every e underflowed to 0)
    gives an all-zero output with no flag, exactly as ``gm.softmax_row``; the
    simulator counts such rows in ``Stats.zero_rows``.
    """
    recip = _RECIP_LUT
    out = np.empty(np.shape(partial), dtype=np.int64)
    for i, s in enumerate(np.asarray(den, dtype=np.int64)):
        s = int(s)
        assert 0 <= s < (1 << gm.SUM_W), "exp-sum overflow"
        M = int(recip[gm.mantissa8(s)])
        sh = gm.RECIP_FRAC - gm.LUT_W + gm.msb_pos(s)
        out[i] = quantise_vec(partial[i], M, sh)
    return out


_RECIP_LUT = gm.build_recip_lut()


def e_activation(e, e_mode: EMode):
    """Exp-LUT output (UQ0.8) -> the activation fed to the A'V pass."""
    if e_mode == "shift1":
        return e >> 1
    if e_mode == "unsigned":
        return e
    raise ValueError(e_mode)


def _absmax_scale(acc, unit: float) -> float:
    """Per-tensor INT8 scale of an accumulator whose real value is acc * unit."""
    m = int(np.abs(np.asarray(acc)).max()) if np.size(acc) else 0
    return m * unit / 127 if m > 0 else unit


# ---------------------------------------------------------------------------
# The CIM core
# ---------------------------------------------------------------------------
def core_pass(weights, acts, unsigned: bool = False):
    """One bit-serial pass of the whole array, batched over activation vectors.

    weights : (N_PART, N_ROWS) INT8 (the active bank)
    acts    : (B, N_ROWS) INT8, or 0..255 when ``unsigned``
    returns : (B, N_PART) 23b accumulators

    MSB first; on the sign cycle the accumulator is loaded with -comb (signed
    activations) or +comb (``unsigned``: the [A] unsigned-activation mode).
    Same algorithm as ``gm.partition_mac``; the adder trees are matmuls over
    0/1 bit planes, done in float64, which is exact (|sums| <= 960).
    """
    msb, lsb = gm.nibble_split(weights)
    a_raw = gm.u8(acts)
    if unsigned:
        assert np.all((np.asarray(acts) >= 0) & (np.asarray(acts) <= 255))
    else:
        assert np.all((np.asarray(acts) >= -128) & (np.asarray(acts) <= 127))
    msbT, lsbT = msb.T.astype(np.float64), lsb.T.astype(np.float64)
    acc = None
    for k in range(gm.DATA_W - 1, -1, -1):
        xb = ((a_raw >> k) & 1).astype(np.float64)
        s_msb = np.rint(xb @ msbT).astype(np.int64)
        s_lsb = np.rint(xb @ lsbT).astype(np.int64)
        assert s_msb.min() >= -(1 << (gm.TREE_W - 1)) and s_msb.max() < (1 << (gm.TREE_W - 1))
        assert s_lsb.min() >= 0 and s_lsb.max() < (1 << gm.TREE_W)
        comb = (s_msb << gm.NIB_W) + s_lsb
        if k == gm.DATA_W - 1:
            acc = comb if unsigned else -comb
        else:
            acc = (acc << 1) + comb
    assert acc.min() >= MAC_MIN and acc.max() <= MAC_MAX
    return acc


@dataclass
class StageStats:
    passes: int = 0      # activation vectors through the core
    macs: int = 0        # useful (unpadded) MACs
    core: int = 0        # core busy cycles = passes * pass_cycles
    stall: int = 0       # core waiting for a bank write (exposed write)
    write: int = 0       # 128b write-port cycles issued
    tiles: int = 0       # tile writes
    quant: int = 0       # quantiser cycles (4 lanes / cycle)
    lut: int = 0         # exp + reciprocal LUT cycles, incl. per-head reloads

    def add(self, o: "StageStats"):
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(o, f.name))


@dataclass
class Stats:
    """Per-stage counters and the phase wall-clock (see module [A]s)."""

    timing: Timing = field(default_factory=Timing)
    stages: dict = field(default_factory=lambda: {s: StageStats() for s in STAGES})
    phases: dict = field(default_factory=lambda: {p: 0 for p in PHASES})
    buf_acc_max: int = 0     # peak 32b intermediate-buffer words in use
    buf_e_max: int = 0       # peak 8b e-buffer entries (exp values awaiting A'V)
    zero_rows: int = 0       # output rows whose exp-sum was 0 (forced to 0)
    tokens: int = 0          # decoder steps (0 for an encoder layer)

    @property
    def cycles(self) -> int:
        return sum(self.phases.values())

    @property
    def macs(self) -> int:
        return sum(s.macs for s in self.stages.values())

    @property
    def ops(self) -> int:
        return 2 * self.macs

    @property
    def utilisation(self) -> float:
        """Useful OP / (cycles * 512 OP/cycle peak)."""
        return self.ops / (self.cycles * PEAK_OPS_PER_CYCLE) if self.cycles else 0.0

    @property
    def time_us(self) -> float:
        return self.cycles / self.timing.freq_mhz

    @property
    def tops(self) -> float:
        """Delivered throughput, useful OP / time (1 MAC = 2 OP, [P] Table I note 4)."""
        return self.ops / (self.time_us * 1e6) if self.cycles else 0.0

    def merge(self, o: "Stats") -> "Stats":
        for k in STAGES:
            self.stages[k].add(o.stages[k])
        for k in PHASES:
            self.phases[k] += o.phases[k]
        self.buf_acc_max = max(self.buf_acc_max, o.buf_acc_max)
        self.buf_e_max = max(self.buf_e_max, o.buf_e_max)
        self.zero_rows += o.zero_rows
        self.tokens += o.tokens
        return self

    def table(self, title: str = "") -> str:
        cols = [f.name for f in fields(StageStats)]
        lines = [title] if title else []
        lines.append(f"{'stage':9s}" + "".join(f"{c:>12s}" for c in cols))
        for k in STAGES:
            s = self.stages[k]
            lines.append(f"{k:9s}" + "".join(f"{getattr(s, c):12d}" for c in cols))
        ph = ", ".join(f"{k} {v}" for k, v in self.phases.items())
        lines.append(f"phases: {ph}")
        lines.append(
            f"total {self.cycles} cycles = {self.time_us:.2f} us @ {self.timing.freq_mhz:.0f} MHz; "
            f"{self.macs} MACs, {self.ops} OPs; utilisation {100 * self.utilisation:.2f}% "
            f"of {PEAK_OPS_PER_CYCLE} OP/cycle; peak acc buffer {self.buf_acc_max} x 32b, "
            f"e buffer {self.buf_e_max} x 8b")
        if self.zero_rows:
            lines.append(f"WARNING: {self.zero_rows} (row, head) outputs had exp-sum 0 "
                         f"(exp LUT underflow) and were output as 0")
        lines.append(
            f"throughput {1e3 * self.tops:.1f} GOPS delivered, "
            f"{PEAK_OPS_PER_CYCLE * self.timing.freq_mhz / 1e3:.1f} GOPS peak "
            f"@ {self.timing.freq_mhz:g} MHz")
        return "\n".join(lines)


class CimpleCore:
    """Two banks of (32 x 64) INT8 weights, one active, with event timing."""

    def __init__(self, timing: Timing = Timing(), seed: int = 0):
        rng = np.random.default_rng(seed)
        # power-up garbage: proves unused rows/partitions never leak into results
        self.banks = rng.integers(-128, 128, (N_BANKS, N_PART, N_ROWS)).astype(np.int64)
        self.timing = timing
        self.active = 0
        self.computing = False
        self.holds = [None] * N_BANKS
        self.t_core = 0
        self.t_write = 0
        self.t_ready = [0] * N_BANKS
        self.t_released = [0] * N_BANKS
        self.stats = Stats(timing)
        self.stage = "proj"

    def _st(self) -> StageStats:
        return self.stats.stages[self.stage]

    def write(self, bank: int, tile, key=None) -> int:
        """Write a (p <= 32 partitions, r <= 64 rows) tile into ``bank``.

        Rows/partitions outside the tile keep stale data. Returns write cycles.
        """
        tile = np.asarray(tile, dtype=np.int64)
        p, r = tile.shape
        assert p <= N_PART and r <= N_ROWS
        assert tile.min(initial=0) >= -128 and tile.max(initial=0) <= 127
        assert not (bank == self.active and self.computing), \
            "write to the active bank while it computes"
        cyc = p * Timing.write_cycles(r)
        free = self.t_core if bank == self.active else self.t_released[bank]
        start = max(self.t_write, free)
        self.t_write = start + cyc
        self.t_ready[bank] = self.t_write
        self.banks[bank, :p, :r] = tile
        self.holds[bank] = key
        st = self._st()
        st.write += cyc
        st.tiles += 1
        return cyc

    def select(self, bank: int):
        if bank != self.active:
            self.t_released[self.active] = self.t_core
            self.active = bank
            self.computing = False

    def load_next(self, tile, key=None):
        """Write ``tile`` into the shadow bank (unless it already holds ``key``)
        and make it active. Its write overlaps whatever ran before."""
        b = self.active ^ 1
        if key is None or self.holds[b] != key:
            self.write(b, tile, key)
        self.select(b)

    def mac(self, acts, used: tuple[int, int], unsigned: bool = False):
        """Run len(acts) passes on the active bank.

        acts : (B, r <= 64) activations (zero-padded to 64 rows)
        used : (partitions, rows) of useful weights, for the MAC count only
        returns (B, 32) 23b partition accumulators.
        """
        acts = np.atleast_2d(np.asarray(acts, dtype=np.int64))
        B, r = acts.shape
        assert r <= N_ROWS
        if r < N_ROWS:
            acts = np.pad(acts, ((0, 0), (0, N_ROWS - r)))
        start = max(self.t_core, self.t_ready[self.active])
        cyc = B * self.timing.pass_cycles
        st = self._st()
        st.stall += start - self.t_core
        st.core += cyc
        st.passes += B
        st.macs += B * used[0] * used[1]
        self.t_core = start + cyc
        self.computing = True
        return core_pass(self.banks[self.active], acts, unsigned)


class _Sim:
    """Core + phase bookkeeping (quantiser and LUT counted per phase)."""

    def __init__(self, timing: Timing, seed: int):
        self.core = CimpleCore(timing, seed)
        self.timing = timing
        self._tags = itertools.count()
        self._phase = None

    def use(self, stats: Stats):
        self.core.stats = stats

    def tag(self):
        return next(self._tags)

    def stage(self, name: str):
        self.core.stage = name

    def quant(self, n: int):
        self.core._st().quant += -(-n // N_LANES) * self.timing.quant_cycles

    def lut(self, n: int, reload: bool = False):
        c = -(-n // N_LANES) * self.timing.lut_cycles + (LUT_LOAD_CYCLES if reload else 0)
        self.core._st().lut += c

    def buf(self, words: int):
        self.core.stats.buf_acc_max = max(self.core.stats.buf_acc_max, words)

    def ebuf(self, n: int):
        self.core.stats.buf_e_max = max(self.core.stats.buf_e_max, n)

    def zero_rows(self, den):
        self.core.stats.zero_rows += int(np.count_nonzero(np.asarray(den) == 0))

    def begin(self, phase: str):
        c = self.core
        t0 = max(c.t_core, c.t_write)
        c.t_core = c.t_write = t0
        snap = {k: (s.quant, s.lut) for k, s in c.stats.stages.items()}
        self._phase = (phase, t0, snap)

    def end(self):
        c = self.core
        phase, t0, snap = self._phase
        q = sum(s.quant - snap[k][0] for k, s in c.stats.stages.items())
        lu = sum(s.lut - snap[k][1] for k, s in c.stats.stages.items())
        span = max(c.t_core, c.t_write) - t0
        dur = max(span, q, lu)
        c.stats.phases[phase] += dur
        c.t_core = c.t_write = t0 + dur
        self._phase = None


# ---------------------------------------------------------------------------
# Weights, scales and the plain-integer calibration model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AttnWeights:
    """Float weights of one attention block, X @ W orientation.

    wq : (d_in, H*dh), wk, wv : (d_kv, H*dh), wo : (H*dh, d_out) or None.
    """

    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    n_heads: int
    wo: np.ndarray | None = None

    @property
    def d_head(self) -> int:
        return self.wq.shape[1] // self.n_heads

    @staticmethod
    def random(d_model: int, n_heads: int, d_head: int, rng, d_kv: int | None = None,
               out_proj: bool = True) -> "AttnWeights":
        d_kv = d_model if d_kv is None else d_kv
        hd = n_heads * d_head
        return AttnWeights(
            wq=rng.standard_normal((d_model, hd)) / np.sqrt(d_model),
            wk=rng.standard_normal((d_kv, hd)) / np.sqrt(d_kv),
            wv=rng.standard_normal((d_kv, hd)) / np.sqrt(d_kv),
            n_heads=n_heads,
            wo=rng.standard_normal((hd, d_model)) / np.sqrt(hd) if out_proj else None)


@dataclass(frozen=True)
class Scales:
    """Real value = INT8 * scale. Static per-tensor scales (score: per head)."""

    x: float                     # query-side input
    x_kv: float                  # key/value-side input (== x for self-attention)
    wq: float
    wk: float
    wv: float
    wo: float | None
    q: float
    k: float
    v: float
    score: tuple                 # per head
    o: float | None              # W_O output

    @property
    def out(self) -> float:
        return self.o if self.o is not None else self.v

    def proj_ratio(self, which: str) -> float:
        src = self.x if which == "q" else self.x_kv
        return src * getattr(self, "w" + which) / getattr(self, which)

    def score_ratio(self, h: int, d_head: int) -> float:
        return self.q * self.k / math.sqrt(d_head) / self.score[h]

    def out_ratio(self) -> float:
        return self.v * self.wo / self.o


@dataclass(frozen=True)
class _QW:
    """INT8 weights + the requant (scale, shift) pairs derived from Scales."""

    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    wo: np.ndarray | None
    n_heads: int
    d_head: int
    rq: dict
    luts: tuple
    tag: int


def _quantise_weights(w: AttnWeights, sc: Scales, z_quant_max: int, tag: int) -> _QW:
    dh = w.d_head
    rq = {k: gm.ratio_to_scale_shift(sc.proj_ratio(k)) for k in "qkv"}
    rq["score"] = [gm.ratio_to_scale_shift(sc.score_ratio(h, dh)) for h in range(w.n_heads)]
    if w.wo is not None:
        rq["o"] = gm.ratio_to_scale_shift(sc.out_ratio())
    luts = tuple(gm.build_exp_lut(s, z_quant_max) for s in sc.score)
    return _QW(gm.to_int8(w.wq, sc.wq), gm.to_int8(w.wk, sc.wk), gm.to_int8(w.wv, sc.wv),
               None if w.wo is None else gm.to_int8(w.wo, sc.wo),
               w.n_heads, dh, rq, luts, tag)


def calibrate(xq, sx: float, w: AttnWeights, xkv=None, sx_kv: float | None = None, *,
              causal: bool = False, e_mode: EMode = "shift1", z_quant_max: int = 127) -> Scales:
    """Static scales for one attention block, from INT8 inputs (plain integer maths).

    Each tensor's scale is absmax/127 of its real value, computed stage by stage
    on the quantised data exactly as the hardware would produce it.
    """
    xq = np.asarray(xq, dtype=np.int64)
    xkv = xq if xkv is None else np.asarray(xkv, dtype=np.int64)
    sx_kv = sx if sx_kv is None else sx_kv
    H, dh = w.n_heads, w.d_head
    sw = {k: float(np.abs(getattr(w, "w" + k)).max()) / 127 for k in "qkv"}
    sw["o"] = None if w.wo is None else float(np.abs(w.wo).max()) / 127
    acc = {"q": xq @ gm.to_int8(w.wq, sw["q"]),
           "k": xkv @ gm.to_int8(w.wk, sw["k"]),
           "v": xkv @ gm.to_int8(w.wv, sw["v"])}
    so = {k: _absmax_scale(acc[k], (sx if k == "q" else sx_kv) * sw[k]) for k in "qkv"}
    ints = {}
    for k in "qkv":
        unit = (sx if k == "q" else sx_kv) * sw[k]
        ints[k] = quantise_vec(acc[k], *gm.ratio_to_scale_shift(unit / so[k]))
    Q, K, V = ints["q"], ints["k"], ints["v"]
    mask = _causal_mask(len(Q), len(K)) if causal else None
    s_score, heads = [], []
    for h in range(H):
        sl = slice(h * dh, (h + 1) * dh)
        a_s = Q[:, sl] @ K[:, sl].T
        unit = so["q"] * so["k"] / math.sqrt(dh)
        ss = _absmax_scale(a_s[~mask] if causal else a_s, unit)
        s_score.append(ss)
        z = quantise_vec(a_s, *gm.ratio_to_scale_shift(unit / ss))
        e = gm.build_exp_lut(ss, z_quant_max)[gm.u8(z)]
        if causal:
            e[mask] = 0
        a = e_activation(e, e_mode)
        heads.append(normalise_rows(a @ V[:, sl], a.sum(1)))
    s_o = None
    if w.wo is not None:
        cat = np.concatenate(heads, axis=1)
        s_o = _absmax_scale(cat @ gm.to_int8(w.wo, sw["o"]), so["v"] * sw["o"])
    return Scales(x=sx, x_kv=sx_kv, wq=sw["q"], wk=sw["k"], wv=sw["v"], wo=sw["o"],
                  q=so["q"], k=so["k"], v=so["v"], score=tuple(s_score), o=s_o)


def _causal_mask(nq: int, nk: int, q_off: int = 0):
    """True where key j is in the future of query i (absolute positions)."""
    return np.arange(nk)[None, :] > (np.arange(nq)[:, None] + q_off)


def input_scale(X) -> float:
    """Per-tensor INT8 scale of a float input (absmax / 127)."""
    m = float(np.abs(X).max())
    return m / 127 if m > 0 else 1.0


# ---------------------------------------------------------------------------
# Mapping building blocks (all run on the simulated core)
# ---------------------------------------------------------------------------
def _ws_matmul(sim: _Sim, A, W, name: str, tag):
    """Y = A @ W, W stationary (Sec. IV-C weight projection), 32b result.

    Loop order [A]: 32-column tile, 64-row K tile, then all tokens of A as a
    batch of passes; partial sums of K tiles accumulate in the 32b buffer.
    """
    core = sim.core
    N, K = A.shape
    M = W.shape[1]
    acc = np.zeros((N, M), dtype=np.int64)
    for m0 in range(0, M, N_PART):
        for k0 in range(0, K, N_ROWS):
            tile = W[k0:k0 + N_ROWS, m0:m0 + N_PART].T          # partitions x rows
            p, r = tile.shape
            core.load_next(tile, key=(tag, name, m0, k0))
            out = core.mac(A[:, k0:k0 + r], used=(p, r))[:, :p]
            acc[:, m0:m0 + p] = acc32(out if k0 == 0 else acc[:, m0:m0 + p] + out)
            if K > N_ROWS:
                sim.buf(N * p)
    return acc


def _project(sim: _Sim, x, W, rq, name: str, tag, stage: str = "proj"):
    sim.stage(stage)
    acc = _ws_matmul(sim, x, W, name, tag)
    sim.quant(acc.size)
    return quantise_vec(acc, *rq)


def _attn_encoder_head(sim: _Sim, Q, K, V, rq, lut, causal: bool, e_mode: EMode, tag):
    """Encoder-only activation-to-activation for one head (Q, V in CIM, K^T streamed)."""
    core = sim.core
    N, dh = Q.shape
    Nk = K.shape[0]
    assert dh <= N_ROWS, "head dim > 64 needs a row-tiled QK^T (not modelled)"
    unsigned = e_mode == "unsigned"
    out = np.zeros((N, dh), dtype=np.int64)
    sim.stage("softmax")
    sim.lut(0, reload=True)
    for q0 in range(0, N, Q_TILE):
        qt = Q[q0:q0 + Q_TILE]
        nq = len(qt)
        acc = np.zeros((nq, dh), dtype=np.int64)
        den = np.zeros(nq, dtype=np.int64)
        for c0 in range(0, Nk, KEY_CHUNK):
            if causal and c0 > q0 + nq - 1:
                break                                   # whole chunk in the future
            kc = K[c0:c0 + KEY_CHUNK]
            nk = len(kc)
            sim.stage("qkt")
            core.load_next(qt, key=(tag, "Q", q0))      # resident unless a V tile evicted it
            s = core.mac(kc, used=(nq, dh))[:, :nq].T   # (nq, nk): one pass per key
            sim.ebuf(s.size)                            # corner-turn: (nk, nq) -> (nq, nk)
            sim.quant(s.size)
            z = quantise_vec(s, *rq)
            sim.stage("softmax")
            e = lut[gm.u8(z)]
            sim.lut(e.size)
            if causal:
                e[_causal_mask(nq, nk, q0 - c0)] = 0
            a = e_activation(e, e_mode)
            den += a.sum(1)
            sim.stage("av")
            for d0 in range(0, dh, N_PART):
                vt = V[c0:c0 + nk, d0:d0 + N_PART].T      # (pd, nk)
                pd = vt.shape[0]
                core.load_next(vt, key=(tag, "V", c0, d0))
                pv = core.mac(a, used=(pd, nk), unsigned=unsigned)[:, :pd]
                acc[:, d0:d0 + pd] = acc32(acc[:, d0:d0 + pd] + pv)
        sim.buf(nq * dh)
        sim.stage("softmax")
        sim.lut(nq)                                      # reciprocal lookups
        sim.stage("norm")
        sim.quant(acc.size)
        sim.zero_rows(den)
        out[q0:q0 + nq] = normalise_rows(acc, den)
    return out


def _attn_decode_head(sim: _Sim, q, Kc, Vc, rq, lut, e_mode: EMode, tag):
    """Decoder-only activation-to-activation for one token and head (Eq. 3).

    K^T tiles (32 keys x dh rows) and V tiles (32 dims x 64 keys) are written
    into the CIM from the cache; q and QK^T live in the input buffer.
    """
    core = sim.core
    L, dh = Kc.shape
    assert dh <= N_ROWS
    unsigned = e_mode == "unsigned"
    acc = np.zeros(dh, dtype=np.int64)
    den = 0
    sim.stage("softmax")
    sim.lut(0, reload=True)
    for c0 in range(0, L, KEY_CHUNK):
        nk = min(KEY_CHUNK, L - c0)
        e = np.zeros(nk, dtype=np.int64)
        for k0 in range(c0, c0 + nk, Q_TILE):
            kt = Kc[k0:k0 + Q_TILE]                      # partitions = keys
            pk = len(kt)
            sim.stage("qkt")
            core.load_next(kt, key=(tag, "K", k0))
            s = core.mac(q, used=(pk, dh))[0, :pk]
            sim.quant(pk)
            z = quantise_vec(s, *rq)
            sim.stage("softmax")
            e[k0 - c0:k0 - c0 + pk] = lut[gm.u8(z)]
            sim.lut(pk)
        sim.ebuf(nk)
        a = e_activation(e, e_mode)
        den += int(a.sum())
        sim.stage("av")
        for d0 in range(0, dh, N_PART):
            vt = Vc[c0:c0 + nk, d0:d0 + N_PART].T
            pd = vt.shape[0]
            core.load_next(vt, key=(tag, "V", c0, d0))
            pv = core.mac(a, used=(pd, nk), unsigned=unsigned)[0, :pd]
            acc[d0:d0 + pd] = acc32(acc[d0:d0 + pd] + pv)
    sim.buf(dh)
    sim.stage("softmax")
    sim.lut(1)
    sim.stage("norm")
    sim.quant(dh)
    sim.zero_rows([den])
    return normalise_rows(acc[None, :], np.array([den]))[0]


def _out_proj(sim: _Sim, cat, qw: _QW):
    if qw.wo is None:
        return cat
    sim.begin("out_proj")
    y = _project(sim, cat, qw.wo, qw.rq["o"], "Wo", qw.tag, stage="out_proj")
    sim.end()
    return y


# ---------------------------------------------------------------------------
# Public layer-level API
# ---------------------------------------------------------------------------
@dataclass
class KVCache:
    """INT8 K and V of the previous tokens, (L, H*dh) each (off-core [A])."""

    k: np.ndarray
    v: np.ndarray

    @staticmethod
    def empty(hd: int) -> "KVCache":
        return KVCache(np.zeros((0, hd), np.int64), np.zeros((0, hd), np.int64))

    def __len__(self):
        return len(self.k)


@dataclass
class LayerResult:
    out: np.ndarray          # INT8 block output (after W_O if present)
    out_scale: float
    scales: Scales
    stats: Stats
    q: np.ndarray            # INT8 projections (N, H*dh)
    k: np.ndarray
    v: np.ndarray
    attn: np.ndarray         # INT8 concatenated head outputs (scale = scales.v)
    cache: KVCache | None = None

    def dequant(self):
        return self.out * self.out_scale


def _encoder_block(sim: _Sim, xq, xkv, qw: _QW, causal: bool, e_mode: EMode):
    """Mapping 1 + 2 (+ W_O) on the core; returns (out, Q, K, V, cat)."""
    sim.begin("proj")
    Q = _project(sim, xq, qw.wq, qw.rq["q"], "Wq", qw.tag)
    K = _project(sim, xkv, qw.wk, qw.rq["k"], "Wk", qw.tag)
    V = _project(sim, xkv, qw.wv, qw.rq["v"], "Wv", qw.tag)
    sim.end()
    sim.begin("attn")
    dh = qw.d_head
    heads = []
    for h in range(qw.n_heads):
        sl = slice(h * dh, (h + 1) * dh)
        heads.append(_attn_encoder_head(sim, Q[:, sl], K[:, sl], V[:, sl], qw.rq["score"][h],
                                        qw.luts[h], causal, e_mode, (sim.tag(), h)))
    sim.end()
    cat = np.concatenate(heads, axis=1)
    return _out_proj(sim, cat, qw), Q, K, V, cat


def _decode_step(sim: _Sim, x_t, qw: _QW, cache: KVCache, e_mode: EMode,
                 kv_fixed: KVCache | None = None):
    """One autoregressive token: project, append to cache, attend, W_O.

    ``kv_fixed`` (cross-attention): K, V come from the encoder, only q is
    projected and the cache is left alone.
    """
    x_t = np.asarray(x_t, dtype=np.int64)[None, :]
    sim.begin("proj")
    q = _project(sim, x_t, qw.wq, qw.rq["q"], "Wq", qw.tag)
    if kv_fixed is None:
        k = _project(sim, x_t, qw.wk, qw.rq["k"], "Wk", qw.tag)
        v = _project(sim, x_t, qw.wv, qw.rq["v"], "Wv", qw.tag)
        cache.k = np.concatenate([cache.k, k])
        cache.v = np.concatenate([cache.v, v])
        kv = cache
    else:
        kv = kv_fixed
    sim.end()
    sim.begin("attn")
    dh = qw.d_head
    heads = []
    for h in range(qw.n_heads):
        sl = slice(h * dh, (h + 1) * dh)
        heads.append(_attn_decode_head(sim, q[0, sl], kv.k[:, sl], kv.v[:, sl],
                                       qw.rq["score"][h], qw.luts[h], e_mode, (sim.tag(), h)))
    sim.end()
    cat = np.concatenate(heads)[None, :]
    y = _out_proj(sim, cat, qw)
    sim.core.stats.tokens += 1
    return y[0], q[0], cat[0]


def encoder_layer(X, w: AttnWeights, scales: Scales | None = None, *, causal: bool = False,
                  e_mode: EMode = "shift1", z_quant_max: int = 127,
                  timing: Timing = Timing(), seed: int = 0) -> LayerResult:
    """Encoder-only mapping (Sec. IV-C) of one multi-head self-attention layer.

    X float (N, d_model). Without ``scales`` they are calibrated from X
    (dynamic, per-tensor); with ``scales`` X is quantised with ``scales.x``.
    ``causal`` gives a decoder prefill on the encoder dataflow.
    """
    sx = scales.x if scales is not None else input_scale(X)
    xq = gm.to_int8(X, sx)
    if scales is None:
        scales = calibrate(xq, sx, w, causal=causal, e_mode=e_mode, z_quant_max=z_quant_max)
    sim = _Sim(timing, seed)
    qw = _quantise_weights(w, scales, z_quant_max, sim.tag())
    out, Q, K, V, cat = _encoder_block(sim, xq, xq, qw, causal, e_mode)
    return LayerResult(out, scales.out, scales, sim.core.stats, Q, K, V, cat,
                       cache=KVCache(K.copy(), V.copy()))


def decoder_layer(X, w: AttnWeights, scales: Scales | None = None, *,
                  cache: KVCache | None = None, e_mode: EMode = "shift1",
                  z_quant_max: int = 127, timing: Timing = Timing(),
                  seed: int = 0) -> LayerResult:
    """Decoder-only mapping (Sec. IV-D): the tokens of X one at a time with a KV cache.

    ``scales`` should be static (the hardware's scale registers); if omitted
    they are calibrated on the whole X with a causal mask, which a real decoder
    could not do (an oracle, for convenience). ``cache`` continues a previous
    context (e.g. a prefill from ``encoder_layer(..., causal=True)``).
    """
    if scales is None:
        sx = input_scale(X)
        scales = calibrate(gm.to_int8(X, sx), sx, w, causal=True, e_mode=e_mode,
                           z_quant_max=z_quant_max)
    xq = gm.to_int8(X, scales.x)
    sim = _Sim(timing, seed)
    qw = _quantise_weights(w, scales, z_quant_max, sim.tag())
    cache = KVCache.empty(w.wq.shape[1]) if cache is None else KVCache(cache.k.copy(), cache.v.copy())
    n0 = len(cache)
    outs, qs, cats = [], [], []
    for t in range(len(xq)):
        y, q, cat = _decode_step(sim, xq[t], qw, cache, e_mode)
        outs.append(y)
        qs.append(q)
        cats.append(cat)
    return LayerResult(np.array(outs), scales.out, scales, sim.core.stats, np.array(qs),
                       cache.k[n0:], cache.v[n0:], np.array(cats), cache=cache)


@dataclass
class EncDecResult:
    enc: LayerResult         # encoder self-attention layer
    out: np.ndarray          # INT8 cross-attention block output per decoder token
    out_scale: float
    self_out: np.ndarray     # INT8 decoder self-attention block output
    scales: dict             # {"enc", "self", "cross"} -> Scales
    stats: dict              # {"enc", "cross_kv", "dec_self", "dec_cross", "total"} -> Stats


def encdec_layer(X_enc, X_dec, enc_w: AttnWeights, self_w: AttnWeights, cross_w: AttnWeights,
                 scales: dict | None = None, *, e_mode: EMode = "shift1",
                 z_quant_max: int = 127, timing: Timing = Timing(),
                 seed: int = 0) -> EncDecResult:
    """Encoder-decoder mapping (Sec. IV-E) on one shared core.

    1. encoder self-attention on X_enc (encoder mapping) -> E (INT8)
    2. cross K, V = E W_k, E W_v projected once (weight-projection flow)
    3. per decoder token: causal self-attention step (decoder mapping), whose
       output is the input of the cross Q projection; cross-attention with the
       encoder K, V written into the CIM (decoder mapping); W_O.
    ``scales`` = {"enc", "self", "cross"} static scales, or None to calibrate
    (decoder side calibrated on the whole X_dec -- an oracle, see decoder_layer).
    """
    zq = z_quant_max
    if scales is None:
        se = input_scale(X_enc)
        s_enc = calibrate(gm.to_int8(X_enc, se), se, enc_w, e_mode=e_mode, z_quant_max=zq)
        sd = input_scale(X_dec)
        xd = gm.to_int8(X_dec, sd)
        s_self = calibrate(xd, sd, self_w, causal=True, e_mode=e_mode, z_quant_max=zq)
        E = _functional_block(gm.to_int8(X_enc, se), gm.to_int8(X_enc, se), enc_w, s_enc,
                              False, e_mode, zq)
        Y1 = _functional_block(xd, xd, self_w, s_self, True, e_mode, zq)
        s_cross = calibrate(Y1, s_self.out, cross_w, E, s_enc.out, e_mode=e_mode, z_quant_max=zq)
        scales = {"enc": s_enc, "self": s_self, "cross": s_cross}
    s_enc, s_self, s_cross = scales["enc"], scales["self"], scales["cross"]
    assert s_cross.x == s_self.out and s_cross.x_kv == s_enc.out

    sim = _Sim(timing, seed)
    st = {k: Stats(timing) for k in ("enc", "cross_kv", "dec_self", "dec_cross")}

    sim.use(st["enc"])
    qe = _quantise_weights(enc_w, s_enc, zq, sim.tag())
    xe = gm.to_int8(X_enc, s_enc.x)
    E, Qe, Ke, Ve, cate = _encoder_block(sim, xe, xe, qe, False, e_mode)
    enc_res = LayerResult(E, s_enc.out, s_enc, st["enc"], Qe, Ke, Ve, cate)

    sim.use(st["cross_kv"])
    qc = _quantise_weights(cross_w, s_cross, zq, sim.tag())
    sim.begin("proj")
    Kc = _project(sim, E, qc.wk, qc.rq["k"], "Wk", qc.tag)
    Vc = _project(sim, E, qc.wv, qc.rq["v"], "Wv", qc.tag)
    sim.end()
    cross_kv = KVCache(Kc, Vc)

    qs = _quantise_weights(self_w, s_self, zq, sim.tag())
    xd = gm.to_int8(X_dec, s_self.x)
    cache = KVCache.empty(self_w.wq.shape[1])
    y1s, ys = [], []
    for t in range(len(xd)):
        sim.use(st["dec_self"])
        y1, _, _ = _decode_step(sim, xd[t], qs, cache, e_mode)
        sim.use(st["dec_cross"])
        y, _, _ = _decode_step(sim, y1, qc, cache, e_mode, kv_fixed=cross_kv)
        y1s.append(y1)
        ys.append(y)
    total = Stats(timing)
    for s in st.values():
        total.merge(s)
    total.tokens = len(xd)
    st["total"] = total
    return EncDecResult(enc_res, np.array(ys), s_cross.out, np.array(y1s), scales, st)


def _functional_block(xq, xkv, w: AttnWeights, sc: Scales, causal: bool, e_mode: EMode,
                      z_quant_max: int):
    """Plain integer forward of one block with given scales (used to calibrate
    a block whose input is another block's output)."""
    qw = _quantise_weights(w, sc, z_quant_max, -1)
    Q = quantise_vec(xq @ qw.wq, *qw.rq["q"])
    K = quantise_vec(xkv @ qw.wk, *qw.rq["k"])
    V = quantise_vec(xkv @ qw.wv, *qw.rq["v"])
    dh, heads = qw.d_head, []
    for h in range(qw.n_heads):
        sl = slice(h * dh, (h + 1) * dh)
        z = quantise_vec(Q[:, sl] @ K[:, sl].T, *qw.rq["score"][h])
        e = qw.luts[h][gm.u8(z)]
        if causal:
            e[_causal_mask(len(Q), len(K))] = 0
        a = e_activation(e, e_mode)
        heads.append(normalise_rows(a @ V[:, sl], a.sum(1)))
    cat = np.concatenate(heads, axis=1)
    return cat if qw.wo is None else quantise_vec(cat @ qw.wo, *qw.rq["o"])


# ---------------------------------------------------------------------------
# CLI: python -m cimple_sw.mapping
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cimple-sw map", description="CIMple mapping simulator (Sec. IV-C/D/E)")
    ap.add_argument("--n", type=int, default=128, help="sequence length / context")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dec-tokens", type=int, default=16, help="decoder tokens (enc-dec)")
    ap.add_argument("--e-mode", choices=["shift1", "unsigned"], default="shift1")
    ap.add_argument("--overlap-out", action="store_true", help="Timing.overlap_out [A]")
    ap.add_argument("--freq", type=float, default=400.0,
                    help="core clock in MHz (400: latency study; Table I range 417-770)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    N, dm, H = a.n, a.d_model, a.heads
    dh = dm // H
    tm = Timing(overlap_out=a.overlap_out, freq_mhz=a.freq)
    w = AttnWeights.random(dm, H, dh, rng)
    X = rng.standard_normal((N, dm))

    print(f"CIMple mapping simulator: d_model={dm}, heads={H}, d_head={dh}, "
          f"pass={tm.pass_cycles} cycles, e_mode={a.e_mode}\n")
    enc = encoder_layer(X, w, e_mode=a.e_mode, timing=tm)
    print(enc.stats.table(f"== encoder-only layer, N={N} (Sec. IV-C) =="))

    pre = encoder_layer(X[:N - 1], w, enc.scales, causal=True, e_mode=a.e_mode, timing=tm)
    dec = decoder_layer(X[N - 1:], w, enc.scales, cache=pre.cache, e_mode=a.e_mode, timing=tm)
    print()
    print(dec.stats.table(f"== decoder-only: one decode step at context {N} (Sec. IV-D) =="))

    wd = AttnWeights.random(dm, H, dh, rng)
    wc = AttnWeights.random(dm, H, dh, rng)
    Xd = rng.standard_normal((a.dec_tokens, dm))
    ed = encdec_layer(X, Xd, w, wd, wc, e_mode=a.e_mode, timing=tm)
    print()
    for k in ("enc", "cross_kv", "dec_self", "dec_cross", "total"):
        title = (f"== encoder-decoder (Sec. IV-E), enc N={N}, {a.dec_tokens} decoder tokens: "
                 f"{k} ==")
        print(ed.stats[k].table(title))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
