"""Cycle-level latency model of CIMple's activation-to-activation stage.

Reproduces the latency study of Ahn et al., arXiv:2604.15944v2, Sec. V-B:
"the overall latency with the LUT-based split softmax is 33% shorter than a
non-split LUT" (encoder-only mapping, head dim 64, 1024 tokens, 400 MHz,
baseline softmax input 32 b). The paper gives no schedule, so this module
builds one from the architecture (Fig. 4, Sec. IV-A/B/C) and reports what it
yields. Nothing here is fitted to 33 %; ``main`` prints which documented knob
settings land near it.

Model
-----
An in-order list scheduler over explicit tasks. Every task holds a set of
resources for ``dur`` cycles and its results are visible ``tail`` cycles later
(pipeline drain). A task starts at max(dependencies done, resources free).

Resources that are scheduled:
  core      the CIM core (one bank active at a time [P] Fig. 6)
  bank0/1   the two SRAM blocks: a core job holds the bank it reads, a load
            holds the bank it writes, so "write the inactive bank while the
            other computes" [P] "supports simultaneous read and write" falls
            out, and writing the active bank does not
  wport     the 128 b write port: a 32 x 64 INT8 tile = 128 cycles [D]
  smx       the baseline's 32 b softmax unit (max / exp-sum / normalise)
  recip     the reciprocal (M) LUT of the split softmax
Downstream of the core (output mux -> quant -> exp LUT -> ACC buffer) every
unit moves 4 values per cycle [F] (CIM OUT 23b x 4, quant 4 x 32b -> 4 x 8b,
LUT 4 x 8b, ACC 4 x 32b), i.e. exactly the rate of the 32->4 mux. With a single
source (the core) they can never stall it, so they are not scheduled; they add
their latency to the job's tail and their occupancy to ``busy`` [D]. If a
downstream unit is made slower (Timing.lut_cycles > 1, ...) the core is
throttled to it (backpressure) [A]: the mux walk takes 8 x (slowest unit's
cycles per group), and a pass takes mac + walk with Timing.overlap_out False
(no shadow register, as in the RTL) or max(mac, walk) with it True.

Mapping (Sec. IV-C, [P] "Q and V buffered in SRAM, K^T streamed as input",
"storing V and Q in the CIM and feeding K^T as input"):
  QK^T  weights = Q tile, 32 queries (partitions) x 64 dims (rows): one bank.
        Activation = one key k_j (64 dims). One pass -> 32 scores (column j).
        d > 64 -> ceil(d/64) Q row tiles, partials summed in the ACC buffer.
  A'V   weights = V^T tile, 32 head dims (partitions) x 64 tokens (rows).
        Activation = e[q, 64-token chunk] (8 b). One pass -> 32 partial outputs,
        accumulated across the N/64 chunks in the ACC buffer (32 b).
  Per Q tile: N passes (x ceil(d/64)) for QK^T and (N/64) * ceil(d/32) * 32
  passes for A'V; N = 1024, d = 64 -> 32 Q tiles x (1024 + 1024) passes.
  [A] Sec. IV-A says "the output of the softmax is sent ... to the write data
  for encoder-only computations" (Fig. 4: "Attention score" -> write data
  select -> WBL 128b), which could be read as writing e / A' into the SRAM.
  We read that output as the normalised attention score (A'V * M; the paper's
  term for the per-head result, Sec. II / IV-B) written back for the W_O stage,
  and keep V^T stationary with e on the input driver, as Sec. IV-C states
  twice. Under the other reading every (Q tile, chunk) would also pay a
  32 x 64 INT8 e-tile write (128 wport cycles into the inactive bank; 65 536
  write cycles at N = 1024); this model does not schedule that variant.

Split (Fig. 7): per 64-key chunk, QK^T -> quant 8 b (scores, [P] 32b -> 8b
before the softmax) -> exp LUT -> e buffer (+ exp-sum registers) -> A'V
partials. z_quant_max replaces the max pass, so no buffer re-read. At the end
of a row the reciprocal LUT gives M; the last chunk's A'V output goes
ACC add -> quant with scale M (golden model ``softmax_row``: quantise(partial,
M, shift)), so the final multiply costs no extra pass [A].

Baseline ("non split LUT softmax, 32 b input" [P]): QK^T of a whole Q tile
over all N keys first, 32 b scores written to the buffer; then "the input is
read three times" [P] (max; exp + sum; divide) at 4 x 32 b per cycle [F] (the
buffer's read select); then A'V on the normalised 8 b A'. [P] "the softmax
computation of QK^T begins only after all QK^T computations are completed".

To first order (P = core cycles per pass, defaults 16; per Q tile):
  core C = N * P * ceil(d/64) + N * ceil(d/32) * P / 2,
  softmax S = sm_reads * 32 * N / sm_lanes,
  reduction ~= S / (C + S)    (d = 64, P = 16: 24N / 56N = 42.9 %).

What the model yields (``python -m cimple_sw.latency``)
-------------------------------------------------------
Defaults, N = 1024, d = 64, 400 MHz: baseline 1 835 362 cycles (4.59 ms),
split 1 049 730 cycles (2.62 ms) -> 42.8 % shorter, not 33 %. The split
schedule is core-bound (every V / Q reload hidden, ~1 k cycles of drain), so
the number is set almost entirely by the baseline softmax cost S relative to
the core work C. Single documented changes that give ~33 %: sm_reads=2 (max
tracked while the scores are written: 33.3 %) and baseline_stream_norm (A'V
consumes A' as it is normalised: 34.0 %). Two-change combinations also land
there: overlap_out + sm_reads=1 (32.7 %) and d = 128 with bank_policy="fixed"
(33.3 %). If the baseline instead overlaps
its softmax with the next Q tile's QK^T (baseline_tile_overlap), both are
core-bound and the advantage disappears (1.4 %). The claim therefore pins
down the baseline only to within these choices; the paper does not say which.
"""
from __future__ import annotations

import argparse
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field, fields, replace

from cimple_sw import hw
from cimple_sw.hw import Timing

ROWS = hw.N_ROWS               # 64: rows per bank (input vector length)
COLS = hw.N_PART               # 32: partitions = outputs per pass
LANES = hw.N_LANES             # 4: values per cycle downstream of the mux
TILE_WRITE = Timing.write_cycles(ROWS * COLS)   # [D] 2048 weights / 16 = 128


# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Knobs:
    """Every schedule choice the paper does not pin down.

    acc_cycles          [A] 1: ACC buffer read-modify-write latency per 4-lane
                        group (and its throughput: 4 x 32b per cycle [F]).
    bank_policy         [A] "pingpong" (default): a tile that is not resident
                        goes to the bank the previous core job did not use, so
                        its load hides behind that job; a resident tile is
                        reused. With interleave the split schedule then keeps
                        Q in one bank and streams V tiles through the other;
                        the baseline, whose Q tile is idle during A'V,
                        alternates V tiles over both. Without interleave and
                        ceil(d/32) > 1 the split's second V tile of a chunk
                        evicts Q, so Q is reloaded every chunk (N=1024, d=64:
                        512 Q writes instead of 32, +61 440 wport cycles);
                        latency is unchanged because those reloads hide too.
                        "fixed": bank0 only ever holds Q, bank1 only V (the
                        literal reading of Sec. IV-C) for both variants.
    interleave          [A] True: the QK^T of chunk c+1 is cut into ceil(d/32)
                        segments placed between the A'V column tiles of chunk
                        c, so a V tile reload always has a Q-bank job to hide
                        behind (needs a double-buffered e buffer, 2 x 2 kB).
                        Matters for latency with bank_policy="fixed"; with
                        "pingpong" the V tiles alternate banks and hide
                        anyway, at the cost of the Q reloads noted above.
    write_while_compute [P]-based True: a load may proceed into the inactive
                        bank while the core computes. False: loads also hold
                        the core.
    bank_switch_cycles  [A] 0: the bank select is the OAI's second input, a
                        static signal; no switch penalty is described.
    recip_cycles        [A] 1 cycle per 4 lookups of the reciprocal LUT
                        (same 4-lane width as the exp LUT).
    sm_reads            [P] 3: "conventional safe softmax ... read the input
                        value three times". 2 = max tracked on the fly while the
                        scores are written; 1 = exp-sum also on the fly.
    sm_lanes            [F] 4: 32 b values through the 4 x 32 b buffer read
                        select per cycle.
    sm_cycles           [A] 1 cycle per sm_lanes group for each pass (a fully
                        pipelined comparator / exp / divider; generous to the
                        baseline -- a real 32 b divider is slower).
    sm_latency          [A] 2: pipeline depth of one baseline softmax pass
                        (buffer read + operate).
    baseline_stream_norm  [A] False: the normalise pass finishes before A'V
                        starts. True: it runs chunk-major and A'V of chunk c
                        starts as soon as its A' is normalised (assumes the
                        score buffer and the ACC buffer have separate ports).
    baseline_tile_overlap [A] False ([P] the baseline is sequential). True:
                        the core computes QK^T of Q tile t+1 while the
                        softmax unit works on tile t (needs 2 score buffers).
    """

    acc_cycles: int = 1
    bank_policy: str = "pingpong"
    interleave: bool = True
    write_while_compute: bool = True
    bank_switch_cycles: int = 0
    recip_cycles: int = 1
    sm_reads: int = 3
    sm_lanes: int = LANES
    sm_cycles: int = 1
    sm_latency: int = 2
    baseline_stream_norm: bool = False
    baseline_tile_overlap: bool = False

    def __post_init__(self):
        if self.bank_policy not in ("fixed", "pingpong"):
            raise ValueError(f"bank_policy {self.bank_policy!r}")
        if self.sm_reads not in (1, 2, 3):
            raise ValueError("sm_reads must be 1, 2 or 3")
        if self.sm_lanes < 1 or self.acc_cycles < 1 or self.sm_cycles < 1:
            raise ValueError("lanes / cycles must be >= 1")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
@dataclass
class Task:
    name: str
    stage: str                 # breakdown label
    res: tuple[str, ...]       # scheduled resources held for [start, end)
    start: int
    end: int
    done: int                  # end + tail: outputs visible downstream
    side: dict = field(default_factory=dict)   # rate-matched units: busy cycles
    # loads only: the [a, b) part that held up the core (None = fully hidden)
    exposed: tuple[int, int] | None = None


class _Sched:
    def __init__(self):
        self.free: dict[str, int] = defaultdict(int)
        self.tasks: list[Task] = []

    def add(self, name, stage, res, dur, tail=0, after=0, side=None) -> Task:
        start = max([after] + [self.free[r] for r in res])
        end = start + dur
        for r in res:
            self.free[r] = end
        t = Task(name, stage, tuple(res), start, end, end + tail, side or {})
        self.tasks.append(t)
        return t


class _Core:
    """Core-job emitter: bank placement, tile loads, pass timing."""

    def __init__(self, s: _Sched, tm: Timing, k: Knobs):
        self.s, self.tm, self.k = s, tm, k
        self.bank_tile: list = [None, None]
        self.prev_bank: int | None = None

    def _place(self, tile) -> tuple[int, Task | None]:
        if tile in self.bank_tile:
            return self.bank_tile.index(tile), None
        if self.k.bank_policy == "fixed":
            b = 0 if tile[0] == "Q" else 1
        else:
            b = 0 if self.prev_bank is None else 1 - self.prev_bank
        res = ("wport", f"bank{b}") + (() if self.k.write_while_compute else ("core",))
        ld = self.s.add(f"load {tile}", f"load_{tile[0].lower()}", res, TILE_WRITE,
                        side={"wport": TILE_WRITE})
        self.bank_tile[b] = tile
        return b, ld

    def job(self, name, stage, tile, passes, path, after=0) -> Task:
        """``passes`` activation vectors through the tile; ``path`` = list of
        (downstream unit, cycles per 4-lane group) the 32 results traverse."""
        tm = self.tm
        b, ld = self._place(tile)
        core_free = self.s.free["core"]
        group = max([c for _, c in path] + [1])
        # backpressure: the mux walk runs at the slowest downstream unit's rate
        walk = max(tm.out_cycles, hw.MUX_PHASES * group)
        # overlap_out=False: no shadow register, the walk still serialises
        # with the next MAC even when it is throttled
        period = max(tm.mac_cycles, walk) if tm.overlap_out else tm.mac_cycles + walk
        drain = tm.mac_cycles + walk - period    # last walk past the job end
        sw = self.k.bank_switch_cycles if self.prev_bank not in (None, b) else 0
        side = {u: passes * hw.MUX_PHASES * c for u, c in path}
        side["mux"] = passes * hw.MUX_PHASES
        t = self.s.add(name, stage, ("core", f"bank{b}"), sw + passes * period,
                       tail=drain + sum(c for _, c in path), after=after, side=side)
        if ld is not None:
            if "core" in ld.res:              # write_while_compute=False
                ld.exposed = (ld.start, ld.end)
            elif t.start == ld.end and ld.end > core_free:   # load gates the job
                ld.exposed = (max(ld.start, core_free), ld.end)
        self.prev_bank = b
        return t


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------
def _geometry(N, d):
    n_qt = math.ceil(N / COLS)
    n_ch = math.ceil(N / ROWS)
    q_rows = [min(COLS, N - t * COLS) for t in range(n_qt)]
    keys = [min(ROWS, N - c * ROWS) for c in range(n_ch)]
    return n_qt, n_ch, q_rows, keys, math.ceil(d / ROWS), math.ceil(d / COLS)


def _segments(units, nseg):
    """Cut a list of (r, k0, k1) pass ranges into ``nseg`` near-equal
    consecutive segments (by passes), keeping order; empty ones dropped."""
    total = sum(k1 - k0 for _, k0, k1 in units)
    bounds = [round(i * total / nseg) for i in range(nseg + 1)]
    flat, pos = [], 0
    for r, k0, k1 in units:
        flat.append((pos, pos + k1 - k0, r, k0))
        pos += k1 - k0
    segs = []
    for i in range(nseg):
        lo, hi = bounds[i], bounds[i + 1]
        seg = []
        for p0, p1, r, k0 in flat:
            a, b = max(lo, p0), min(hi, p1)
            if a < b:
                seg.append((r, k0 + a - p0, k0 + b - p0))
        segs.append(seg)
    return segs


def _split(N, d, tm, k):
    s = _Sched()
    core = _Core(s, tm, k)
    n_qt, n_ch, q_rows, keys, n_qr, n_ct = _geometry(N, d)
    flat = [(t, c) for t in range(n_qt) for c in range(n_ch)]
    e_done: dict = {}
    recip_done: dict = {}

    def qk(seg, t, c):
        for r, k0, k1 in seg:
            last_r = r == n_qr - 1
            if last_r:   # (ACC add of the partial) -> quant 8b -> exp LUT
                path = ([("acc", k.acc_cycles)] if n_qr > 1 else []) + \
                       [("quant", tm.quant_cycles), ("lut_exp", tm.lut_cycles)]
            else:        # partial score -> ACC buffer
                path = [("acc", k.acc_cycles)]
            j = core.job(f"QK t{t} c{c} r{r} k{k0}:{k1}", "qk", ("Q", t, r),
                         k1 - k0, path)
            if last_r:
                e_done[(t, c)] = max(e_done.get((t, c), 0), j.done)
                # every exp-sum of tile t is complete after its last key
                if c == n_ch - 1 and k1 == keys[c]:
                    rc = s.add(f"recip t{t}", "recip", ("recip",),
                               math.ceil(q_rows[t] / LANES) * k.recip_cycles,
                               after=e_done[(t, c)])
                    recip_done[t] = rc.done

    def units(t, c):
        return [(r, 0, keys[c]) for r in range(n_qr)]

    qk(units(*flat[0]), *flat[0])
    for i, (t, c) in enumerate(flat):
        nxt = flat[i + 1] if i + 1 < len(flat) else None
        segs = _segments(units(*nxt), n_ct) if (nxt and k.interleave) else []
        final = c == n_ch - 1
        for ct in range(n_ct):
            after = e_done[(t, c)]
            path = [("acc", k.acc_cycles)]
            if final:   # ACC add -> quant with scale M (fused final multiply)
                path.append(("quant", tm.quant_cycles))
                # M is needed when the first result reaches the quantiser
                after = max(after, recip_done[t] - tm.mac_cycles - k.acc_cycles)
            core.job(f"AV t{t} c{c} v{ct}", "av", ("V", c, ct), q_rows[t], path, after)
            if segs:
                qk(segs[ct], *nxt)
        if nxt and not k.interleave:
            qk(units(*nxt), *nxt)
    return s


_SM_LABELS = {3: ("sm_max", "sm_expsum", "sm_norm"), 2: ("sm_expsum", "sm_norm"),
              1: ("sm_norm",)}


def _baseline(N, d, tm, k):
    s = _Sched()
    core = _Core(s, tm, k)
    n_qt, n_ch, q_rows, keys, n_qr, n_ct = _geometry(N, d)

    def sm_pass(label, t, n_vals, after, tag=""):
        return s.add(f"{label} t{t}{tag}", label, ("smx",),
                     math.ceil(n_vals / k.sm_lanes) * k.sm_cycles,
                     tail=k.sm_latency, after=after,
                     side={"buf_rd": math.ceil(n_vals / LANES)})

    def qk_tile(t):
        j = None
        for r in range(n_qr):   # 32b scores (r > 0: RMW) into the score buffer
            j = core.job(f"QK t{t} r{r}", "qk", ("Q", t, r), N,
                         [("acc", k.acc_cycles)])
        return j.done

    def sm_and_av(t, qk_done):
        labels = _SM_LABELS[k.sm_reads]
        done = qk_done
        for lab in labels[:-1]:
            done = sm_pass(lab, t, q_rows[t] * N, done).done
        for c in range(n_ch):
            if k.baseline_stream_norm:
                norm = sm_pass(labels[-1], t, q_rows[t] * keys[c], done, f" c{c}").done
            elif c == 0:
                norm = sm_pass(labels[-1], t, q_rows[t] * N, done).done
            final = c == n_ch - 1
            path = [("acc", k.acc_cycles)] + ([("quant", tm.quant_cycles)] if final else [])
            for ct in range(n_ct):
                core.job(f"AV t{t} c{c} v{ct}", "av", ("V", c, ct), q_rows[t],
                         path, after=norm)

    if k.baseline_tile_overlap:
        pending = qk_tile(0)
        for t in range(n_qt):
            nxt = qk_tile(t + 1) if t + 1 < n_qt else None
            sm_and_av(t, pending)
            pending = nxt
    else:
        for t in range(n_qt):
            sm_and_av(t, qk_tile(t))
    return s


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
STAGES = ("qk", "av", "sm_max", "sm_expsum", "sm_norm", "recip",
          "load_q", "load_v", "drain", "idle")
"""Breakdown labels, highest priority first: a cycle in which several things
happen is charged to the first one listed, so the breakdown sums to the total.
A load counts only over the part that held up the core (its consumer started
when it ended, or it held the core itself); a load running under a core stall
that something else caused (e.g. waiting for e) is hidden and that cycle goes
to the stall's own stage (drain, recip, ...)."""


@dataclass
class LatencyResult:
    variant: str
    N: int
    d: int
    cycles: int
    freq_mhz: float
    breakdown: dict[str, int]      # exposed cycles per stage; sums to cycles
    busy: dict[str, int]           # per-unit busy cycles (may overlap)
    buffer_bytes: dict[str, int]   # on-chip storage the schedule needs
    knobs: Knobs
    timing: Timing
    tasks: list[Task] = field(repr=False, default_factory=list)

    @property
    def us(self) -> float:
        return self.cycles / self.freq_mhz

    @property
    def ops(self) -> int:
        """QK^T + A'V, 1 MAC = 2 OP: 2*N*N*d each [D]."""
        return 4 * self.N * self.N * self.d

    @property
    def tops(self) -> float:
        return self.ops / (self.us * 1e-6) / 1e12

    @property
    def core_util(self) -> float:
        """Fraction of the 512 OP/cycle peak actually delivered."""
        return self.ops / (self.cycles * hw.PEAK_OPS_PER_CYCLE)


def _breakdown(tasks, total) -> dict[str, int]:
    prio = {s: i for i, s in enumerate(STAGES)}
    ev = []
    for t in tasks:
        a, b = (t.exposed or (0, 0)) if t.stage.startswith("load") else (t.start, t.end)
        if b > a:
            ev += [(a, prio[t.stage], 1), (b, prio[t.stage], -1)]
        if t.done > t.end:
            ev += [(t.end, prio["drain"], 1), (t.done, prio["drain"], -1)]
    ev.sort()
    cnt = [0] * len(STAGES)
    out = dict.fromkeys(STAGES, 0)
    prev, i = 0, 0
    while i < len(ev):
        x = ev[i][0]
        if x > prev:
            top = next((j for j, n in enumerate(cnt) if n), prio["idle"])
            out[STAGES[top]] += x - prev
            prev = x
        while i < len(ev) and ev[i][0] == x:
            cnt[ev[i][1]] += ev[i][2]
            i += 1
    if total > prev:
        out["idle"] += total - prev
    return out


def _buffers(variant, N, d, k) -> dict[str, int]:
    n_qr = math.ceil(d / ROWS)
    acc = COLS * d * hw.ACC_W // 8                  # A'V partials of one Q tile
    if variant == "split":
        b = {"acc_partials": acc, "e_buffer": (2 if k.interleave else 1) * COLS * ROWS,
             "exp_sums": COLS * 3}                  # 24b sums [A] golden SUM_W
        if n_qr > 1:
            b["score_partials"] = COLS * ROWS * hw.ACC_W // 8
        return b
    nbuf = 2 if k.baseline_tile_overlap else 1
    return {"acc_partials": acc, "score_buffer": nbuf * COLS * N * hw.ACC_W // 8,
            "a_prime": COLS * N}


def latency(N: int = 1024, d: int = 64, timing: Timing = Timing(),
            variant: str = "split", **knobs) -> LatencyResult:
    """Schedule one attention head's activation-to-activation stage."""
    if N < 1 or d < 1:
        raise ValueError("N, d must be >= 1")
    k = Knobs(**knobs)
    if variant == "split":
        s = _split(N, d, timing, k)
    elif variant == "baseline":
        s = _baseline(N, d, timing, k)
    else:
        raise ValueError(f"variant {variant!r}")
    total = max(t.done for t in s.tasks)
    busy: dict[str, int] = defaultdict(int)
    for t in s.tasks:
        for r in t.res:
            if not r.startswith("bank"):
                busy[r] += t.end - t.start
        for u, c in t.side.items():
            if u != "wport":
                busy[u] += c
    return LatencyResult(variant, N, d, total, timing.freq_mhz,
                         _breakdown(s.tasks, total), dict(busy),
                         _buffers(variant, N, d, k), k, timing, s.tasks)


def compare(N: int = 1024, d: int = 64, timing: Timing = Timing(), **knobs) -> dict:
    """Both variants under the same knobs, and the latency reduction (%)."""
    sp = latency(N, d, timing, "split", **knobs)
    bl = latency(N, d, timing, "baseline", **knobs)
    return {"split": sp, "baseline": bl,
            "reduction_pct": 100.0 * (1 - sp.cycles / bl.cycles)}


SWEEP_N = (64, 128, 256, 512, 1024, 2048)
SWEEP_D = (32, 64, 128)


def sweep(Ns=SWEEP_N, ds=SWEEP_D, overlaps=(False, True), knob_sets=({},),
          timing: Timing = Timing()) -> list[dict]:
    """Reduction over a grid of N, d, Timing.overlap_out and knob dicts."""
    rows = []
    for kn in knob_sets:
        for ov in overlaps:
            tm = replace(timing, overlap_out=ov)
            for N in Ns:
                for d in ds:
                    r = compare(N, d, tm, **kn)
                    rows.append(dict(N=N, d=d, overlap_out=ov, knobs=dict(kn),
                                     split=r["split"].cycles,
                                     baseline=r["baseline"].cycles,
                                     reduction_pct=r["reduction_pct"]))
    return rows


# Knob settings shown in the CLI's sensitivity table (one change at a time,
# then the combinations that move the headline most).
SENSITIVITY: tuple[tuple[str, dict, dict], ...] = (
    ("defaults", {}, {}),
    ("overlap_out", {"overlap_out": True}, {}),
    ("sm_reads=2 (max on the fly)", {}, {"sm_reads": 2}),
    ("sm_reads=1", {}, {"sm_reads": 1}),
    ("sm_lanes=8", {}, {"sm_lanes": 8}),
    ("sm_lanes=16 (8b-wide port)", {}, {"sm_lanes": 16}),
    ("sm_cycles=2 (slow divider)", {}, {"sm_cycles": 2}),
    ("baseline_stream_norm", {}, {"baseline_stream_norm": True}),
    ("baseline_tile_overlap", {}, {"baseline_tile_overlap": True}),
    ("interleave=False", {}, {"interleave": False}),
    ("bank_policy=fixed", {}, {"bank_policy": "fixed"}),
    ("write_while_compute=False", {}, {"write_while_compute": False}),
    ("bank_switch_cycles=2", {}, {"bank_switch_cycles": 2}),
    ("lut_cycles=2", {"lut_cycles": 2}, {}),
    ("overlap_out + sm_reads=1", {"overlap_out": True}, {"sm_reads": 1}),
    ("overlap_out + stream_norm", {"overlap_out": True}, {"baseline_stream_norm": True}),
    ("stream_norm + tile_overlap", {}, {"baseline_stream_norm": True,
                                       "baseline_tile_overlap": True}),
)


def near_paper(N=1024, timing: Timing = Timing(), base_knobs=None,
               lo=30.0, hi=36.0) -> list[dict]:
    """Grid-search the documented knobs and return the settings whose
    reduction lands in [lo, hi] %. This is a disclosure of how much freedom
    the unspecified baseline leaves, not a fit: the defaults are not changed."""
    out = []
    grid = itertools.product((32, 64, 128), (False, True), (3, 2, 1), (4, 8),
                             (False, True), ("pingpong", "fixed"))
    for d, ov, reads, lanes, stream, pol in grid:
        kn = {**(base_knobs or {}), "sm_reads": reads, "sm_lanes": lanes,
              "baseline_stream_norm": stream, "bank_policy": pol}
        r = compare(N, d, replace(timing, overlap_out=ov), **kn)["reduction_pct"]
        if lo <= r <= hi:
            out.append(dict(d=d, reduction_pct=r, settings={
                "overlap_out": ov, "sm_reads": reads, "sm_lanes": lanes,
                "stream_norm": stream, "bank_policy": pol}))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _fmt_breakdown(r: LatencyResult) -> str:
    return "  ".join(f"{s}={v}" for s, v in r.breakdown.items() if v)


def _parse_knob(kv: str):
    key, _, val = kv.partition("=")
    names = {f.name: f.type for f in fields(Knobs)}
    if key not in names:
        raise SystemExit(f"unknown knob {key!r}; known: {', '.join(names)}")
    if val.lower() in ("true", "false"):
        return key, val.lower() == "true"
    try:
        return key, int(val)
    except ValueError:
        return key, val


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cimple-sw latency",
                                 description="CIMple split vs non-split softmax latency")
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--overlap-out", action="store_true")
    ap.add_argument("--freq", type=float, default=400.0)
    ap.add_argument("--knob", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--no-sweep", action="store_true")
    a = ap.parse_args(argv)
    tm = Timing(overlap_out=a.overlap_out, freq_mhz=a.freq)
    kn = dict(_parse_knob(x) for x in a.knob)

    r = compare(a.N, a.d, tm, **kn)
    print(f"CIMple activation-to-activation latency, N={a.N} d={a.d} "
          f"@ {tm.freq_mhz:g} MHz, overlap_out={tm.overlap_out}, knobs={kn or 'defaults'}")
    print(f"{'variant':9} {'cycles':>10} {'us':>9} {'GOPS':>8} {'core util':>9}  breakdown")
    for v in ("baseline", "split"):
        x = r[v]
        print(f"{v:9} {x.cycles:10d} {x.us:9.2f} {x.tops * 1e3:8.1f} "
              f"{x.core_util:9.1%}  {_fmt_breakdown(x)}")
    sp = r["split"]
    print(f"ops (QK^T + A'V) = {sp.ops:,}  peak = {hw.PEAK_OPS_PER_CYCLE} OP/cycle "
          f"= {hw.PEAK_OPS_PER_CYCLE * tm.freq_mhz / 1e6:.3f} TOPS")
    print(f"latency reduction (split vs baseline) = {r['reduction_pct']:.1f} %   "
          f"[paper Sec. V-B: 33 %]")
    print("buffers: baseline", r["baseline"].buffer_bytes, " split", sp.buffer_bytes)
    if a.no_sweep:
        return 0

    print(f"\nSensitivity at N={a.N} d={a.d} (one change from the defaults unless "
          "combined); * = within 30-36 %")
    print(f"{'setting':32} {'baseline':>10} {'split':>10} {'reduction':>9}")
    for name, tkw, kkw in SENSITIVITY:
        x = compare(a.N, a.d, replace(tm, **tkw), **{**kn, **kkw})
        red = x["reduction_pct"]
        mark = "*" if 30 <= red <= 36 else " "
        print(f"{name:32} {x['baseline'].cycles:10d} {x['split'].cycles:10d} "
              f"{red:8.1f}%{mark}")

    print(f"\nKnob combinations within 30-36 % at N={a.N} (grid: d, overlap_out, "
          "sm_reads, sm_lanes, baseline_stream_norm, bank_policy)")
    for row in near_paper(a.N, tm, kn):
        print(f"  {row['reduction_pct']:5.1f}%  d={row['d']:<4} " +
              " ".join(f"{k}={v}" for k, v in row["settings"].items()))

    print("\nReduction (%) over N x d, defaults otherwise")
    for ov in (False, True):
        rows = sweep(SWEEP_N, SWEEP_D, (ov,), (kn,), tm)
        print(f"  overlap_out={ov}")
        print("  " + f"{'N':>6}" + "".join(f"{'d=' + str(d):>9}" for d in SWEEP_D))
        for N in SWEEP_N:
            vals = [x["reduction_pct"] for x in rows if x["N"] == N]
            print("  " + f"{N:6d}" + "".join(f"{v:9.1f}" for v in vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
