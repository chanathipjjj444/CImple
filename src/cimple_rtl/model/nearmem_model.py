"""
Near-memory schedule model: the CIMple core as a Versal PL accelerator.

Builds on cimple_model (the single-pass, bit-exact datapath) and adds what a
real layer needs: tiling, the two-bank array as a double buffer, an UltraRAM
weight store, and the acc_buffer used the way the RTL implements it. Every
number that comes out is still exact integer arithmetic; cycles are counted
alongside so the schedule itself can be checked.

Target: AMD Versal Premium VP1202 (BoostX HHHL card).

DECISIONS  [A] -- the paper does not say; these are this model's choices:
  1. BANK POLICY: double buffering. The active bank computes while the shadow
     bank is filled with the NEXT weight tile from UltraRAM; they swap at the
     job boundary. The loader never writes the active bank (asserted).
  2. LOOP ORDER: for each 32-column output tile, for each block of tokens, for
     each 64-row K tile. A block is as many tokens as the acc_buffer holds:
     DEPTH / MUX_PHASES = 64 / 8 = 8 tokens.
  3. PADDING: the loader only writes the used rows/partitions of a tile; the
     rest of the shadow bank keeps STALE weights from an older tile. This is
     safe because unused rows see zero activations and unused partitions are
     never read. The model fills the banks with garbage to prove it.
  4. A'V ACTIVATIONS: the core reads activations as SIGNED INT8, but the exp
     LUT gives UQ0.8 (0..255). The exp values are fed as e >> 1 (UQ0.7). The
     sum is taken over the same e >> 1, so the softmax ratio is unchanged and
     only one bit of numerator precision is lost. (Alternative: a 1-bit
     "unsigned activation" mode in cim_partition that adds instead of
     subtracting on the sign cycle. That is an RTL change; this is not.)

TIMING (from the RTL, in clock cycles):
  * one activation vector = 8 bit-serial MAC cycles + 8 output phases. The
    partition accumulator is read combinationally while the 3b counter walks,
    so the next pass cannot start until all 8 phases are out.
  * a tile load writes LOAD_WPC weights per cycle (16 = the paper's 128b WBL).
    The current RTL write port is 8b, i.e. LOAD_WPC = 1 -- widen it to 128b to
    get the numbers below.
  * after a block's last K tile, its DEPTH entries are drained to the quant
    unit at one entry (4 lanes) per cycle. Counted serially here; overlapping
    it with the next block is a later optimisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import cimple_model as m

# ---- VP1202 (from the BoostX resource table) -------------------------------
URAM_BLOCKS = 677
URAM_BITS = URAM_BLOCKS * 288 * 1024          # ~190 Mb
DSP_PEAK_INT8_TOPS = 27.5

ACC_MIN, ACC_MAX = -(1 << (m.ACC_W - 1)), (1 << (m.ACC_W - 1)) - 1


@dataclass
class Config:
    depth: int = 64            # acc_buffer entries (4 lanes each)
    load_wpc: int = 16         # weights written per cycle (128b / 8b)
    mac_cycles: int = m.DATA_W
    out_cycles: int = m.MUX_PHASES
    clock_mhz: float = 400.0   # the paper's clock; the FPGA's is unknown yet

    @property
    def block(self):
        """Tokens per block: what one acc_buffer pass can hold."""
        return self.depth // m.MUX_PHASES


@dataclass
class Stats:
    jobs: int = 0
    loads: int = 0
    compute: int = 0           # cycles the core is busy
    load_hidden: int = 0       # load cycles that overlapped compute
    stall: int = 0             # cycles the core waited for the loader
    drain: int = 0
    prologue: int = 0          # first load, nothing to overlap with
    macs: int = 0              # useful MACs (unpadded)

    @property
    def total(self):
        return self.prologue + self.compute + self.stall + self.drain

    def merge(self, o: "Stats"):
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(o, k))
        return self


def wrap32(x):
    """Two's-complement wrap to ACC_W bits (the acc_buffer adder wraps)."""
    x = np.asarray(x, dtype=np.int64) & ((1 << m.ACC_W) - 1)
    return np.where(x > ACC_MAX, x - (1 << m.ACC_W), x)


# ---------------------------------------------------------------------------
# One 8-cycle pass over the whole array, vectorised over partitions.
# Same algorithm as cimple_model.partition_mac (MSB first, sign cycle
# subtracts), checked against it in tests/test_nearmem_model.py.
# ---------------------------------------------------------------------------
def core_pass(bank, acts):
    """bank: (N_PART, N_ROWS) INT8, acts: (N_ROWS,) INT8 -> (N_PART,) 23b."""
    msb, lsb = m.nibble_split(bank)
    a_raw = m.u8(acts)
    acc = np.zeros(m.N_PART, dtype=np.int64)
    for k in range(m.DATA_W - 1, -1, -1):
        xb = (a_raw >> k) & 1
        comb = ((msb * xb).sum(axis=1) << m.NIB_W) + (lsb * xb).sum(axis=1)
        acc = -comb if k == m.DATA_W - 1 else (acc << 1) + comb
    assert np.all((acc >= -(1 << (m.MAC_W - 1))) & (acc < (1 << (m.MAC_W - 1))))
    return acc


# ---------------------------------------------------------------------------
# The weight store and the two-bank array
# ---------------------------------------------------------------------------
class UramStore:
    """Weight tiles resident in UltraRAM, (k_tile, m_tile) -> (parts, rows)."""

    def __init__(self, W):
        """W: (K, M) INT8, the layer's weights in X @ W orientation."""
        K, M = W.shape
        self.k_tiles = -(-K // m.N_ROWS)
        self.m_tiles = -(-M // m.N_PART)
        self.bits = K * M * m.DATA_W
        assert self.bits <= URAM_BITS, f"{self.bits} b does not fit in UltraRAM"
        self.tiles = {}
        for kt in range(self.k_tiles):
            for mt in range(self.m_tiles):
                blk = W[kt * m.N_ROWS:(kt + 1) * m.N_ROWS,
                        mt * m.N_PART:(mt + 1) * m.N_PART]
                self.tiles[kt, mt] = np.ascontiguousarray(blk.T)   # parts x rows


class Array:
    """Two banks; one active (compute), one shadow (load)."""

    def __init__(self, rng):
        # power-up garbage: proves padding never depends on zeroed storage
        self.bank = rng.integers(-128, 128, (m.N_BANKS, m.N_PART, m.N_ROWS))
        self.active = 0
        self.holds = [None, None]           # which tile each bank holds

    def load(self, bank, key, tile):
        assert bank != self.active, "loader wrote the active bank"
        p, r = tile.shape
        self.bank[bank, :p, :r] = tile      # used region only, rest stays stale
        self.holds[bank] = key

    def swap(self):
        self.active ^= 1


# ---------------------------------------------------------------------------
# A whole matmul on the near-memory schedule
# ---------------------------------------------------------------------------
def nm_matmul(A, W, cfg: Config = Config(), seed: int = 0):
    """Y = A @ W on the double-buffered CIM.

    A : (N, K) INT8 activations, W : (K, M) INT8 weights.
    Returns the (N, M) 32b sums exactly as the acc_buffer holds them, and the
    cycle Stats for the schedule.
    """
    A = np.asarray(A, dtype=np.int64)
    N, K = A.shape
    store = UramStore(np.asarray(W, dtype=np.int64))
    arr = Array(np.random.default_rng(seed))
    st = Stats()
    Y = np.zeros((N, W.shape[1]), dtype=np.int64)

    def load_cycles(key):
        p, r = store.tiles[key].shape
        return -(-(p * r) // cfg.load_wpc)

    # [A] loop order (decision 2): m tile, token block, k tile
    jobs = [(mt, b0, kt)
            for mt in range(store.m_tiles)
            for b0 in range(0, N, cfg.block)
            for kt in range(store.k_tiles)]

    first = (jobs[0][2], jobs[0][0])
    arr.load(1, first, store.tiles[first])  # prologue into the idle bank
    arr.swap()
    st.prologue = load_cycles(first)
    st.loads = 1

    buf = np.zeros((cfg.depth, m.N_LANES), dtype=np.int64)
    for j, (mt, b0, kt) in enumerate(jobs):
        key = (kt, mt)
        assert arr.holds[arr.active] == key
        bank = arr.bank[arr.active]
        p, r = store.tiles[key].shape
        toks = range(b0, min(b0 + cfg.block, N))

        # --- compute this job from the active bank ---------------------------
        for t_i, n in enumerate(toks):
            x = np.zeros(m.N_ROWS, dtype=np.int64)
            seg = A[n, kt * m.N_ROWS:kt * m.N_ROWS + r]
            x[:len(seg)] = seg              # unused rows: zero activations
            acc = core_pass(bank, x)
            for ph in range(m.MUX_PHASES):  # the 3b counter walk
                lanes = np.array(m.out_mux(acc, ph))
                e = t_i * m.MUX_PHASES + ph
                buf[e] = lanes if kt == 0 else wrap32(buf[e] + lanes)
            st.macs += p * len(seg)
        c = len(toks) * (cfg.mac_cycles + cfg.out_cycles)
        st.compute += c
        st.jobs += 1

        # --- meanwhile, the loader fills the shadow bank ---------------------
        if j + 1 < len(jobs):
            nmt, _, nkt = jobs[j + 1]
            nkey = (nkt, nmt)
            if nkey != key:
                L = load_cycles(nkey)
                arr.load(arr.active ^ 1, nkey, store.tiles[nkey])
                st.loads += 1
                st.load_hidden += min(L, c)
                st.stall += max(0, L - c)
                arr.swap()

        # --- last K tile of the block: drain to the quant unit ---------------
        if kt == store.k_tiles - 1:
            for t_i, n in enumerate(toks):
                for ph in range(m.MUX_PHASES):
                    for l in range(m.N_LANES):
                        col = mt * m.N_PART + l * m.MUX_PHASES + ph
                        if col < Y.shape[1]:
                            Y[n, col] = buf[t_i * m.MUX_PHASES + ph, l]
            st.drain += len(toks) * m.MUX_PHASES

    return Y, st


# ---------------------------------------------------------------------------
# An attention head on the near-memory schedule
# ---------------------------------------------------------------------------
def requant(acc, ratio):
    scl, sh = m.ratio_to_scale_shift(ratio)
    return np.vectorize(lambda a: m.quantise(int(a), scl, sh))(acc)


def attention_head_nm(X, Wq, Wk, Wv, cfg: Config = Config(), score_scale=None):
    """Encoder-only head, any shape: same calibration as attention_head_cim,
    but every matmul runs through nm_matmul (tiled, double-buffered).

        1. projection : Q,K,V = X @ W         (W stationary)
        2. QK^T       : Q @ K^T               (K^T stationary)
        3. softmax    : z -> e (exp LUT) -> e >> 1   [A] decision 4
        4. A'V        : (e >> 1) @ V, then the reciprocal scale per row
    """
    stats = {}
    sx = np.abs(X).max() / 127
    qX = m.to_int8(X, sx)

    proj = []
    for name, W in zip("QKV", (Wq, Wk, Wv)):
        sw = np.abs(W).max() / 127
        acc, stats[f"proj_{name}"] = nm_matmul(qX, m.to_int8(W, sw), cfg)
        sc = np.abs(acc * sx * sw).max() / 127
        proj.append((requant(acc, sx * sw / sc), sc))
    (Qi, sq), (Ki, sk), (Vi, sv) = proj

    d = Qi.shape[1]
    acc_s, stats["qkt"] = nm_matmul(Qi, Ki.T, cfg)
    if score_scale is None:
        score_scale = np.abs(acc_s * sq * sk / np.sqrt(d)).max() / 127
    z = requant(acc_s, sq * sk / np.sqrt(d) / score_scale)

    exp_lut, recip_lut = m.build_exp_lut(score_scale), m.build_recip_lut()
    e7 = exp_lut[m.u8(z)] >> 1
    partial, stats["av"] = nm_matmul(e7, Vi, cfg)

    out = np.zeros_like(partial)
    for i in range(partial.shape[0]):
        s = int(e7[i].sum())
        assert s < (1 << m.SUM_W)
        M = int(recip_lut[m.mantissa8(s)])
        shift = m.RECIP_FRAC - m.LUT_W + m.msb_pos(s)
        out[i] = [m.quantise(int(p), M, shift) for p in partial[i]]

    return dict(out=out, out_scale=sv, Q=Qi, K=Ki, V=Vi, z=z, e7=e7,
                acc_s=acc_s, score_scale=score_scale, stats=stats)


def summary(st: Stats, cfg: Config = Config()):
    us = st.total / cfg.clock_mhz
    tops = 2 * st.macs / (st.total / (cfg.clock_mhz * 1e6)) / 1e12
    return (f"{st.total} cycles ({us:.1f} us @ {cfg.clock_mhz:.0f} MHz): "
            f"compute {st.compute}, stall {st.stall}, drain {st.drain}, "
            f"prologue {st.prologue}; {st.loads} tile loads, "
            f"{st.load_hidden} load cycles hidden; {tops:.3f} TOPS effective")
