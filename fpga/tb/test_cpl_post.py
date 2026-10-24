"""cocotb test: cpl_post -- ACC buffer, quantiser, EXP/NORM split softmax, RAW32.

    uv run make -C fpga/tb MODULE=test_cpl_post TOPLEVEL=cpl_post

Reference: a direct Python transcription of ARCH.md §5 MATVEC steps 3-4
(class PostRef) built on the golden-model functions gm.quantise,
gm.mantissa8, gm.msb_pos, gm.build_recip_lut, gm.build_exp_lut. Every test
streams 4-lane groups (column p = 8*lane + phase) with random gaps or
back-to-back, checks every output (data on enabled lanes, lane enables, t,
phase, order) and the fixed latency POST_LAT, and waits for out_busy_o = 0
between instructions as the engine does.
"""
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, ReadOnly

from cimple_rtl.model import cimple_model as gm

N_LANES, N_PH, N_SUMS = 4, 8, 32
POST_LAT = 13   # cpl_post.sv: +1 ACC BRAM output register (DOB_REG), +1 quantiser input register
ACC_DEPTH = 1024
MAC_MIN, MAC_MAX = -(1 << 22), (1 << 22) - 1
ACC_NONE, ACC_FIRST, ACC_ADD = 0, 1, 2
OUT_NONE, OUT_Q8, OUT_EXP, OUT_NORM, OUT_RAW32 = 0, 1, 2, 3, 4


def s32(x):
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x & 0x80000000 else x


def cfg(**kw):
    c = dict(acc_mode=ACC_NONE, out_mode=OUT_Q8, scale=1, shift=0, ncols=32,
             sum_by_t=0, causal=0, qbase=0, kbase=0, acc_base=0)
    c.update(kw)
    return c


class PostRef:
    """ARCH.md §5 MATVEC steps 3-4, one 4-lane group at a time."""

    def __init__(self):
        self.acc = {}
        self.sums = [0] * N_SUMS
        self.exp_lut = [0] * 256
        self.recip = [int(x) for x in gm.build_recip_lut()]

    def group(self, c, cim, phase, t):
        e = (c["acc_base"] + 8 * t + phase) % ACC_DEPTH
        en, data = [], []
        for l in range(N_LANES):
            p = 8 * l + phase
            a = int(cim[l])
            if c["acc_mode"] == ACC_NONE:
                v = a
            elif c["acc_mode"] == ACC_FIRST:
                self.acc[(e, l)] = a
                v = a
            else:
                assert (e, l) in self.acc, "test bug: ADD to an entry never written by FIRST"
                self.acc[(e, l)] = s32(self.acc[(e, l)] + a)
                v = self.acc[(e, l)]
            om = c["out_mode"]
            on = om != OUT_NONE and p < c["ncols"]
            en.append(on)
            d = None
            if not on:
                pass
            elif om == OUT_Q8:
                d = gm.quantise(v, c["scale"], c["shift"]) & 0xFF
            elif om == OUT_EXP:
                z = gm.quantise(v, c["scale"], c["shift"])
                if c["sum_by_t"]:
                    q, k = c["qbase"] + t, c["kbase"] + p
                else:
                    q, k = c["qbase"] + p, c["kbase"] + t
                ev = 0 if (c["causal"] and k > q) else self.exp_lut[z & 0xFF]
                d = ev
                i = (t % 32) if c["sum_by_t"] else p
                self.sums[i] = (self.sums[i] + ev) % (1 << 24)
            elif om == OUT_NORM:
                i = (t % 32) if c["sum_by_t"] else p
                s = self.sums[i]
                d = gm.quantise(v, self.recip[gm.mantissa8(s)], 7 + gm.msb_pos(s)) & 0xFF
            elif om == OUT_RAW32:
                d = v & 0xFFFFFFFF
            data.append(d)
        return en, data


class Tb:
    def __init__(self, dut):
        self.dut = dut
        self.ref = PostRef()
        self.cycle = 0
        self.outs = []
        self.rng = np.random.default_rng(4)

    async def start(self):
        d = self.dut
        cocotb.start_soon(Clock(d.clk_i, 5, unit="ns").start())   # 200 MHz
        d.rst_ni.value = 0
        self.set_cfg(cfg())
        d.sum_clr_i.value = 0
        d.lut_we_i.value = 0
        d.lut_waddr_i.value = 0
        d.lut_wdata_i.value = 0
        d.in_valid_i.value = 0
        d.in_phase_i.value = 0
        d.in_t_i.value = 0
        for l in range(N_LANES):
            d.in_cim_i[l].value = 0
        for _ in range(3):
            await FallingEdge(d.clk_i)
        d.rst_ni.value = 1
        cocotb.start_soon(self.monitor())
        await FallingEdge(d.clk_i)

    async def monitor(self):
        d = self.dut
        while True:
            await RisingEdge(d.clk_i)
            self.cycle += 1
            await ReadOnly()
            if int(d.out_valid_o.value):
                raw = int(d.out_data_o.value)
                self.outs.append(dict(
                    cycle=self.cycle, t=int(d.out_t_o.value), phase=int(d.out_phase_o.value),
                    en=[(int(d.out_lane_en_o.value) >> l) & 1 for l in range(N_LANES)],
                    data=[(raw >> (32 * l)) & 0xFFFFFFFF for l in range(N_LANES)]))

    def set_cfg(self, c):
        d = self.dut
        d.acc_mode_i.value = c["acc_mode"]
        d.out_mode_i.value = c["out_mode"]
        d.scale_i.value = c["scale"]
        d.shift_i.value = c["shift"]
        d.ncols_i.value = c["ncols"]
        d.sum_by_t_i.value = c["sum_by_t"]
        d.causal_i.value = c["causal"]
        d.qbase_i.value = c["qbase"]
        d.kbase_i.value = c["kbase"]
        d.acc_base_i.value = c["acc_base"]

    async def sum_clr(self):
        self.dut.sum_clr_i.value = 1
        await FallingEdge(self.dut.clk_i)
        self.dut.sum_clr_i.value = 0
        self.ref.sums = [0] * N_SUMS

    async def lut_load(self, table):
        d = self.dut
        for a in range(16):
            d.lut_we_i.value = 1
            d.lut_waddr_i.value = a
            d.lut_wdata_i.value = sum(int(table[16 * a + i]) << (8 * i) for i in range(16))
            await FallingEdge(d.clk_i)
        d.lut_we_i.value = 0
        self.ref.exp_lut = [int(x) for x in table]

    async def instr(self, c, groups, gap=0.0):
        """Stream groups = [(cim[4], phase, t)] under config c, check all outputs."""
        d = self.dut
        assert int(d.out_busy_o.value) == 0
        self.set_cfg(c)
        self.outs = []
        sent = []
        for cim, phase, t in groups:
            while self.rng.random() < gap:
                d.in_valid_i.value = 0
                await FallingEdge(d.clk_i)
            d.in_valid_i.value = 1
            d.in_phase_i.value = phase
            d.in_t_i.value = t
            for l in range(N_LANES):
                d.in_cim_i[l].value = int(cim[l]) & 0x7FFFFF
            sent.append(self.cycle)
            await FallingEdge(d.clk_i)
            assert int(d.out_busy_o.value) == 1
        d.in_valid_i.value = 0
        while int(d.out_busy_o.value):
            await FallingEdge(d.clk_i)
        assert len(self.outs) == len(groups), f"{len(self.outs)} outputs for {len(groups)} inputs"
        for n, ((cim, phase, t), o) in enumerate(zip(groups, self.outs)):
            en, data = self.ref.group(c, cim, phase, t)
            where = f"group {n} (t={t}, phase={phase}, cfg={c})"
            assert o["cycle"] - sent[n] == POST_LAT, f"{where}: latency {o['cycle'] - sent[n]}"
            assert (o["t"], o["phase"]) == (t, phase), f"{where}: got t={o['t']} phase={o['phase']}"
            assert o["en"] == [int(x) for x in en], f"{where}: lane_en {o['en']} != {en}"
            for l in range(N_LANES):
                if en[l]:
                    assert o["data"][l] == data[l], \
                        f"{where} lane {l}: data {o['data'][l]:#x} != ref {data[l]:#x}"
        return self.outs

    def rand_cim(self, n=N_LANES, lo=MAC_MIN, hi=MAC_MAX):
        return [int(x) for x in self.rng.integers(lo, hi + 1, n)]

    def drain(self, ntok, t0=0, cim_fn=None):
        """The engine's drain order: for each t, phases 0..7."""
        cim_fn = cim_fn or (lambda t, ph: self.rand_cim())
        return [(cim_fn(t, ph), ph, t) for t in range(t0, t0 + ntok) for ph in range(N_PH)]


# ---------------------------------------------------------------------------
@cocotb.test()
async def test_acc_first_add_none(dut):
    """FIRST then several ADDs (RAW32 shows v exactly), NONE bypass, gaps + back-to-back."""
    tb = Tb(dut)
    await tb.start()
    for acc_base, gap in ((0, 0.0), (1000, 0.3), (123, 0.0)):   # 1000: wraps mod ACC_DEPTH
        c = cfg(acc_mode=ACC_FIRST, out_mode=OUT_RAW32, acc_base=acc_base)
        await tb.instr(c, tb.drain(6), gap=gap)
        for k in range(4):
            c = cfg(acc_mode=ACC_ADD, out_mode=OUT_RAW32, acc_base=acc_base)
            await tb.instr(c, tb.drain(6), gap=gap if k % 2 else 0.0)
    # NONE: no buffer access, v = acc; then ADD must still see the old buffer
    await tb.instr(cfg(acc_mode=ACC_NONE, out_mode=OUT_RAW32, acc_base=0), tb.drain(6))
    await tb.instr(cfg(acc_mode=ACC_ADD, out_mode=OUT_RAW32, acc_base=0), tb.drain(6), gap=0.5)
    # ADD with Q8 output and a random scale/shift, out_mode NONE (no lane enables)
    await tb.instr(cfg(acc_mode=ACC_ADD, out_mode=OUT_Q8, scale=12345, shift=24), tb.drain(6))
    await tb.instr(cfg(acc_mode=ACC_ADD, out_mode=OUT_NONE), tb.drain(6))
    await tb.instr(cfg(acc_mode=ACC_ADD, out_mode=OUT_RAW32), tb.drain(6))


@cocotb.test()
async def test_acc_wrap32(dut):
    """ADD wraps mod 2^32 (one group per instruction, 520 x +2^22-1)."""
    tb = Tb(dut)
    await tb.start()
    g = [([MAC_MAX, MAC_MIN, MAC_MAX, 7], 3, 2)]
    await tb.instr(cfg(acc_mode=ACC_FIRST, out_mode=OUT_RAW32), g)
    for _ in range(520):
        await tb.instr(cfg(acc_mode=ACC_ADD, out_mode=OUT_RAW32), g)
    assert tb.ref.acc[(19, 0)] < 0 and tb.ref.acc[(19, 1)] > 0, "test did not wrap"


@cocotb.test()
async def test_q8_random(dut):
    """Q8 with random scale/shift/ncols, odd drain orders (any phase/t order)."""
    tb = Tb(dut)
    await tb.start()
    for k in range(12):
        sh = int(tb.rng.integers(0, 32))
        c = cfg(acc_mode=int(tb.rng.integers(0, 2)), out_mode=OUT_Q8,
                scale=int(tb.rng.integers(0, 1 << 16)), shift=sh,
                ncols=int(tb.rng.integers(0, 33)), acc_base=int(tb.rng.integers(0, 1 << 16)))
        groups = [(tb.rand_cim(), int(tb.rng.integers(0, 8)), int(tb.rng.integers(0, 1 << 16)))
                  for _ in range(40)]
        # FIRST: entries must not repeat inside one instruction
        if c["acc_mode"] == ACC_FIRST:
            seen, uniq = set(), []
            for g in groups:
                e = (c["acc_base"] + 8 * g[2] + g[1]) % ACC_DEPTH
                if e not in seen:
                    seen.add(e)
                    uniq.append(g)
            groups = uniq
        await tb.instr(c, groups, gap=0.2 if k % 2 else 0.0)


@cocotb.test()
async def test_exp_sums(dut):
    """EXP: LUT + causal mask + exp-sums, both index modes, ncols < 32, consecutive same-sum hits."""
    tb = Tb(dut)
    await tb.start()
    await tb.lut_load(gm.build_exp_lut(0.06))
    small = lambda t, ph: tb.rand_cim(lo=-3000, hi=3000)
    cases = [
        cfg(out_mode=OUT_EXP, scale=1, shift=4, sum_by_t=0),
        cfg(out_mode=OUT_EXP, scale=3, shift=5, sum_by_t=1, ncols=20),
        cfg(out_mode=OUT_EXP, scale=1, shift=4, sum_by_t=1, causal=1, qbase=5, kbase=0),
        cfg(out_mode=OUT_EXP, scale=1, shift=4, sum_by_t=0, causal=1, qbase=3, kbase=7, ncols=29),
        cfg(out_mode=OUT_EXP, scale=1, shift=3, sum_by_t=0, causal=1, qbase=0, kbase=40, ncols=13),
    ]
    for k, c in enumerate(cases):
        await tb.sum_clr()
        # 40 tokens: with sum_by_t, t mod 32 revisits sums 0..7
        await tb.instr(c, tb.drain(40, cim_fn=small), gap=0.25 if k % 2 else 0.0)
        # accumulate further without a clear, back-to-back
        await tb.instr(c, tb.drain(9, t0=40, cim_fn=small))
    # the same t repeated on consecutive cycles (all 8 groups hit S[t mod 32])
    await tb.sum_clr()
    c = cfg(out_mode=OUT_EXP, scale=1, shift=2, sum_by_t=1)
    await tb.instr(c, [(tb.rand_cim(lo=-500, hi=500), ph % 8, 7) for ph in range(24)])
    # random LUT contents; sum wraps mod 2^24 would need 2^16 groups -- not exercised
    await tb.lut_load(tb.rng.integers(0, 256, 256))
    await tb.instr(cfg(out_mode=OUT_EXP, scale=7, shift=9, sum_by_t=0), tb.drain(16), gap=0.3)


@cocotb.test()
async def test_norm(dut):
    """NORM reads the sums of the previous EXP instruction (both index modes)."""
    tb = Tb(dut)
    await tb.start()
    await tb.lut_load(gm.build_exp_lut(0.05))
    small = lambda t, ph: tb.rand_cim(lo=-2500, hi=2500)
    for sbt in (0, 1):
        await tb.sum_clr()
        await tb.instr(cfg(out_mode=OUT_EXP, scale=1, shift=4, sum_by_t=sbt,
                           causal=1, qbase=10, kbase=0), tb.drain(20, cim_fn=small))
        # partial sums sum_j e_j V_j are up to ~2^21 here
        await tb.instr(cfg(acc_mode=ACC_FIRST, out_mode=OUT_NORM, sum_by_t=sbt), tb.drain(20), gap=0.2)
        await tb.instr(cfg(acc_mode=ACC_ADD, out_mode=OUT_NORM, sum_by_t=sbt, ncols=17), tb.drain(20))
    # columns p >= ncols must not count in the sums: EXP with ncols < 32, then
    # NORM over all 32 columns reads every sum (incl. the untouched ones)
    for sbt, nc in ((1, 20), (0, 13)):
        await tb.sum_clr()
        await tb.instr(cfg(out_mode=OUT_EXP, scale=1, shift=4, sum_by_t=sbt, ncols=nc),
                       tb.drain(12, cim_fn=small))
        await tb.instr(cfg(out_mode=OUT_NORM, sum_by_t=sbt), tb.drain(12), gap=0.1)
    # empty sums (s = 0): msb = 0, m = 0, RECIP[0] = 128, shift 7 -> identity
    await tb.sum_clr()
    await tb.instr(cfg(out_mode=OUT_NORM), tb.drain(2, cim_fn=lambda t, ph: tb.rand_cim(lo=-200, hi=200)))


@cocotb.test()
async def test_softmax_row(dut):
    """End to end against gm.softmax_row: one attention row = column p, keys = t."""
    tb = Tb(dut)
    await tb.start()
    lut = gm.build_exp_lut(0.04)
    await tb.lut_load(lut)
    await tb.sum_clr()
    rng = np.random.default_rng(7)
    ntok = 24
    scores = rng.integers(-128, 128, (ntok, 32))          # z[t][p] (already INT8)
    values = rng.integers(-128, 128, (ntok, 32))          # V[t] reduced per column
    # EXP with scale 1 / shift 0: z = v -> sums S[p] = sum_t EXPLUT[z]
    await tb.instr(cfg(out_mode=OUT_EXP, sum_by_t=0),
                   tb.drain(ntok, cim_fn=lambda t, ph: [int(scores[t][8 * l + ph]) for l in range(4)]))
    # NORM on the partial attention score sum_t e[t][p] * V[t][p]
    part = [[int(sum(int(lut[int(scores[t][p]) & 0xFF]) * int(values[t][p]) for t in range(ntok)))
             for p in range(32)]]
    outs = await tb.instr(cfg(out_mode=OUT_NORM, sum_by_t=0),
                          [([part[0][8 * l + ph] for l in range(4)], ph, 0) for ph in range(8)])
    for o in outs:
        for l in range(4):
            p = 8 * l + o["phase"]
            r = gm.softmax_row(scores[:, p], values[:, p], lut, gm.build_recip_lut())
            got = o["data"][l] - 256 if o["data"][l] & 0x80 else o["data"][l]
            assert got == r["score"], f"column {p}: {got} != gm.softmax_row {r['score']}"


@cocotb.test()
async def test_sum_clr_and_raw32(dut):
    """sum_clr zeroes every sum; RAW32 of NONE is the sign-extended input."""
    tb = Tb(dut)
    await tb.start()
    await tb.lut_load(np.full(256, 255))
    await tb.instr(cfg(out_mode=OUT_EXP, sum_by_t=0), tb.drain(3))
    assert any(tb.ref.sums)
    await tb.sum_clr()
    # NORM after the clear sees s = 0 everywhere (identity quantise, shift 7)
    await tb.instr(cfg(out_mode=OUT_NORM, sum_by_t=0), tb.drain(1, cim_fn=lambda t, ph: tb.rand_cim(lo=-100, hi=100)))
    await tb.instr(cfg(out_mode=OUT_NORM, sum_by_t=1), tb.drain(1, t0=31, cim_fn=lambda t, ph: tb.rand_cim(lo=-100, hi=100)))
    edges = [MAC_MIN, MAC_MAX, -1, 0]
    await tb.instr(cfg(out_mode=OUT_RAW32), [(edges, ph, 9) for ph in range(8)])
