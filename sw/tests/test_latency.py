"""Tests for the Sec. V-B latency model (cimple_sw.latency)."""
from __future__ import annotations

from collections import defaultdict

import pytest

from cimple_sw.hw import Timing
from cimple_sw.latency import STAGES, Knobs, compare, latency, main, sweep

T16 = Timing()                      # mac 8 + mux 8 = 16 cycles / pass
T8 = Timing(overlap_out=True)       # 8 cycles / pass, mux walk overlapped


def test_tiny_split_by_hand():
    """N=64, d=64, defaults. 2 Q tiles x 1 chunk, 2 V column tiles.
    P = 16, QK path = quant 1 + LUT 1 (tail 2), A'V final path = ACC 1 +
    quant 1 (tail 2), tile write = 2048 / 16 = 128. Pingpong banks.

      load Q0 -> bank0            [   0,  128)
      QK t0 (64 keys)             [ 128, 1152)  e ready 1154
      load V(0,0) -> bank1        [ 128,  256)  hidden
      AV t0 v0 (32 queries)       [1154, 1666)  waits for e (2 cycles)
      load Q1 -> bank0            [1152, 1280)  hidden
      QK t1 keys 0-31 (segment)   [1666, 2178)
      load V(0,1) -> bank1        [1666, 1794)  hidden behind the QK segment
      AV t0 v1                    [2178, 2690)
      QK t1 keys 32-63            [2690, 3202)  e ready 3204
      load V(0,0) -> bank1        [2690, 2818)  hidden
      AV t1 v0                    [3204, 3716)
      load V(0,1) -> bank0        [3202, 3330)  hidden (Q1 no longer needed)
      AV t1 v1                    [3716, 4228)  done 4230
    = 256 passes * 16 + 128 + 2 + 2 + 2 = 4230.
    With bank_policy="fixed" the last V(0,1) must go to bank1 after AV t1 v0
    releases it: +128 exposed -> 4358.
    """
    r = latency(64, 64, T16, "split")
    assert r.cycles == 4230
    assert r.breakdown["qk"] == 128 * 16 and r.breakdown["av"] == 128 * 16
    # only the first Q load stalls the core; the 2 e-drain cycles that run
    # under the hidden Q1 load are drain, not load (LAT-3)
    assert r.breakdown["load_q"] == 128 and r.breakdown["load_v"] == 0
    f = latency(64, 64, T16, "split", bank_policy="fixed")
    assert f.cycles == 4358 and f.breakdown["load_v"] == 128


def test_tiny_baseline_by_hand():
    """N=64, d=64, defaults: per Q tile QK^T (64 passes, tail ACC 1), then
    3 softmax passes of 32*64/4 = 512 cycles (tail 2 each), then 2 A'V jobs
    of 32 passes (last tail 2). V / next-Q loads hide in the other bank
    (pingpong) behind QK^T, the softmax and the previous A'V; only the first
    Q load (128) is exposed.
      tile = 1024 + 1 + 3 * (512 + 2) + 2 * 512 = 3591
    total = 128 + 2 * 3591 + 2 (last A'V drain) = 7312.
    """
    r = latency(64, 64, T16, "baseline")
    assert r.cycles == 7312
    for s in ("sm_max", "sm_expsum", "sm_norm"):
        assert r.breakdown[s] == 2 * 512
    red = compare(64, 64, T16)["reduction_pct"]
    assert red == pytest.approx(100 * (1 - 4230 / 7312))


def test_headline_regression():
    """N=1024, d=64, 400 MHz: core-bound split vs sequential baseline."""
    r = compare(1024, 64, T16)
    core = 2 * (1024 * 1024 // 32) * 16           # QK^T + A'V passes x 16
    sm = 32 * 3 * (32 * 1024 // 4)                # 32 tiles x 3 reads
    assert r["split"].cycles == 1049730
    assert r["baseline"].cycles == 1835362
    assert r["split"].cycles - core < 0.002 * core
    assert r["baseline"].cycles - (core + sm) < 0.002 * core
    assert r["reduction_pct"] == pytest.approx(100 * sm / (core + sm), abs=0.1)
    assert r["split"].us == pytest.approx(1049730 / 400)


@pytest.mark.parametrize("variant", ["split", "baseline"])
@pytest.mark.parametrize("N,d", [(64, 64), (100, 40), (256, 128), (96, 32)])
@pytest.mark.parametrize("kn", [{}, {"interleave": False},
                                {"bank_policy": "fixed"},
                                {"write_while_compute": False},
                                {"baseline_stream_norm": True,
                                 "baseline_tile_overlap": True},
                                {"sm_reads": 2, "bank_switch_cycles": 3}])
@pytest.mark.parametrize("tm", [T16, T8, Timing(lut_cycles=3)])
def test_breakdown_sums_and_resources_exclusive(variant, N, d, kn, tm):
    r = latency(N, d, tm, variant, **kn)
    assert sum(r.breakdown.values()) == r.cycles
    assert set(r.breakdown) == set(STAGES)
    assert r.breakdown["idle"] == 0
    by_res = defaultdict(list)
    for t in r.tasks:
        assert t.start <= t.end <= t.done <= r.cycles
        for x in t.res:
            by_res[x].append((t.start, t.end))
    for x, iv in by_res.items():
        iv.sort()
        for (a0, a1), (b0, b1) in zip(iv, iv[1:]):
            assert a1 <= b0, f"{x} double-booked"


@pytest.mark.parametrize("tm", [T16, T8])
def test_split_never_slower_with_default_knobs(tm):
    for row in sweep((1, 17, 64, 100, 256, 1024), (16, 32, 64, 96, 128),
                     (tm.overlap_out,)):
        assert row["split"] <= row["baseline"], row


@pytest.mark.parametrize("variant", ["split", "baseline"])
@pytest.mark.parametrize("d", [32, 64, 128])
def test_monotonic_in_N(variant, d):
    prev = 0
    for N in list(range(1, 200, 7)) + [256, 512, 1024]:
        c = latency(N, d, T16, variant).cycles
        assert c >= prev, (N, c, prev)
        prev = c


@pytest.mark.parametrize("N,d", [(64, 64), (1024, 64), (100, 40)])
def test_ops_and_tops(N, d):
    r = latency(N, d, T16, "split")
    assert r.ops == 2 * N * N * d * 2
    assert r.tops == pytest.approx(r.ops * 400e6 / r.cycles / 1e12)
    assert r.core_util <= 0.5 + 1e-12          # 16-cycle passes: half of peak
    r8 = latency(N, d, T8, "split")
    assert r8.core_util <= 1.0


def test_core_busy_equals_passes():
    for d, n_qr, n_ct in ((32, 1, 1), (64, 1, 2), (128, 2, 4)):
        N = 256
        passes = (N // 32) * N * n_qr + (N // 32) * (N // 64) * n_ct * 32
        for v in ("split", "baseline"):
            assert latency(N, d, T16, v).busy["core"] == passes * 16


def test_exposed_v_reload_without_interleave():
    """Fixed banks, no interleave: V(c,1) can only load after A'V(c,0) has
    released bank1 -> 128 exposed cycles per chunk (511 of 512; the last one
    is exposed in both schedules)."""
    a = latency(1024, 64, T16, "split", bank_policy="fixed").cycles
    b = latency(1024, 64, T16, "split", bank_policy="fixed", interleave=False).cycles
    assert b - a == 511 * 128
    # pingpong banks hide the reloads without interleaving ...
    il = latency(1024, 64, T16, "split")
    no = latency(1024, 64, T16, "split", interleave=False)
    assert no.cycles == il.cycles
    # ... but the second V tile of each chunk evicts Q, so Q is rewritten
    # every chunk: 512 Q writes instead of 32 (documented in Knobs, LAT-4)
    nq = lambda r: sum(t.stage == "load_q" for t in r.tasks)      # noqa: E731
    nv = lambda r: sum(t.stage == "load_v" for t in r.tasks)      # noqa: E731
    assert (nq(il), nv(il)) == (32, 1024)
    assert (nq(no), nv(no)) == (512, 1024)
    assert no.busy["wport"] - il.busy["wport"] == (512 - 32) * 128


@pytest.mark.parametrize("lut,period", [(1, 16), (2, 24), (3, 32)])
def test_backpressure_respects_overlap_out(lut, period):
    """LAT-1: a slower LUT stretches the mux walk to 8 * lut cycles. Without
    overlap_out the walk still serialises with the MAC (8 + walk); with it
    the pass takes max(8, walk). Checked on the QK^T jobs of the split."""
    for ov, want in ((False, period), (True, max(8, 8 * lut))):
        r = latency(1024, 64, Timing(lut_cycles=lut, overlap_out=ov), "split")
        qk = [t for t in r.tasks if t.stage == "qk"]
        assert all(t.end - t.start == (t.side["mux"] // 8) * want for t in qk)
        assert r.breakdown["qk"] == 32 * 1024 * want
        # the last walk never runs past the job end without overlap
        if not ov:
            assert all(t.done - t.end == 1 + lut for t in qk)   # quant + LUT
    r = compare(1024, 64, Timing(lut_cycles=2))
    assert r["split"].cycles - 32 * 1024 * (24 + 16) < 2048
    assert r["reduction_pct"] == pytest.approx(28.5, abs=0.2)


def test_acc_backpressure_without_overlap():
    """LAT-1: acc_cycles=2 throttles every pass that goes through the ACC
    buffer to 8 + 16 = 24 cycles (no overlap_out). The split's QK^T at d=64
    has no ACC on its path (quant -> LUT), so it stays at 16."""
    for v, qk_p in (("split", 16), ("baseline", 24)):
        r = latency(1024, 64, T16, v, acc_cycles=2)
        assert r.breakdown["qk"] == 32 * 1024 * qk_p
        assert r.breakdown["av"] == 32 * 1024 * 24


def test_load_breakdown_is_causal():
    """LAT-3: at the headline point only the first 128-cycle Q load stalls
    the core; the e-drain stalls that overlap hidden loads count as drain."""
    for v in ("split", "baseline"):
        r = latency(1024, 64, T16, v)
        assert r.breakdown["load_q"] == 128 and r.breakdown["load_v"] == 0
    # a load that holds the core is fully exposed
    r = latency(64, 64, T16, "split", write_while_compute=False)
    loads = [t for t in r.tasks if t.stage.startswith("load")]
    assert loads and all(t.exposed == (t.start, t.end) for t in loads)


def test_tile_overlap_removes_the_advantage():
    """Honesty check: if the baseline overlaps the softmax of tile t with the
    core work of tile t+1, both are core-bound and the 33 % disappears."""
    r = compare(1024, 64, T16, baseline_tile_overlap=True)
    assert abs(r["reduction_pct"]) < 2


def test_knob_settings_near_paper():
    """The two documented single-knob settings that land near 33 %."""
    assert compare(1024, 64, T16, sm_reads=2)["reduction_pct"] == pytest.approx(33.3, abs=0.3)
    assert compare(1024, 64, T16, baseline_stream_norm=True)["reduction_pct"] == \
        pytest.approx(34.0, abs=0.3)


def test_near_paper_is_disclosure_only():
    from cimple_sw.latency import near_paper
    rows = near_paper(256)
    assert rows and all(30 <= r["reduction_pct"] <= 36 for r in rows)


def test_buffers():
    sp = latency(1024, 64, T16, "split").buffer_bytes
    bl = latency(1024, 64, T16, "baseline").buffer_bytes
    assert bl["score_buffer"] == 32 * 1024 * 4      # 128 kB of 32b scores
    assert sp["e_buffer"] == 2 * 32 * 64


def test_bad_args():
    with pytest.raises(ValueError):
        latency(64, 64, variant="nope")
    with pytest.raises(ValueError):
        Knobs(sm_reads=4)
    with pytest.raises(TypeError):
        latency(64, 64, not_a_knob=1)


def test_cli(capsys):
    assert main(["--N", "128", "--d", "64", "--no-sweep"]) == 0
    out = capsys.readouterr().out
    assert "latency reduction" in out and "split" in out
    assert main(["--N", "64", "--knob", "sm_reads=2"]) == 0
    out = capsys.readouterr().out
    assert "Sensitivity" in out and "overlap_out=True" in out
