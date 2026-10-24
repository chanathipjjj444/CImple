"""Regression tests for README / CLI completeness (performance metrics, LUT
output files, documented study layout, assumption tags)."""
from __future__ import annotations

import typing
from pathlib import Path

import numpy as np
import pytest

from cimple_sw import hw
from cimple_sw import mapping as mp
from cimple_sw.config import PRESETS, RecipMode
from cimple_sw.hw import Timing

SW = Path(__file__).resolve().parents[1]
README = (SW / "README.md").read_text()


# --- F1: performance metrics -------------------------------------------------
def test_peak_ops_and_tops_documented():
    assert hw.PEAK_OPS_PER_CYCLE == 512
    for mhz, tops in ((417, "0.214"), (770, "0.394")):
        assert f"{hw.PEAK_OPS_PER_CYCLE * mhz / 1e6:.3f}" == tops
        assert f"{tops} TOPS at {mhz} MHz" in README
    assert "1 MAC = 2 OP" in README
    assert "417-770 MHz" in README
    assert "--freq" in README


def test_map_freq_flag_and_tops(capsys):
    assert mp.main(["--n", "40", "--d-model", "64", "--heads", "2", "--dec-tokens", "3",
                    "--freq", "770"]) == 0
    out = capsys.readouterr().out
    assert "@ 770 MHz" in out
    assert f"{512 * 770 / 1e3:.1f} GOPS peak" in out


def test_stats_tops_consistent():
    rng = np.random.default_rng(0)
    w = mp.AttnWeights.random(64, 2, 32, rng)
    X = rng.standard_normal((40, 64))
    st = mp.encoder_layer(X, w, timing=Timing(freq_mhz=770.0)).stats
    peak_tops = hw.PEAK_OPS_PER_CYCLE * 770e6 / 1e12
    assert st.tops == pytest.approx(st.utilisation * peak_tops)
    # 16-cycle pass (no output overlap) caps delivery at half of peak
    assert Timing().pass_cycles == 2 * Timing().mac_cycles
    assert st.utilisation <= 0.5 + 1e-12


# --- F2: the study actually being run is documented -------------------------
def test_run_main_layout_documented():
    script = (SW / "results" / "run_main.sh").read_text()
    for d in ("main3", "hellaswag2000", "ablation250"):
        assert f"results/{d}" in script
        assert f"results/{d}" in README
    assert "--limit 2000" in script and "--limit 2000" in README
    assert "run_main.sh" in README


# --- F3: full-precision LUT is not written as $readmemh hex ------------------
def test_cli_luts_full_precision_not_hex(tmp_path, capsys):
    from cimple_rtl.model import cimple_model as gm

    from cimple_sw.cli import main
    assert main(["luts", "--score-scale", "0.05", "--precision", "full",
                 "--out-dir", str(tmp_path / "full")]) == 0
    assert not (tmp_path / "full" / "exp_lut.hex").exists()
    vals = [float(x) for x in (tmp_path / "full" / "exp_lut_full.txt").read_text().split()]
    assert len(vals) == 256
    rec = [int(x, 16) for x in (tmp_path / "full" / "recip_lut.hex").read_text().split()]
    assert rec == gm.build_recip_lut().tolist()

    assert main(["luts", "--score-scale", "0.05", "--out-dir", str(tmp_path / "hw8")]) == 0
    hx = [int(x, 16) for x in (tmp_path / "hw8" / "exp_lut.hex").read_text().split()]
    assert hx == gm.build_exp_lut(0.05).tolist()


# --- F4: 32.7% is a two-knob combination, not a single choice ---------------
def test_latency_single_vs_combined_wording():
    assert "Several single, documented baseline choices give 32.7" not in README
    assert "33.3-34.0%" in README and "overlap_out` + `sm_reads=1` gives 32.7%" in README


# --- F5: exact reciprocal is an assumption ----------------------------------
def test_exact_recip_tagged_assumption():
    assert typing.get_args(RecipMode) == ("exact", "lut")
    assert PRESETS["cimple"].recip == "exact"
    row = next(ln for ln in README.splitlines() if ln.startswith("| `cimple` |"))
    assert "[A]" in row and "exact `1/sum`" in row
    from cimple_sw import config
    doc = config.SoftmaxConfig.__doc__
    i = doc.index('"exact" --')
    assert "[A]" in doc[i:i + 300]
