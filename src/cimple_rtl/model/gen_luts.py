#!/usr/bin/env python3
"""Emit the two softmax tables as $readmemh files for the RTL."""
import argparse
import pathlib

from cimple_rtl.model import cimple_model as m


def write_hex(path, values, width_bits=8):
    nib = (width_bits + 3) // 4
    with open(path, "w") as f:
        for v in values:
            f.write(f"{int(v):0{nib}x}\n")
    print(f"wrote {path} ({len(values)} entries)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-scale", type=float, default=0.05,
                    help="dequantisation scale of the INT8 QK^T score")
    ap.add_argument("--zmax", type=int, default=127)
    ap.add_argument("--outdir", default="build")
    a = ap.parse_args()

    out = pathlib.Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    write_hex(out / "exp_lut.hex", m.build_exp_lut(a.score_scale, a.zmax))
    write_hex(out / "recip_lut.hex", m.build_recip_lut())

    nz = int((m.build_exp_lut(a.score_scale, a.zmax) > 0).sum())
    print(f"note: {nz}/256 exp entries are non-zero at scale={a.score_scale}. "
          f"If that is low, the 8b table is wasting range -- see docs/01.")
