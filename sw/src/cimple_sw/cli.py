"""``cimple-sw`` command line.

Subcommands (submodules are imported lazily, so a missing optional module only
breaks its own subcommand):

  eval     lm-evaluation-harness accuracy study, Sec. V-B / Fig. 11
           (:mod:`cimple_sw.eval`); ``--merge`` only rebuilds summary.md.
  latency  -> ``cimple_sw.latency.main(rest)``
  map      -> ``cimple_sw.mapping.main(rest)``
  luts     exp / reciprocal LUT contents for a score scale, as $readmemh hex
           for ``--precision hw8``; ``--precision full`` writes the full-precision exp
           table as decimal text to exp_lut_full.txt (not loadable by
           $readmemh). ``cimple_sw.luts.main(rest)`` is used if defined.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Optional, Sequence

_DELEGATES = {"latency": "cimple_sw.latency", "map": "cimple_sw.mapping"}


def _delegate(modname: str, rest: list[str]) -> int:
    try:
        mod = importlib.import_module(modname)
    except ImportError as e:
        print(f"cimple-sw: {modname} is not available ({e})", file=sys.stderr)
        return 2
    rc = mod.main(rest)
    return 0 if rc is None else int(rc)


def _eval_main(argv: list[str]) -> int:
    from .config import PRESETS
    from .eval import DEFAULT_OUT, DEFAULT_TASKS

    ap = argparse.ArgumentParser(prog="cimple-sw eval",
                                 description="TinyLlama INT8 + CIMple softmax on lm-eval tasks")
    ap.add_argument("--presets", nargs="+", default=["float", "cimple"],
                    help=f"softmax presets, run in order ({', '.join(PRESETS)})")
    ap.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    ap.add_argument("--limit", type=float, default=None,
                    help="docs per task (int) or fraction (<1); default all")
    ap.add_argument("--batch-size", default="8")
    ap.add_argument("--num-fewshot", type=int, default=0)
    ap.add_argument("--quant", default="w8a8", choices=["w8a8", "w8", "dynamic", "none"])
    ap.add_argument("--quantize-lm-head", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="switch split_lut presets to static score scales, calibrated on "
                         "built-in text (no effect on float-mode presets)")
    ap.add_argument("--model", default=None, help="HF model id (default TinyLlama-1.1B-Chat-v1.0)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--threads", type=int, default=None, help="torch threads (default: physical cores)")
    ap.add_argument("--merge", action="store_true",
                    help="only rebuild summary.md from the per-preset JSON in --out-dir")
    a = ap.parse_args(argv)

    from . import eval as ev
    if a.merge:
        print(ev.merge(a.out_dir))
        return 0
    for p in a.presets:
        if p not in PRESETS:
            ap.error(f"unknown preset {p!r}")
    limit = None if a.limit is None else (int(a.limit) if a.limit >= 1 else a.limit)
    bs = a.batch_size if a.batch_size.startswith("auto") else int(a.batch_size)
    print(ev.run_eval(a.presets, a.tasks, limit=limit, batch_size=bs, num_fewshot=a.num_fewshot,
                      quant=a.quant, out_dir=a.out_dir, model_name=a.model, threads=a.threads,
                      quantize_lm_head=a.quantize_lm_head, calibrate=a.calibrate))
    return 0


def _luts_main(argv: list[str]) -> int:
    try:
        luts = importlib.import_module("cimple_sw.luts")
    except ImportError as e:
        print(f"cimple-sw: cimple_sw.luts is not available ({e})", file=sys.stderr)
        return 2
    if hasattr(luts, "main"):
        rc = luts.main(argv)
        return 0 if rc is None else int(rc)

    from .config import PRESETS

    ap = argparse.ArgumentParser(prog="cimple-sw luts",
                                 description="emit EXP_LUT / RECIP_LUT for a score scale")
    ap.add_argument("--score-scale", type=float, required=True,
                    help="real value of one score LSB (e.g. a calibrated static scale)")
    ap.add_argument("--precision", choices=["hw8", "full"], default="hw8")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write exp_lut.hex / recip_lut.hex ($readmemh) here; with "
                         "--precision full the exp table goes to exp_lut_full.txt "
                         "(decimal floats, not $readmemh); default: print")
    a = ap.parse_args(argv)

    cfg = PRESETS["cimple"].with_(exp_precision=a.precision)
    exp = luts.exp_table(a.score_scale, cfg).tolist()
    rec = luts.recip_lut().tolist()
    if a.precision == "hw8":
        exp_txt = "\n".join(f"{int(v):02x}" for v in exp)
    else:
        exp_txt = "\n".join(f"{float(v):.9g}" for v in exp)
    rec_txt = "\n".join(f"{int(v):02x}" for v in rec)
    if a.out_dir:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        # only the hw8 table is $readmemh hex; the full-precision table is decimal text
        exp_name = "exp_lut.hex" if a.precision == "hw8" else "exp_lut_full.txt"
        (a.out_dir / exp_name).write_text(exp_txt + "\n")
        (a.out_dir / "recip_lut.hex").write_text(rec_txt + "\n")
        print(f"wrote {a.out_dir}/{exp_name} and recip_lut.hex")
    else:
        print(f"# EXP_LUT ({a.precision}), score_scale={a.score_scale}, index = uint8(score)")
        print(exp_txt)
        print("# RECIP_LUT, index = 8 bits below the leading 1 of the exp-sum")
        print(rec_txt)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = ["eval", "luts", *_DELEGATES]
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in cmds:
        print("usage: cimple-sw {" + ",".join(cmds) + "} [args...]\n\n" + (__doc__ or ""),
              file=sys.stderr if argv and argv[0] not in ("-h", "--help") else sys.stdout)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "eval":
        return _eval_main(rest)
    if cmd == "luts":
        return _luts_main(rest)
    return _delegate(_DELEGATES[cmd], rest)


if __name__ == "__main__":
    raise SystemExit(main())
