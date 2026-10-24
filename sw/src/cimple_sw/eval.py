"""lm-evaluation-harness driver for the paper's accuracy study (Sec. V-B, Fig. 11).

Ahn et al. (arXiv:2604.15944v2) report, for INT8 TinyLlama with CIMple's
LUT-based split softmax, ARC-Challenge 31.74, HellaSwag 60.52, OpenBookQA
36.8 and Winogrande 57.22 (%), i.e. +0.6, -0.54, -0.2, -0.32 points versus the
same INT8 model with PyTorch's softmax. :func:`run_eval` reproduces that
comparison: one lm-eval run per softmax preset on the same quantised model,
raw results saved per preset (``<out_dir>/<preset>.json``) and a markdown
summary (``<out_dir>/summary.md``) rebuilt from all JSON files present, so
presets can run in separate processes and be merged afterwards.

[A] The paper does not say which lm-eval metric it plots. Its numbers match
the usual TinyLlama leaderboard convention: acc_norm for ARC-C, HellaSwag and
OBQA, acc for Winogrande; the summary shows both and marks that guess.
[A] Zero-shot (num_fewshot=0), the lm-eval default for these tasks; the paper
does not state the shot count.

Padding: lm-eval right-pads loglikelihood batches with token 0 and calls the
model without a mask. :class:`CimpleHFLM` records each row's true length
(by wrapping lm-eval's ``pad_and_concat``) and hands it to
``hf_attention.sequence_lengths`` so CIMple's dynamic scales never see pad
tokens. Residual batch-size dependence comes only from fp32 BLAS rounding in
the linear layers (~1e-6), which INT8 rounding in the kernel can amplify to
~1e-3 nats per token and so flip near-tie answers; :func:`merge` therefore
flags presets run with different batch sizes.

Baselines: "float" keeps Q/K/V and the attention output in fp32, so
"cimple" - "float" mixes the LUT softmax with the per-(b,h) INT8 attention
quantisation. "float_int8attn" (same INT8 attention, fp32 softmax) is the
matched baseline; when present, :func:`merge` adds deltas against it.

Causality: lm-eval scores a continuation from one forward over
ctx + cont[:-1]. Under dynamic per-sequence INT8 scales (Q/K/V absmax, score
scale) the log-prob of each non-final continuation token depends on the
tokens after it; the float baseline has no such effect.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_TASKS = ("arc_challenge", "hellaswag", "openbookqa", "winogrande")
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "results"
TASK_LABEL = {"arc_challenge": "ARC-Challenge", "hellaswag": "HellaSwag",
              "openbookqa": "OpenBookQA", "winogrande": "Winogrande"}

# [P] Sec. V-B / Fig. 11: CIMple accuracy (%) and delta vs the INT8 baseline (points).
# Baseline = CIMple - delta [D]. Metric per task is our [A] reading (see module doc).
PAPER = {
    "arc_challenge": {"metric": "acc_norm", "cimple": 31.74, "delta": +0.60},
    "hellaswag":     {"metric": "acc_norm", "cimple": 60.52, "delta": -0.54},
    "openbookqa":    {"metric": "acc_norm", "cimple": 36.80, "delta": -0.20},
    "winogrande":    {"metric": "acc",      "cimple": 57.22, "delta": -0.32},
}
METRICS = ("acc", "acc_norm")


# ---------------------------------------------------------------------------
# lm-eval model wrapper
# ---------------------------------------------------------------------------
_LAST_PAD_LENS: Optional[list[int]] = None


@contextmanager
def _record_pad_lengths():
    """Wrap lm-eval's pad_and_concat to remember each row's unpadded length."""
    import lm_eval.models.huggingface as lmhf
    orig = lmhf.pad_and_concat

    def wrapped(max_length, tensors, padding_side="right"):
        global _LAST_PAD_LENS
        _LAST_PAD_LENS = [int(t.shape[-1]) for t in tensors] if padding_side == "right" else None
        return orig(max_length, tensors, padding_side=padding_side)

    lmhf.pad_and_concat = wrapped
    try:
        yield
    finally:
        lmhf.pad_and_concat = orig


def _make_hflm(model, tokenizer, batch_size):
    from lm_eval.models.huggingface import HFLM

    from . import hf_attention as hfa

    class CimpleHFLM(HFLM):
        """HFLM that forwards the true row lengths of right-padded batches."""

        def _model_call(self, inps, attn_mask=None, labels=None):
            lens = _LAST_PAD_LENS
            if not (lens and len(lens) == inps.shape[0] and max(lens) == inps.shape[1]):
                lens = None
            with hfa.sequence_lengths(lens):
                return super()._model_call(inps, attn_mask=attn_mask, labels=labels)

    return CimpleHFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size,
                      backend="causal")


def _count_requests(lm) -> dict:
    """Count requests handed to ``lm`` (loglikelihood / rolling / generate)."""
    box = {"n": 0}
    for name in ("loglikelihood", "loglikelihood_rolling", "generate_until"):
        fn = getattr(lm, name)

        def wrapped(requests, *a, _fn=fn, **k):
            box["n"] += len(requests)
            return _fn(requests, *a, **k)
        setattr(lm, name, wrapped)
    return box


def physical_cores() -> int:
    """Physical core count (SMT siblings excluded); falls back to os.cpu_count()."""
    try:
        cores, phys = set(), None
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    cores.add((phys, line.split(":")[1].strip()))
        if cores:
            return len(cores)
    except OSError:
        pass
    return os.cpu_count() or 1


def set_threads(threads: Optional[int]) -> int:
    """Default: one thread per physical core (measured on the 16C/32T EPYC 8124P:
    a TinyLlama forward takes 2.8 s with 16 threads vs 4.1 s with 32)."""
    import torch
    n = threads or physical_cores()
    torch.set_num_threads(n)
    return n


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------
def eval_preset(preset: str, tasks: Iterable[str] = DEFAULT_TASKS, *, limit=None,
                batch_size: int | str = 8, num_fewshot: int = 0, quant: str = "w8a8",
                model_name: Optional[str] = None, out_dir=DEFAULT_OUT,
                threads: Optional[int] = None, quantize_lm_head: bool = False,
                calibrate: bool = False) -> dict:
    """Load the model with ``preset``, run lm-eval, save ``<out_dir>/<preset>.json``."""
    import lm_eval

    from .model import DEFAULT_MODEL, load_model

    model_name = model_name or DEFAULT_MODEL
    nthr = set_threads(threads)
    t0 = time.time()
    model, tok = load_model(model_name, quant=quant, softmax=preset, calibrate=calibrate,
                            quantize_lm_head=quantize_lm_head)
    t_load = time.time() - t0
    lm = _make_hflm(model, tok, batch_size)
    n_req = _count_requests(lm)
    # one simple_evaluate per task so wall time and request count are per task;
    # lm-eval evaluates tasks independently, so the scores are unchanged.
    res = {"results": {}, "n-samples": {}, "versions": {}}
    per_task = {}
    t1 = time.time()
    with _record_pad_lengths():
        for task in tasks:
            n_req["n"] = 0
            tt = time.time()
            r = lm_eval.simple_evaluate(model=lm, tasks=[task], num_fewshot=num_fewshot,
                                        limit=limit, log_samples=False, verbosity="WARNING")
            dt = time.time() - tt
            for key in res:
                res[key].update(r.get(key) or {})
            nd = sum(int(v.get("effective", 0)) for v in (r.get("n-samples") or {}).values())
            per_task[task] = {"docs": nd, "requests": n_req["n"], "eval_s": round(dt, 1),
                              "req_per_s": round(n_req["n"] / dt, 2) if dt > 0 else None,
                              "s_per_doc": round(dt / nd, 3) if nd else None}
    wall = time.time() - t1
    n = sum(int(v.get("effective", 0)) for v in res.get("n-samples", {}).values())
    cfg = getattr(model, "cimple_cfg", None)
    out = {
        "preset": preset,
        "softmax_config": repr(cfg),
        "score_scale": getattr(cfg, "score_scale", None), "calibrate": bool(calibrate),
        "model": model_name, "quant": quant, "quantize_lm_head": quantize_lm_head,
        "num_fewshot": num_fewshot, "limit": limit, "batch_size": batch_size,
        "threads": nthr, "load_s": round(t_load, 1), "eval_s": round(wall, 1),
        "n_docs": n, "docs_per_s": round(n / wall, 3) if wall > 0 else None,
        "per_task": per_task,
        "results": res["results"], "n-samples": res.get("n-samples"),
        "versions": res.get("versions"),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{preset}.json").write_text(json.dumps(out, indent=2, default=str))
    del model, lm
    return out


def run_eval(presets: Iterable[str] = ("float", "cimple"), tasks: Iterable[str] = DEFAULT_TASKS,
             limit=None, batch_size: int | str = 8, num_fewshot: int = 0, quant: str = "w8a8",
             out_dir=DEFAULT_OUT, model_name: Optional[str] = None,
             threads: Optional[int] = None, **kw) -> str:
    """Evaluate each preset in turn, then rebuild and return summary.md."""
    for p in presets:
        r = eval_preset(p, tasks, limit=limit, batch_size=batch_size, num_fewshot=num_fewshot,
                        quant=quant, model_name=model_name, out_dir=out_dir, threads=threads,
                        **kw)
        print(f"[{p}] {r['n_docs']} docs in {r['eval_s']} s ({r['docs_per_s']} docs/s), "
              f"load {r['load_s']} s", flush=True)
    return merge(out_dir)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _metric(task_res: dict, m: str):
    v = task_res.get(f"{m},none")
    se = task_res.get(f"{m}_stderr,none")
    if v is None:
        return None
    return 100.0 * float(v), (100.0 * float(se) if isinstance(se, (int, float)) else None)


BASELINES = ("float", "float_int8attn")
MATCHED_BASELINE = "float_int8attn"


def _settings(r: dict) -> tuple:
    """Run settings that must match across presets for a delta to be meaningful.

    ``calibrate`` is absent from JSON written before it was recorded; it was a
    no-op then, so it defaults to False.
    """
    return (r["model"], r["quant"], r["num_fewshot"], r["limit"],
            bool(r.get("quantize_lm_head", False)), str(r.get("batch_size")),
            bool(r.get("calibrate", False)))


def merge(out_dir=DEFAULT_OUT) -> str:
    """Rebuild ``summary.md`` from every ``<preset>.json`` in ``out_dir``."""
    out_dir = Path(out_dir)
    runs = {}
    for f in sorted(out_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "results" in d and "preset" in d:
            runs[d["preset"]] = d
    if not runs:
        raise FileNotFoundError(f"no per-preset result JSON in {out_dir}")
    order = sorted(runs, key=lambda p: (p not in BASELINES, BASELINES.index(p)
                                        if p in BASELINES else 0, p))
    tasks = [t for t in DEFAULT_TASKS if any(t in r["results"] for r in runs.values())]
    tasks += sorted({t for r in runs.values() for t in r["results"]} - set(tasks))
    base = runs.get("float")
    matched = MATCHED_BASELINE if MATCHED_BASELINE in runs else None
    vs_matched = [p for p in order if p not in BASELINES] if matched else []

    lines = ["# CIMple softmax accuracy study (Sec. V-B, Fig. 11)", ""]
    settings = {_settings(r) for r in runs.values()}
    for mdl, q, fs, lim, lmh, bs, cal in sorted(settings, key=str):
        lines.append(f"- model `{mdl}`, quant `{q}`, {fs}-shot, limit `{lim}`, "
                     f"quantize_lm_head `{lmh}`, batch `{bs}`, calibrate `{cal}`")
    if len(settings) > 1:
        lines.append("- **WARNING: presets were run with different settings; deltas are not comparable**")
    lines += ["- values in %, +- = lm-eval stderr; delta = preset - float (points)",
              "- `float` keeps Q/K/V and the attention output in fp32, so `delta <preset>` "
              "includes the per-(batch, head) INT8 attention quantisation, not only the "
              "softmax" + (f"; `vs {matched}` = preset - {matched} (same INT8 attention, "
                           "fp32 softmax) isolates the softmax" if matched else
                           f" (run `{MATCHED_BASELINE}` for a softmax-only delta)"),
              "- dynamic per-sequence INT8 scales make the INT8-attention presets' "
              "continuation log-probs depend on later tokens (not strictly causal)",
              "- paper column: [A] the paper's metric is assumed to be acc_norm for "
              "ARC-C/HellaSwag/OBQA and acc for Winogrande (marked `*`)", ""]
    hdr = ["task", "metric"] + [p for p in order]
    hdr += [f"delta {p}" for p in order if p != "float" and base]
    hdr += [f"vs {matched} {p}" for p in vs_matched]
    hdr += ["paper CIMple", "paper delta"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for t in tasks:
        for m in METRICS:
            vals = {p: _metric(r["results"].get(t, {}), m) for p, r in runs.items()}
            if all(v is None for v in vals.values()):
                continue
            row = [TASK_LABEL.get(t, t), m]
            for p in order:
                v = vals[p]
                row.append("-" if v is None else
                           f"{v[0]:.2f}" + (f" +- {v[1]:.2f}" if v[1] is not None else ""))
            if base:
                b = vals.get("float")
                for p in order:
                    if p == "float":
                        continue
                    v = vals[p]
                    row.append("-" if v is None or b is None else f"{v[0] - b[0]:+.2f}")
            mb = vals.get(matched) if matched else None
            for p in vs_matched:
                v = vals[p]
                row.append("-" if v is None or mb is None else f"{v[0] - mb[0]:+.2f}")
            pp = PAPER.get(t)
            if pp and pp["metric"] == m:
                row += [f"{pp['cimple']:.2f}*", f"{pp['delta']:+.2f}*"]
            else:
                row += ["", ""]
            lines.append("| " + " | ".join(row) + " |")
    lines += ["", "| preset | docs | eval s | docs/s | threads | batch | softmax config |",
              "|---|---|---|---|---|---|---|"]
    for p in order:
        r = runs[p]
        lines.append(f"| {p} | {r['n_docs']} | {r['eval_s']} | {r['docs_per_s']} | "
                     f"{r['threads']} | {r['batch_size']} | `{r['softmax_config']}` |")
    timed = [p for p in order if runs[p].get("per_task")]
    if timed:
        lines += ["", "| preset | task | docs | requests | eval s | requests/s | s/doc |",
                  "|---|---|---|---|---|---|---|"]
        for p in timed:
            for t, v in runs[p]["per_task"].items():
                lines.append(f"| {p} | {TASK_LABEL.get(t, t)} | {v['docs']} | {v['requests']} | "
                             f"{v['eval_s']} | {v['req_per_s']} | {v['s_per_doc']} |")
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text)
    return text
