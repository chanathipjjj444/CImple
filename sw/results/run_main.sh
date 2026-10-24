#!/usr/bin/env bash
# Main accuracy study (paper Sec. V-B / Fig. 11). Each preset in its own process;
# rebuild the tables afterwards with: uv run cimple-sw eval --merge --out-dir <dir>
set -u
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV
for p in float cimple; do
  ( uv run cimple-sw eval --presets $p --tasks arc_challenge openbookqa winogrande \
        --threads 11 --out-dir results/main3 > results/logs/main3_$p.log 2>&1
    uv run cimple-sw eval --presets $p --tasks hellaswag --limit 2000 \
        --threads 11 --out-dir results/hellaswag2000 > results/logs/hs2000_$p.log 2>&1 ) &
done
for p in float cimple_hw8 cimple_hw8_rowmax; do
  uv run cimple-sw eval --presets $p --tasks arc_challenge hellaswag openbookqa winogrande \
      --limit 250 --threads 4 --out-dir results/ablation250 > results/logs/abl250_$p.log 2>&1 &
done
wait
for d in main3 hellaswag2000 ablation250; do uv run cimple-sw eval --merge --out-dir results/$d; done
