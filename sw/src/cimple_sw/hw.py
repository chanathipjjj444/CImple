"""CIMple hardware constants, shared by the latency and mapping models.

Geometry and widths come from the RTL golden model
(``cimple_rtl.model.cimple_model``), which in turn mirrors ``rtl/cimple_pkg.sv``
and the paper's Fig. 4. Timing constants that the paper does not state are
marked [A] (assumption) with the reason; the latency model reports how
sensitive its results are to them.
"""
from __future__ import annotations

from dataclasses import dataclass

from cimple_rtl.model import cimple_model as gm

# ---- geometry, from the golden model (single source of truth) --------------
N_ROWS = gm.N_ROWS            # 64 activation inputs (64b input bus, 1 bit each)
N_PART = gm.N_PART            # 32 partitions = 32 output columns
N_BANKS = gm.N_BANKS          # 2 SRAM blocks per partition, one active
DATA_W = gm.DATA_W            # INT8 weights and activations
N_LANES = gm.N_LANES          # CIM OUT is 23b x 4 lanes
MUX_PHASES = gm.MUX_PHASES    # 32 -> 4 output mux under a 3b counter: 8 phases
ACC_W = gm.ACC_W              # intermediate accumulator / buffer width
LUT_DEPTH = gm.LUT_DEPTH      # 256-entry LUTs (8b index)
LUT_W = gm.LUT_W              # 8b LUT outputs in hardware
WRITE_W = 128                 # [P] "write data bit width is 128b"
WEIGHTS_PER_WRITE = WRITE_W // DATA_W   # 16 INT8 weights per write cycle


@dataclass(frozen=True)
class Timing:
    """Cycle costs of the primitive operations, in core clock cycles.

    mac_cycles     [P] 8 cycles per bit-serial INT8 pass ("accumulates partial
                   products over 8 cycles").
    out_cycles     [F] the 3b counter walks 8 phases to move 32 results through
                   the 4-lane output mux. [A] The mux walk of pass i can overlap
                   the MAC of pass i+1 when ``overlap_out`` is True (the 23b
                   partition accumulators would need a shadow register for that;
                   the current RTL does not overlap).
    lut_cycles     [A] one cycle per 4-lane LUT lookup (128b LUT output).
    quant_cycles   [A] one cycle per 4-lane requantisation.
    write_cycles_per_row  [D] one partition row of 64 INT8 weights = 512b over
                   a 128b write port = 4 cycles; a whole bank (32 x 64) = 128.
    freq_mhz       [P] 400 MHz in the latency study (Sec. V-B). The chip's
                   operating range is 417-770 MHz [P] (Table I).
    """

    mac_cycles: int = DATA_W
    out_cycles: int = MUX_PHASES
    overlap_out: bool = False
    lut_cycles: int = 1
    quant_cycles: int = 1
    freq_mhz: float = 400.0

    @property
    def pass_cycles(self) -> int:
        """Cycles between successive activation vectors on the core."""
        if self.overlap_out:
            return max(self.mac_cycles, self.out_cycles)
        return self.mac_cycles + self.out_cycles

    @staticmethod
    def write_cycles(n_weights: int) -> int:
        """Cycles to write ``n_weights`` INT8 weights over the 128b port."""
        return -(-n_weights // WEIGHTS_PER_WRITE)


PEAK_OPS_PER_CYCLE = 2 * N_ROWS * N_PART // DATA_W   # 512 OP/cycle (1 MAC = 2 OP)
