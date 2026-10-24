// ---------------------------------------------------------------------------
// cpl_core.sv -- 32 kb CIM core: 32 partitions + 32->4 output select
//
// Starts from src/cimple_rtl/rtl/cim_core.sv (commit d195ccd). Paper: Ahn et
// al., arXiv:2604.15944v2, Sec. IV-A, Fig. 4; contract fpga/docs/ARCH.md §2.
//
// One 64-wide activation vector is broadcast bit-serially to all 32
// partitions; each partition holds a different 64-weight column per bank, so
// one 8-cycle pass produces 32 dot products of length 64 (= 2048 MACs / pass)
// [P].
//
// The 32 results leave through 4 lanes ("MUX 32x8", "23b x 4" in Fig. 4) [F]:
// lane l in phase k shows partition 8*l + k. Unlike the original, the phase
// is an input driven by the engine (out_phase_i) instead of a free-running
// 3b counter inside the core [A] (ARCH.md §2), and the mux is combinational
// from the accumulator registers.
//
// Changes vs cim_core: 128b write port (wpart_i + wgrp_i select 16 rows of
// one partition, 16 weights per cycle) [P]; act_unsigned_i (UQ8 activations,
// MSB cycle adds) [A]; PIPE_TREE (0..2) pipeline stages in the adder trees [A].
// See cpl_partition.sv for the details.
//
// TIMING:
//   MAC_LAT = 8 + PIPE_TREE. Drive x_bits_i (bit k of every row's activation,
//   k = 7 first) with mac_en_i = 1 for 8 cycles, mac_first_i = 1 on the
//   first (cycle c0). The 32 results are on cim_out_o (selected by
//   out_phase_i, combinationally) from cycle c0 + MAC_LAT on, and hold until
//   the next pass's first accumulator update: a mac_first in cycle n changes
//   the accumulators from cycle n + 1 + PIPE_TREE. So a drain of 8 phases
//   must finish before (or overlap at most 1 + PIPE_TREE cycles with) the next
//   pass. bank_sel_i and act_unsigned_i must be stable during a pass.
//   Writes (we_i) may be issued in any cycle, including during a pass on the
//   other bank.
//
// [A] Fanout: bank_sel_i reaches 16k OAI LUTs and each x_bits_i bit 256 LUTs.
//     Drive both straight from flip-flops in the engine; Vivado replicates
//     the drivers (phys_opt_design) as needed.
// ---------------------------------------------------------------------------
module cpl_core
  import cpl_pkg::*;
#(
  parameter int unsigned PIPE_TREE = 2           // 0..2 stages, cpl_partition.sv
) (
  input  logic                     clk_i,
  input  logic                     rst_ni,
  // --- weight write port (128b = 16 INT8 weights per cycle) ----------------
  input  logic                     we_i,
  input  logic                     wbank_i,
  input  logic [PART_AW-1:0]       wpart_i,
  input  logic [WGRP_W-1:0]        wgrp_i,
  input  logic [WR_W-1:0]          wdata_i,
  // --- compute -------------------------------------------------------------
  input  logic                     bank_sel_i,
  input  logic [N_ROWS-1:0]        x_bits_i,
  input  logic                     mac_en_i,
  input  logic                     mac_first_i,
  input  logic                     act_unsigned_i,
  // --- output select -------------------------------------------------------
  input  logic [CNT_W-1:0]         out_phase_i,
  output logic signed [MAC_W-1:0]  cim_out_o [N_LANES]
);

  /* verilator lint_off UNUSEDPARAM */  // informative constant, read by humans / tests
  localparam int unsigned MAC_LAT = DATA_W + PIPE_TREE;
  /* verilator lint_on UNUSEDPARAM */

  logic signed [MAC_W-1:0] acc [N_PART];

  for (genvar p = 0; p < N_PART; p++) begin : g_part
    cpl_partition #(
      .PIPE_TREE (PIPE_TREE)
    ) u_part (
      .clk_i,
      .rst_ni,
      .we_i           (we_i && (wpart_i == PART_AW'(p))),
      .wbank_i,
      .wgrp_i,
      .wdata_i,
      .bank_sel_i,
      .x_bits_i,
      .mac_en_i,
      .mac_first_i,
      .act_unsigned_i,
      .acc_o          (acc[p])
    );
  end

  // lane l, phase k  ->  partition l*MUX_PHASES + k   [F]
  for (genvar l = 0; l < N_LANES; l++) begin : g_lane
    assign cim_out_o[l] = acc[l*MUX_PHASES + int'(out_phase_i)];
  end

endmodule
