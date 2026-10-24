// ---------------------------------------------------------------------------
// cpl_partition.sv -- one of the 32 CIM partitions (VP1202 version)
//
// Starts from src/cimple_rtl/rtl/cim_partition.sv (commit d195ccd). Paper:
// Ahn et al., arXiv:2604.15944v2, Sec. IV-A, Fig. 4.
//
// 64 macros -> two adder trees (MSB nibbles, LSB nibbles) -> "<<4" combine
// -> "<<1 & ACC" bit-serial shift-accumulator [F].
//
// Activations arrive MSB-FIRST, one bit per row per cycle, 8 cycles per
// INT8 x INT8 dot product [P] ("accumulates partial products over 8 cycles").
// Signed activations (act_unsigned_i = 0): cycle 0 carries the sign bit, so
// its partial sum is SUBTRACTED [D]:
//     x = -x7*2^7 + x6*2^6 + ... + x0
//   acc <- -P7 ;  acc <- 2*acc + P6 ; ... ; acc <- 2*acc + P0
//
// Changes vs the original (fpga/docs/ARCH.md §2):
//   1. 128b write port [P]: wdata_i carries 16 INT8 weights; byte i goes to
//      row 16*wgrp_i + i of bank wbank_i, i.e. 16 macros are written per
//      cycle and a full partition takes 4 cycles. The original wrote 8b/cycle.
//   2. act_unsigned_i [A]: when 1 the first (MSB) cycle ADDS instead of
//      subtracting (acc <- +P7), so the activations are UQ8 (0..255). Range:
//      |acc| <= 64 * 255 * 128 = 2,088,960 < 2^22, fits MAC_W = 23 signed.
//      Sampled on every mac_en cycle; it must be constant during a pass
//      (only the mac_first cycle actually uses it).
//   3. PIPE_TREE [A] (timing on Versal at 200 MHz), 0..2 pipeline stages
//      between the bits and the shift-accumulator:
//        0  the original single-stage path (OAI -> trees -> combine -> acc);
//        1  comb = (sum_msb << 4) + sum_lsb is registered together with
//           mac_en / mac_first / act_unsigned;
//        2  (default) additionally the 2 x 8 level-3 tree nodes (7b each)
//           are registered, so neither the broadcast + OAI + 3 adder levels
//           nor the 3 remaining levels + combine share a cycle with
//           anything else (the PIPE_TREE = 1 path was ~14 LUT levels).
//      Each stage delays the accumulator update by one cycle.
//   4. [A] The two adder trees are written as explicit balanced binary trees
//      (6 levels, width NIB_W + level) instead of a for-loop reduction, so the
//      logic depth does not depend on the synthesis tool re-balancing a
//      63-adder chain.
//
// TIMING (what the driver of this block must know):
//   Present bit k (k = 7 first) on x_bits_i with mac_en_i = 1 for 8 cycles,
//   mac_first_i = 1 on the first. If the mac_first cycle is cycle c0, the
//   final dot product is on acc_o from cycle c0 + MAC_LAT on, where
//       MAC_LAT = 8 + PIPE_TREE.
//   acc_o has no reset (the mac_first update overwrites it); its value
//   after reset and before the first pass is undefined [A].
//   It holds until the next pass's first update: a new mac_first presented in
//   cycle n changes acc_o from cycle n + 1 + PIPE_TREE. mac_en_i may drop
//   (stall) in the middle of a pass; the accumulator and the pipeline stage
//   hold. bank_sel_i must be stable during a pass (the OAIs read it
//   combinationally). Writes may happen during a MAC on the other bank; a
//   write to the bank being computed on takes effect on the next edge.
// ---------------------------------------------------------------------------
module cpl_partition
  import cpl_pkg::*;
#(
  parameter int unsigned PIPE_TREE = 2
) (
  input  logic                    clk_i,
  /* verilator lint_off UNUSEDSIGNAL */  // unused when PIPE_TREE = 0
  input  logic                    rst_ni,
  /* verilator lint_on UNUSEDSIGNAL */
  // --- write port: 16 weights / cycle ----------------------------------------
  input  logic                    we_i,
  input  logic                    wbank_i,
  input  logic [WGRP_W-1:0]       wgrp_i,       // row group: rows 16*wgrp_i ..+15
  input  logic [WR_W-1:0]         wdata_i,      // byte i -> row 16*wgrp_i + i
  // --- compute port --------------------------------------------------------
  input  logic                    bank_sel_i,
  input  logic [N_ROWS-1:0]       x_bits_i,     // one bit per row (XIN), this cycle
  input  logic                    mac_en_i,     // advance the shift-accumulator
  input  logic                    mac_first_i,  // this cycle carries the MSB
  input  logic                    act_unsigned_i, // 1: UQ8 activations (MSB adds)
  output logic signed [MAC_W-1:0] acc_o
);

  // Documentation / for instantiating logic: cycles from the mac_first cycle
  // until the final result is visible on acc_o.
  /* verilator lint_off UNUSEDPARAM */  // informative constant, read by humans / tests
  localparam int unsigned MAC_LAT = DATA_W + PIPE_TREE;
  /* verilator lint_on UNUSEDPARAM */

  if (PIPE_TREE > 2) begin : g_bad_pipe
    $error("cpl_partition: PIPE_TREE must be 0, 1 or 2");
  end

  // --- the 64 macros (one INT8 weight per bank each) ------------------------
  logic signed [NIB_W-1:0] p_msb [N_ROWS];
  logic        [NIB_W-1:0] p_lsb [N_ROWS];

  for (genvar r = 0; r < N_ROWS; r++) begin : g_macro
    localparam int unsigned GRP  = r / WR_BYTES;   // row group of this row
    localparam int unsigned BYTE = r % WR_BYTES;   // byte lane inside wdata_i
    cpl_macro u_macro (
      .clk_i,
      .we_i       (we_i && (wgrp_i == WGRP_W'(GRP))),
      .wbank_i,
      .wdata_i    (wdata_i[BYTE*DATA_W +: DATA_W]),
      .bank_sel_i,
      .x_bit_i    (x_bits_i[r]),
      .p_msb_o    (p_msb[r]),
      .p_lsb_o    (p_lsb[r])
    );
  end

  // --- adder trees [F] -------------------------------------------------------
  // Balanced binary trees, level lv has N_ROWS >> lv nodes of NIB_W + lv bits.
  // Range checks (no overflow at any level):
  //   MSB tree: 2^lv * [-8, 7] fits NIB_W+lv bits signed; root 64 * [-8, 7]
  //             -> [-512, 448] -> 10b signed   ("10b x 32")
  //   LSB tree: 2^lv * 15 < 2^(NIB_W+lv);    root 64 * [0, 15]
  //             -> [0, 960]    -> 10b unsigned ("10b x 32")
  // Node (lv, i) is kept sign- (MSB tree) / zero- (LSB tree) extended to
  // TREE_W bits in mt / ut; the add of level lv is NIB_W + lv bits wide (the
  // extension bits are copies of the sign / zeros, so synthesis drops them).
  // mr / ur = the node after the optional PIPE_TREE = 2 register of level
  // CUT_LVL (= mt / ut elsewhere). [A] Plain 2-D arrays instead of
  // per-level generate arrays reached through g_lv[lv-1]: sv2v mistranslates
  // that cross-iteration reference (it uses the current level's width).
  localparam int unsigned N_LVL   = $clog2(N_ROWS);   // 6
  localparam int unsigned CUT_LVL = 3;
  localparam bit          CUT     = (PIPE_TREE >= 2);

  // control aligned with the (optionally cut) tree output
  logic en_c, first_c, uns_c;

  /* verilator lint_off UNUSEDSIGNAL */  // entries i >= N_ROWS >> lv are unused (tied to 0)
  logic signed [TREE_W-1:0] mt [N_LVL+1][N_ROWS];
  logic        [TREE_W-1:0] ut [N_LVL+1][N_ROWS];
  logic signed [TREE_W-1:0] mr [N_LVL+1][N_ROWS];
  logic        [TREE_W-1:0] ur [N_LVL+1][N_ROWS];
  /* verilator lint_on UNUSEDSIGNAL */

  for (genvar lv = 0; lv <= N_LVL; lv++) begin : g_lv
    localparam int unsigned W = NIB_W + lv;
    localparam int unsigned N = N_ROWS >> lv;
    for (genvar i = 0; i < N_ROWS; i++) begin : g_n
      if (i >= N) begin : g_unused
        assign mt[lv][i] = '0;
        assign ut[lv][i] = '0;
        assign mr[lv][i] = '0;
        assign ur[lv][i] = '0;
      end else begin : g_node
        if (lv == 0) begin : g_leaf
          assign mt[lv][i] = TREE_W'(p_msb[i]);                 // sign-extend
          assign ut[lv][i] = TREE_W'(p_lsb[i]);                 // zero-extend
        end else begin : g_add
          logic signed [W-1:0] ms;
          logic        [W-1:0] us;
          // W-bit add of two (W-1)-bit values, each extended by one bit
          assign ms = W'(mr[lv-1][2*i]) + W'(mr[lv-1][2*i+1]);
          assign us = W'(ur[lv-1][2*i]) + W'(ur[lv-1][2*i+1]);
          assign mt[lv][i] = TREE_W'(ms);
          assign ut[lv][i] = TREE_W'(us);
        end
        if (CUT && lv == CUT_LVL) begin : g_cut
          // tree stage register (no reset: only read when en_c); holds when idle
          logic signed [W-1:0] mq;
          logic        [W-1:0] uq;
          always_ff @(posedge clk_i) begin
            if (mac_en_i) begin
              mq <= W'(mt[lv][i]);
              uq <= W'(ut[lv][i]);
            end
          end
          assign mr[lv][i] = TREE_W'(mq);
          assign ur[lv][i] = TREE_W'(uq);
        end else begin : g_thru
          assign mr[lv][i] = mt[lv][i];
          assign ur[lv][i] = ut[lv][i];
        end
      end
    end
  end

  if (CUT) begin : g_cut_ctl
    always_ff @(posedge clk_i) begin
      if (!rst_ni) begin
        en_c    <= 1'b0;
        first_c <= 1'b0;
        uns_c   <= 1'b0;
      end else begin
        en_c    <= mac_en_i;
        first_c <= mac_first_i;
        uns_c   <= act_unsigned_i;
      end
    end
  end else begin : g_nocut_ctl
    assign en_c    = mac_en_i;
    assign first_c = mac_first_i;
    assign uns_c   = act_unsigned_i;
  end

  logic signed [TREE_W-1:0] sum_msb;
  logic        [TREE_W-1:0] sum_lsb;
  assign sum_msb = mr[N_LVL][0];
  assign sum_lsb = ur[N_LVL][0];

  // --- "<<4" combine [F] -------------------------------------------------------
  //   (sum_msb << 4) + sum_lsb  ->  [-8192, 8128] -> 15b signed ("15b x 32")
  logic signed [COMB_W-1:0] comb_val;
  assign comb_val = (COMB_W'(sum_msb) <<< NIB_W) + COMB_W'($signed({1'b0, sum_lsb}));

  // --- optional pipeline stage after the trees [A] -----------------------------
  logic signed [COMB_W-1:0] comb_s;
  logic                     en_s, first_s, uns_s;

  if (PIPE_TREE >= 1) begin : g_pipe
    logic signed [COMB_W-1:0] comb_q;
    logic                     en_q, first_q, uns_q;
    // Control bits have a synchronous reset (rst_ni comes from
    // proc_sys_reset, synchronous to clk_i); the data register needs none
    // (only read when en_q).
    always_ff @(posedge clk_i) begin
      if (!rst_ni) begin
        en_q    <= 1'b0;
        first_q <= 1'b0;
        uns_q   <= 1'b0;
      end else begin
        en_q    <= en_c;
        first_q <= first_c;
        uns_q   <= uns_c;
      end
    end
    // Hold the stage when idle so a stalled pass does not toggle 15b x 32.
    always_ff @(posedge clk_i) begin
      if (en_c) comb_q <= comb_val;
    end
    assign comb_s  = comb_q;
    assign en_s    = en_q;
    assign first_s = first_q;
    assign uns_s   = uns_q;
  end else begin : g_nopipe
    assign comb_s  = comb_val;
    assign en_s    = en_c;
    assign first_s = first_c;
    assign uns_s   = uns_c;
  end

  // --- "<<1 & ACC" bit-serial shift-accumulator [F] ----------------------------
  logic signed [MAC_W-1:0] comb_ext, acc_q;
  assign comb_ext = MAC_W'(comb_s);

  // no reset [A]: the mac_first update overwrites it (frees DSP / retiming)
  always_ff @(posedge clk_i) begin
    if (en_s) begin
      if (first_s) acc_q <= uns_s ? comb_ext : -comb_ext;
      else         acc_q <= (acc_q <<< 1) + comb_ext;
    end
  end

  assign acc_o = acc_q;

endmodule
