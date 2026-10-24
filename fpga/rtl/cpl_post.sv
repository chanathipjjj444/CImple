// ---------------------------------------------------------------------------
// cpl_post.sv -- everything after CIM OUT: ACC buffer, quantiser, split softmax
//
// Paper: Ahn et al., arXiv:2604.15944v2, Sec. IV-A/B, Fig. 4 (intermediate
// accumulator + buffer, "Quantization 32b to 8b", softmax module) and Fig. 7
// (blue: buffer + accumulation, purple: 1/x LUT + quantiser, red: EXP LUT).
// Contract: fpga/docs/ARCH.md §5 MATVEC steps 3-4 (normative), §2.
//
// The engine drains the CIM core one 4-lane group per cycle (lane l, phase k
// = column p = 8*l + k [F]) together with the token index t. Per group:
//
//   e  = (acc_base + 8*t + phase) mod ACC_DEPTH  -- one 128b ACC entry holds
//        the 4 lanes of a group, so a group is one BRAM read + one write [A]
//   v  = NONE: acc | FIRST: A[e] = acc | ADD: A[e] = wrap32(A[e] + acc) [P][A]
//   Q8    out = quantise(v, scale, shift)                          [P]
//   EXP   z = quantise(v, scale, shift); ev = causal-masked EXPLUT[u8(z)];
//         S[i] += ev, i = sum_by_t ? t mod 32 : p                  [P][F][A]
//   NORM  s = S[i]; out = quantise(v, RECIP[mantissa8(s)], 7 + msb(s))
//         (shift = RECIP_FRAC - LUT_W + msb, as gm.softmax_row)   [P][D]
//   RAW32 out = v                                                   [A]
// Columns p >= ncols produce no output (out_lane_en_o = 0) and no exp-sum
// update, but the ACC buffer is updated for all 4 lanes (ARCH.md §5 step 3).
//
// PIPELINE (stage n = cycle n after the input cycle 0), POST_LAT = 13 [A]:
//   0  entry address (16b add)                              -> s1
//   1  ACC BRAM read (ADD only); lane enables, causal mask;
//      exp-sum select S[i] per lane (NORM)                  -> s2
//   2  BRAM output register (DOB_REG)                       -> s3
//   3  BRAM data: v = acc / rd + acc (32b add); msb(s)      -> s4
//   4  ACC BRAM write (FIRST/ADD); mantissa8(s)             -> s5
//   5  recip ROM read (1 cycle)                              -> s6
//   6..10 cpl_quant, LAT = 5 (input regs, DSP mult, round add, shift,
//      sat)                                                 -> q in 11
//   11 exp LUT read (1 cycle)                               -> s12
//   12 causal mask, output select, per-lane exp values      -> s13 = outputs
//   13 outputs valid; exp-sums updated at the end of this cycle
// An input in cycle n appears on out_* in cycle n + POST_LAT, in input order,
// with any gaps preserved. out_busy_o = OR of all stage valids (a registered
// OR would lag by a cycle), so it is 1 from cycle n+1 to n+POST_LAT and
// the exp-sum update of the last group has landed when it falls.
//
// HAZARDS [A]: the ACC entry is read in stage 1 and written at the end of
// stage 4, so two groups with the same entry must be at least 4 groups
// apart -- within one instruction entries repeat only after ACC_DEPTH/8
// tokens, and the engine waits for out_busy_o = 0 between instructions
// (ARCH.md / engine contract), so no forwarding is needed. The exp-sums are
// flip-flops updated in a single cycle (S <= S + delta), so consecutive
// groups hitting the same sum register (sum_by_t, or the same phase) chain
// correctly without forwarding. NORM reads the sums in stage 1; they must
// have been written by an earlier instruction.
//
// sum_clr_i clears all 32 sums and wins over a same-cycle update [A] (the
// engine only pulses it while the pipeline is empty).
// Configuration inputs are used in several stages and must be held from the
// first in_valid_i until out_busy_o falls (engine contract).
// ACC_DEPTH must be a power of two >= 8 so "mod ACC_DEPTH" is a bit select [A].
// ---------------------------------------------------------------------------
module cpl_post
  import cpl_pkg::*;
#(
  parameter int unsigned ACC_DEPTH = 1024,
  parameter bit          RND_FIX   = 1'b0
)(
  input  logic                         clk_i,
  input  logic                         rst_ni,

  // ---- per-instruction configuration ---------------------------------------
  input  logic [1:0]                   acc_mode_i,
  input  logic [2:0]                   out_mode_i,
  input  logic [15:0]                  scale_i,
  input  logic [4:0]                   shift_i,
  input  logic [5:0]                   ncols_i,
  input  logic                         sum_by_t_i,
  input  logic                         causal_i,
  input  logic [15:0]                  qbase_i,
  input  logic [15:0]                  kbase_i,
  /* verilator lint_off UNUSEDSIGNAL */  // bits >= log2(ACC_DEPTH) vanish in the mod
  input  logic [15:0]                  acc_base_i,
  /* verilator lint_on UNUSEDSIGNAL */

  input  logic                         sum_clr_i,

  // ---- exp LUT load (LUTLOAD) -----------------------------------------------
  input  logic                         lut_we_i,
  input  logic [3:0]                   lut_waddr_i,
  input  logic [WR_W-1:0]              lut_wdata_i,

  // ---- drain stream from the CIM core ----------------------------------------
  input  logic                         in_valid_i,
  input  logic signed [MAC_W-1:0]      in_cim_i [N_LANES],
  input  logic [CNT_W-1:0]             in_phase_i,
  input  logic [15:0]                  in_t_i,

  // ---- results ---------------------------------------------------------------
  output logic                         out_valid_o,
  output logic [15:0]                  out_t_o,
  output logic [CNT_W-1:0]             out_phase_o,
  output logic [N_LANES-1:0]           out_lane_en_o,
  output logic [N_LANES-1:0][31:0]     out_data_o,
  output logic                         out_busy_o
);

  localparam int unsigned POST_LAT = 13;
  localparam int unsigned QLAT     = 5;     // cpl_quant pipeline depth
  localparam int unsigned AW       = $clog2(ACC_DEPTH);
  localparam int unsigned MSB_W    = $clog2(SUM_W);        // 5
  localparam int unsigned LANE_AW  = $clog2(N_LANES);      // 2
  localparam int unsigned ENT_W    = N_LANES * ACC_W;      // 128b ACC entry

  if (ACC_DEPTH < 8 || (ACC_DEPTH & (ACC_DEPTH - 1)) != 0) begin : g_bad_depth
    $error("cpl_post: ACC_DEPTH must be a power of two >= 8");
  end

  // ===========================================================================
  // valid / ordering pipeline (the only registers with a reset)
  // ===========================================================================
  logic [POST_LAT:1] vld_q;

  // synchronous reset (rst_ni comes from proc_sys_reset, synchronous to clk_i)
  always_ff @(posedge clk_i) begin
    if (!rst_ni) vld_q <= '0;
    else         vld_q <= {vld_q[POST_LAT-1:1], in_valid_i};
  end

  assign out_busy_o  = |vld_q;
  assign out_valid_o = vld_q[POST_LAT];

  // t and phase ride along unchanged (SRL-friendly: no reset, no enable)
  logic [15:0]      t_q     [1:POST_LAT];
  logic [CNT_W-1:0] phase_q [1:POST_LAT];

  always_ff @(posedge clk_i) begin
    t_q[1]     <= in_t_i;
    phase_q[1] <= in_phase_i;
    for (int n = 2; n <= POST_LAT; n++) begin
      t_q[n]     <= t_q[n-1];
      phase_q[n] <= phase_q[n-1];
    end
  end

  assign out_t_o     = t_q[POST_LAT];
  assign out_phase_o = phase_q[POST_LAT];

  // ===========================================================================
  // stage 0: ACC entry address
  // ===========================================================================
  // (acc_base + 8*t + phase) mod 2^AW: only the low AW bits of each term matter
  logic [AW-1:0] addr0;

  assign addr0 = AW'(acc_base_i) + AW'({in_t_i, 3'b000}) + AW'(in_phase_i);

  logic        [AW-1:0]    s1_addr;
  logic signed [MAC_W-1:0] s1_cim [N_LANES];

  always_ff @(posedge clk_i) begin
    s1_addr <= addr0;
    s1_cim  <= in_cim_i;
  end

  // ===========================================================================
  // exp-sums S[0..31] (24b, wrap mod 2^24)
  // ===========================================================================
  logic [SUM_W-1:0] sums_q [N_SUMS];

  // ===========================================================================
  // stage 1: BRAM read, lane enables, causal mask, exp-sum select
  // ===========================================================================
  logic [ENT_W-1:0] acc_mem [ACC_DEPTH];
  logic [ENT_W-1:0] acc_rd;              // valid in stage 2
  logic [ENT_W-1:0] acc_rd_q;            // valid in stage 3 (BRAM DOB_REG)

  initial begin   // power-up content (BRAM INIT); the ISA never relies on it
    for (int e = 0; e < ACC_DEPTH; e++) acc_mem[e] = '0;
  end

  always_ff @(posedge clk_i) begin
    if (vld_q[1] && acc_mode_i == ACC_ADD) acc_rd <= acc_mem[s1_addr];
  end
  always_ff @(posedge clk_i) acc_rd_q <= acc_rd;

  logic [N_LANES-1:0] en1, mask1;
  logic [SUM_W-1:0]   ssel1 [N_LANES];
  logic [SUM_W-1:0]   ssum_t1;           // S[t mod 32], shared by all lanes

  assign ssum_t1 = sums_q[t_q[1][4:0]];

  always_comb begin
    for (int l = 0; l < N_LANES; l++) begin
      logic [4:0]  p;
      logic [16:0] q, k;
      p = {LANE_AW'(l), phase_q[1]};
      // column p has an output to write
      en1[l] = (out_mode_i != OUT_NONE) && ({1'b0, p} < ncols_i);
      // causal mask, on unbounded (17b) integers
      if (sum_by_t_i) begin
        q = 17'(qbase_i) + 17'(t_q[1]);
        k = 17'(kbase_i) + 17'(p);
      end else begin
        q = 17'(qbase_i) + 17'(p);
        k = 17'(kbase_i) + 17'(t_q[1]);
      end
      mask1[l] = causal_i && (k > q);
      // NORM denominator: i = sum_by_t ? t mod 32 : p (lane l owns 8l..8l+7)
      ssel1[l] = sum_by_t_i ? ssum_t1 : sums_q[p];
    end
  end

  logic        [AW-1:0]    s2_addr, s3_addr;
  logic signed [MAC_W-1:0] s2_cim [N_LANES];
  logic signed [MAC_W-1:0] s3_cim [N_LANES];
  logic [N_LANES-1:0]      en_q   [2:POST_LAT];
  logic [N_LANES-1:0]      mask_q [2:POST_LAT-1];
  logic [SUM_W-1:0]        s2_sum [N_LANES];
  logic [SUM_W-1:0]        s3_sum [N_LANES];

  always_ff @(posedge clk_i) begin
    s2_addr   <= s1_addr;
    s2_cim    <= s1_cim;
    s2_sum    <= ssel1;
    s3_addr   <= s2_addr;                // stage 2: BRAM output register
    s3_cim    <= s2_cim;
    s3_sum    <= s2_sum;
    en_q[2]   <= en1;
    mask_q[2] <= mask1;
    for (int n = 3; n <= POST_LAT; n++)   en_q[n]   <= en_q[n-1];
    for (int n = 3; n <= POST_LAT-1; n++) mask_q[n] <= mask_q[n-1];
  end

  assign out_lane_en_o = en_q[POST_LAT];

  // ===========================================================================
  // stage 3: accumulate, msb of the exp-sum
  // ===========================================================================
  logic signed [ACC_W-1:0] v3 [N_LANES];
  logic [MSB_W-1:0]        msb3 [N_LANES];

  always_comb begin
    for (int l = 0; l < N_LANES; l++) begin
      if (acc_mode_i == ACC_ADD)
        v3[l] = $signed(acc_rd_q[ACC_W*l +: ACC_W]) + ACC_W'(s3_cim[l]);   // wrap32
      else
        v3[l] = ACC_W'(s3_cim[l]);                                          // NONE / FIRST
      msb3[l] = '0;
      for (int b = 0; b < SUM_W; b++)
        if (s3_sum[l][b]) msb3[l] = MSB_W'(b);
    end
  end

  logic [AW-1:0]          s4_addr;
  logic signed [ACC_W-1:0] v_q [4:POST_LAT-1][N_LANES];
  logic [SUM_W-1:0]       s4_sum [N_LANES];
  logic [MSB_W-1:0]       msb_q  [4:6][N_LANES];

  always_ff @(posedge clk_i) begin
    s4_addr  <= s3_addr;
    s4_sum   <= s3_sum;
    v_q[4]   <= v3;
    msb_q[4] <= msb3;
    for (int n = 5; n <= POST_LAT-1; n++) v_q[n]   <= v_q[n-1];
    for (int n = 5; n <= 6; n++)          msb_q[n] <= msb_q[n-1];
  end

  // ===========================================================================
  // stage 4: ACC write-back, mantissa8 of the exp-sum
  // ===========================================================================
  logic [ENT_W-1:0] wdata4;

  always_comb begin
    for (int l = 0; l < N_LANES; l++) wdata4[ACC_W*l +: ACC_W] = v_q[4][l];
  end

  always_ff @(posedge clk_i) begin
    if (vld_q[4] && (acc_mode_i == ACC_FIRST || acc_mode_i == ACC_ADD))
      acc_mem[s4_addr] <= wdata4;
  end

  logic [LUT_W-1:0] mant4 [N_LANES];

  always_comb begin
    for (int l = 0; l < N_LANES; l++) begin
      // gm.mantissa8: the 8 bits below the leading one
      if (msb_q[4][l] >= MSB_W'(LUT_W))
        mant4[l] = LUT_W'(s4_sum[l] >> (msb_q[4][l] - MSB_W'(LUT_W)));
      else
        mant4[l] = LUT_W'(s4_sum[l] << (MSB_W'(LUT_W) - msb_q[4][l]));
    end
  end

  logic [LUT_W-1:0] s5_mant [N_LANES];

  always_ff @(posedge clk_i) s5_mant <= mant4;

  // ===========================================================================
  // stage 5: reciprocal ROM read; stage 11: exp LUT read (cpl_lut, 1 cycle)
  // ===========================================================================
  logic [LUT_W-1:0] recip6 [N_LANES];
  logic [LUT_W-1:0] q11    [N_LANES];   // quantiser output, valid in stage 11
  logic [LUT_W-1:0] exp12  [N_LANES];

  cpl_lut #(.N_RD(N_LANES)) u_lut (
    .clk_i,
    .exp_we_i      (lut_we_i),
    .exp_waddr_i   (lut_waddr_i),
    .exp_wdata_i   (lut_wdata_i),
    .exp_raddr_i   (q11),             // u8(z)
    .exp_rdata_o   (exp12),
    .recip_raddr_i (s5_mant),
    .recip_rdata_o (recip6)
  );

  // ===========================================================================
  // stages 6..10: quantiser (one per lane)
  // ===========================================================================
  logic norm;
  assign norm = (out_mode_i == OUT_NORM);

  for (genvar l = 0; l < N_LANES; l++) begin : g_q
    logic [15:0] scale6;
    logic [4:0]  shift6;
    logic signed [QOUT_W-1:0] q;

    // NORM: shift = RECIP_FRAC - LUT_W + msb = 7 + msb <= 30 [D]
    // (registered at the quantiser input, LAT = 5)
    assign scale6 = norm ? 16'(recip6[l]) : scale_i;
    assign shift6 = norm ? 5'(RECIP_FRAC - LUT_W) + msb_q[6][l] : shift_i;

    cpl_quant #(.IN_W(ACC_W), .SCALE_W(16), .RND_FIX(RND_FIX), .LAT(QLAT)) u_quant (
      .clk_i,
      .acc_i   (v_q[6][l]),
      .scale_i (scale6),
      .shift_i (shift6),
      .q_o     (q)
    );

    assign q11[l] = q;
  end

  logic [LUT_W-1:0] s12_q [N_LANES];

  always_ff @(posedge clk_i) s12_q <= q11;

  // ===========================================================================
  // stage 12: causal mask, output select
  // ===========================================================================
  logic [N_LANES-1:0][31:0] data12;
  logic [LUT_W-1:0]         ev12 [N_LANES];   // exp value counted in the sums

  always_comb begin
    for (int l = 0; l < N_LANES; l++) begin
      logic [LUT_W-1:0] ev;
      ev = mask_q[12][l] ? '0 : exp12[l];
      ev12[l] = (out_mode_i == OUT_EXP && en_q[12][l]) ? ev : '0;
      unique case (out_mode_i)
        OUT_Q8, OUT_NORM: data12[l] = {24'b0, s12_q[l]};
        OUT_EXP:          data12[l] = {24'b0, ev};
        OUT_RAW32:        data12[l] = v_q[12][l];
        default:          data12[l] = '0;
      endcase
    end
  end

  logic [N_LANES-1:0][31:0] s13_data;
  logic [LUT_W-1:0]         s13_ev [N_LANES];

  always_ff @(posedge clk_i) begin
    s13_data <= data12;
    s13_ev   <= ev12;
  end

  assign out_data_o = s13_data;

  // ===========================================================================
  // stage 13: exp-sum update
  // ===========================================================================
  logic [LUT_W+LANE_AW-1:0] ev_all13;   // sum of the 4 lanes (sum_by_t)

  always_comb begin
    ev_all13 = '0;
    for (int l = 0; l < N_LANES; l++) ev_all13 = ev_all13 + (LUT_W+LANE_AW)'(s13_ev[l]);
  end

  // synchronous reset / clear (they are the same operation)
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      for (int j = 0; j < N_SUMS; j++) sums_q[j] <= '0;
    end else if (sum_clr_i) begin
      for (int j = 0; j < N_SUMS; j++) sums_q[j] <= '0;
    end else if (vld_q[POST_LAT] && out_mode_i == OUT_EXP) begin
      for (int j = 0; j < N_SUMS; j++) begin
        if (sum_by_t_i) begin
          if (t_q[POST_LAT][4:0] == 5'(j))
            sums_q[j] <= sums_q[j] + SUM_W'(ev_all13);
        end else begin
          // column j = 8*lane + phase
          if (phase_q[POST_LAT] == CNT_W'(j % MUX_PHASES))
            sums_q[j] <= sums_q[j] + SUM_W'(s13_ev[j / MUX_PHASES]);
        end
      end
    end
  end

endmodule
