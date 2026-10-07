// ---------------------------------------------------------------------------
// split_softmax.sv -- LUT-based split softmax (paper section IV-B, Fig. 7)
//
// The two halves of softmax are decoupled so neither stalls the CIM:
//
//   NUMERATOR  (streaming, 1 score/cycle/lane)
//       zdot_i = z_i - z_quant_max          <- the max is CONSTANT (=127),
//                                              never recomputed per token
//       e_i    = EXP_LUT[z_i]               <- table bakes in the -127 offset
//                                              and the dequant scale
//       e_i goes straight back out as an INT8 activation for the A'V MAC
//       in the CIM, and is simultaneously accumulated into the running sum.
//
//   DENOMINATOR (once per row, after the last score)
//       S      = sum(e_i)
//       S      = (256 + m) * 2^(e-8)        <- normalise: e = MSB position
//       M      = RECIP_LUT[m] ~= 2^15/(256+m)
//       out    = sat8( (partial * M) >> (RECIP_FRAC - 8 + e) )
//
// Latency of the LUT is 1 cycle; `e_valid_o` is aligned with `e_o`.
// ---------------------------------------------------------------------------
module split_softmax
  import cimple_pkg::*;
#(
  // Defaults assume the simulator runs from tb/ and `model/gen_luts.py` has
  // written build/. Override with -GEXP_HEX='"..."' in other flows.
  parameter string EXP_HEX   = "../build/exp_lut.hex",
  parameter string RECIP_HEX = "../build/recip_lut.hex"
)(
  input  logic                      clk_i,
  input  logic                      rst_ni,

  // ---- numerator stream ---------------------------------------------------
  input  logic                      row_start_i,  // clear the exp-sum
  input  logic                      z_valid_i,
  input  logic signed [DATA_W-1:0]  z_i,          // INT8 quantised QK^T score
  output logic                      e_valid_o,
  output logic        [LUT_W-1:0]   e_o,          // UQ0.8 exp value

  // ---- denominator / final scaling ---------------------------------------
  input  logic                      row_done_i,   // last score of the row seen
  output logic        [SUM_W-1:0]   sum_o,
  output logic                      recip_valid_o,
  output logic        [LUT_W-1:0]   recip_o,      // M
  output logic        [4:0]         recip_shift_o,

  input  logic signed [ACC_W-1:0]   partial_i,    // sum_j e_j * V_j  (from CIM)
  input  logic                      partial_valid_i,
  output logic signed [QOUT_W-1:0]  score_o,      // final INT8 attention score
  output logic                      score_valid_o
);

  // ======================= numerator =======================================
  logic [LUT_W-1:0] exp_data;

  lut_rom #(.DEPTH(LUT_DEPTH), .WIDTH(LUT_W), .INIT_HEX(EXP_HEX)) u_exp (
    .clk_i,
    .en_i   (z_valid_i),
    .addr_i (z_i[DATA_W-1:0]),   // INT8 reinterpreted as the 8b table index
    .data_o (exp_data)
  );

  logic e_valid_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) e_valid_q <= 1'b0;
    else         e_valid_q <= z_valid_i;
  end
  assign e_valid_o = e_valid_q;
  assign e_o       = exp_data;

  // running exp-sum
  logic [SUM_W-1:0] sum_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if      (!rst_ni)     sum_q <= '0;
    else if (row_start_i) sum_q <= '0;
    else if (e_valid_q)   sum_q <= sum_q + SUM_W'(exp_data);
  end
  assign sum_o = sum_q;

  // ======================= denominator =====================================
  // leading-one position of the sum
  logic [4:0] msb_pos;
  always_comb begin
    msb_pos = '0;
    for (int unsigned b = 0; b < SUM_W; b++)
      if (sum_q[b]) msb_pos = 5'(b);
  end

  // 8 bits below the implicit leading 1
  logic [LUT_W-1:0] mant;
  always_comb begin
    if (msb_pos >= 5'(LUT_W)) mant = LUT_W'(sum_q >> (msb_pos - 5'(LUT_W)));
    else                      mant = LUT_W'(sum_q << (5'(LUT_W) - msb_pos));
  end

  logic row_done_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) row_done_q <= 1'b0;
    else         row_done_q <= row_done_i;
  end

  lut_rom #(.DEPTH(LUT_DEPTH), .WIDTH(LUT_W), .INIT_HEX(RECIP_HEX)) u_recip (
    .clk_i,
    .en_i   (row_done_i),
    .addr_i (mant),
    .data_o (recip_o)
  );

  logic [4:0] shift_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) shift_q <= '0;
    // 1/S ~= M * 2^-RECIP_FRAC * 2^(8-msb_pos); e_i are UQ0.8 so the
    // partial carries one 2^-8 factor that cancels the +8 here.
    else if (row_done_i) shift_q <= 5'(RECIP_FRAC) - 5'(LUT_W) + msb_pos;
  end

  assign recip_valid_o  = row_done_q;
  assign recip_shift_o  = shift_q;

  // ======================= final scale + requantise ========================
  quant_unit #(.IN_W(ACC_W), .SCALE_W(LUT_W)) u_q (
    .acc_i   (partial_i),
    .scale_i (recip_o),
    .shift_i (shift_q),
    .q_o     (score_o)
  );

  logic score_valid_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) score_valid_q <= 1'b0;
    else         score_valid_q <= partial_valid_i;
  end
  assign score_valid_o = score_valid_q;

endmodule
