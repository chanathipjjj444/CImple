// ---------------------------------------------------------------------------
// cpl_quant.sv -- 32b -> 8b requantiser ("Quantization 32b to 8b", Fig. 4)
//
// Starts from src/cimple_rtl/rtl/quant_unit.sv (commit d195ccd). Paper: Ahn et
// al., arXiv:2604.15944v2, Fig. 4 / Fig. 7; contract fpga/docs/ARCH.md §2, §9.
//
//   q = sat8( (acc * scale + rc) >>> shift )
//
//   rc = 0                      if shift == 0
//      = +2^(shift-1)           if acc * scale >= 0
//      = -2^(shift-1)           if acc * scale <  0   (RND_FIX = 0)
//      = +2^(shift-1) - 1       if acc * scale <  0   (RND_FIX = 1)
//
// RND_FIX = 0 is bit-exact with cimple_model.quantise: "add -2^(s-1) then
// arithmetic shift" is floor(x - 1/2) for negative x, i.e. negative ties
// round away from zero but negative non-ties carry a -1 LSB bias [D]
// (ARCH.md §9, kept for bit-exactness). RND_FIX = 1 is true round half away
// from zero: floor(x + 1/2 - 2^-s) = -floor(|x| + 1/2) for x < 0 [D].
// Saturation is symmetric to [-128, 127] as in the original [D].
//
// The sign of the product is known before the multiply (acc < 0 and
// scale != 0), so rc is formed in parallel with it and the rounding add
// becomes the DSP58 post-adder (P = A*B + C) [A]. rc is a pure decoder of
// shift (no arithmetic): +2^(s-1) is a one-hot, 2^(s-1) - 1 a low mask and
// -2^(s-1) = ~(2^(s-1) - 1) a thermometer [D].
//
// PIPELINE (parameter LAT, 0..5) [A]: the datapath is cut into
//   I  input register         -> register if LAT >= 5   (DSP AREG / BREG)
//   A  multiply (+ rc)        -> register if LAT >= 1   (DSP MREG / CREG)
//   B  round add              -> register if LAT >= 2   (DSP PREG)
//   C  arithmetic shift       -> register if LAT >= 3
//   D  saturate               -> register if LAT >= 4   (q_o is registered)
// LAT = 0 is purely combinational (clk_i unused); cpl_post uses LAT = 5 for
// 200 MHz on Versal -2: 32x17 signed is two cascaded DSP58s whose operands
// come straight from registers, the 49b shifter and the 42-bit saturation
// check get a cycle each. The latency is exactly LAT cycles; acc_i, scale_i
// and shift_i are sampled together. The data registers have no reset so
// they can be absorbed into DSP/SRL primitives.
// ---------------------------------------------------------------------------
module cpl_quant
  import cpl_pkg::*;
#(
  parameter int unsigned IN_W    = ACC_W,
  parameter int unsigned SCALE_W = 16,
  parameter bit          RND_FIX = 1'b0,
  parameter int unsigned LAT     = 0
)(
  /* verilator lint_off UNUSEDSIGNAL */   // clk_i is unused when LAT == 0
  input  logic                      clk_i,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic signed [IN_W-1:0]    acc_i,
  input  logic        [SCALE_W-1:0] scale_i,   // unsigned multiplier
  input  logic [4:0]                shift_i,   // right shift after the multiply
  output logic signed [QOUT_W-1:0]  q_o
);

  // |acc * scale| < 2^(IN_W-1+SCALE_W); +1 sign bit, +1 headroom for rc [D]
  localparam int unsigned PROD_W = IN_W + SCALE_W + 1;
  localparam int signed   sd127  = 127;
  localparam int signed   sd128  = 128;
  localparam logic signed [PROD_W-1:0] SAT_HI =  PROD_W'(sd127);
  localparam logic signed [PROD_W-1:0] SAT_LO = -PROD_W'(sd128);

  // ---- I: input register ------------------------------------------------------
  logic signed [IN_W-1:0]    i_acc;
  logic        [SCALE_W-1:0] i_scale;
  logic        [4:0]         i_shift;

  if (LAT >= 5) begin : g_ri
    always_ff @(posedge clk_i) begin
      i_acc   <= acc_i;
      i_scale <= scale_i;
      i_shift <= shift_i;
    end
  end else begin : g_ni
    assign i_acc   = acc_i;
    assign i_scale = scale_i;
    assign i_shift = shift_i;
  end

  // ---- A: multiply, rounding constant ----------------------------------------
  logic signed [PROD_W-1:0] a_prod, a_rc;
  logic        [4:0]        a_shift;

  assign a_prod  = PROD_W'(i_acc) * $signed({1'b0, i_scale});
  assign a_shift = i_shift;

  always_comb begin
    logic neg, lo, at;
    neg = i_acc[IN_W-1] && (i_scale != '0);
    for (int b = 0; b < int'(PROD_W); b++) begin
      lo = (b + 1 <  int'(i_shift));             // bit below 2^(shift-1)
      at = (b + 1 == int'(i_shift));             // bit 2^(shift-1)
      if (i_shift == 5'd0) a_rc[b] = 1'b0;
      else if (!neg)       a_rc[b] = at;         //  2^(s-1)
      else if (RND_FIX)    a_rc[b] = lo;         //  2^(s-1) - 1
      else                 a_rc[b] = !lo;        // -2^(s-1)
    end
  end

  logic signed [PROD_W-1:0] b_prod, b_rc;
  logic        [4:0]        b_shift;

  if (LAT >= 1) begin : g_ra
    always_ff @(posedge clk_i) begin
      b_prod  <= a_prod;
      b_rc    <= a_rc;
      b_shift <= a_shift;
    end
  end else begin : g_na
    assign b_prod  = a_prod;
    assign b_rc    = a_rc;
    assign b_shift = a_shift;
  end

  // ---- B: round add ------------------------------------------------------------
  logic signed [PROD_W-1:0] b_rnd, c_rnd;
  logic        [4:0]        c_shift;

  assign b_rnd = b_prod + b_rc;

  if (LAT >= 2) begin : g_rb
    always_ff @(posedge clk_i) begin
      c_rnd   <= b_rnd;
      c_shift <= b_shift;
    end
  end else begin : g_nb
    assign c_rnd   = b_rnd;
    assign c_shift = b_shift;
  end

  // ---- C: arithmetic shift -----------------------------------------------------
  logic signed [PROD_W-1:0] c_sh, d_sh;

  assign c_sh = c_rnd >>> c_shift;

  if (LAT >= 3) begin : g_rc
    always_ff @(posedge clk_i) d_sh <= c_sh;
  end else begin : g_nc
    assign d_sh = c_sh;
  end

  // ---- D: saturate ---------------------------------------------------------------
  logic signed [QOUT_W-1:0] d_q;

  always_comb begin
    if      (d_sh > SAT_HI) d_q =  8'sd127;
    else if (d_sh < SAT_LO) d_q = -8'sd128;
    else                    d_q = QOUT_W'(d_sh);
  end

  if (LAT >= 4) begin : g_rd
    always_ff @(posedge clk_i) q_o <= d_q;
  end else begin : g_nd
    assign q_o = d_q;
  end

  // LAT > 5 is not supported; with LAT == 0 clk_i is intentionally unused.
  if (LAT > 5) begin : g_bad_lat
    $error("cpl_quant: LAT must be 0..5");
  end

endmodule
