// ---------------------------------------------------------------------------
// quant_unit.sv -- 32b -> 8b requantiser ("Quantization 32b to 8b", 4 lanes)
//
// q = sat8( round( acc * scale / 2^shift ) )
//
// Round-half-away-from-zero, symmetric saturation to [-128, 127].
// Purely combinational; register it outside if timing needs it.
// ---------------------------------------------------------------------------
module quant_unit
  import cimple_pkg::*;
#(
  parameter int unsigned IN_W    = ACC_W,
  parameter int unsigned SCALE_W = 16
)(
  input  logic signed [IN_W-1:0]    acc_i,
  input  logic        [SCALE_W-1:0] scale_i,   // unsigned multiplier
  input  logic [4:0]                shift_i,   // right shift after the multiply
  output logic signed [QOUT_W-1:0]  q_o
);

  localparam int unsigned PROD_W = IN_W + SCALE_W + 1;
  localparam int signed  sd127 =  127;
  localparam int signed  sd128 =  128;

  //rnd == round
  logic signed [PROD_W-1:0] prod, rnd, shifted;
  
  assign prod = $signed(acc_i) * $signed({1'b0, scale_i});

  // round half away from zero: add +/- 2^(shift-1) before the arithmetic shift
  always_comb begin
    if (shift_i == 0) rnd = prod;
    else if (prod >= 0) rnd = prod + (PROD_W'(1) <<< (shift_i - 1));
    else                rnd = prod - (PROD_W'(1) <<< (shift_i - 1));
  end

  assign shifted = rnd >>> shift_i;

  localparam logic signed [PROD_W-1:0] SAT_HI =  PROD_W'(sd127);
  localparam logic signed [PROD_W-1:0] SAT_LO = -PROD_W'(sd128);

  always_comb begin
    if      (shifted > SAT_HI) q_o =  8'sd127;
    else if (shifted < SAT_LO) q_o = -8'sd128;
    else                       q_o = QOUT_W'(shifted);
  end

endmodule
