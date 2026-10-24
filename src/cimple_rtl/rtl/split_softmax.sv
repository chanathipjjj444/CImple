// ---------------------------------------------------------------------------
// split_softmax.sv -- LUT-based split softmax (paper section IV-B, Fig. 7)
//
// WHAT THE PAPER SHOWS:  [P]
//   Softmax is split so neither half stalls the CIM:
//     - the max is a CONSTANT, z_quant_max = 127, never searched per row:
//           zdot_i = z_i - z_quant_max
//     - NUMERATOR streams: e_i = exp(zdot_i) from a LUT, sent straight back
//       into the CIM as the A'V activation, while a running sum grows.
//     - DENOMINATOR once per row: after the last score, multiply the A'V
//       result by M ~= 1 / sum, also read from a LUT, and quantise to 8b.
//
// WHAT THE PAPER DOES NOT SAY:  [A]  (decided in model/cimple_model.py)
//   - EXP_LUT[i] = round(255 * exp((int8(i) - 127) * score_scale))
//     The table already contains the -127 offset and all the scaling, so the
//     hardware just uses the INT8 score as the address. e_i is UQ0.8 (0..255).
//   - The reciprocal is taken of a NORMALISED sum:
//         p      = position of the leading 1 of the sum
//         m      = the 8 bits just below that leading 1   (sum = 1.m * 2^p)
//         M      = RECIP_LUT[m] ~= 2^15 / (256 + m)
//         shift  = RECIP_FRAC - LUT_W + p
//         score  = sat8( round( partial * M / 2^shift ) )
//   - The golden model is softmax_row() / mantissa8() / msb_pos() in
//     cimple_model.py. Match it bit for bit.
//
// TIMING THE TESTBENCH EXPECTS (tb/test_split_softmax.py):
//   1. row_start_i pulse          -> sum cleared
//   2. z_valid_i + z_i on edge N  -> e_valid_o = 1 and e_o valid right after
//                                    edge N (the LUT's 1 cycle)
//   3. each valid e is added to the sum; sum_o is checked one clock after
//      the last e_valid_o
//   4. row_done_i pulse on edge N -> recip_o (M) and recip_shift_o valid by
//                                    edge N+1, and they must then HOLD
//   5. partial_valid_i + partial_i on edge N -> score_o correct right after
//                                    edge N (combinational is fine);
//                                    score_valid_o one clock later
// BUILDING BLOCKS YOU ALREADY HAVE:
//   - lut_rom   x2 (EXP_HEX for e^x, RECIP_HEX for M)
//   - quant_unit   for the final  sat8(round(partial * M >> shift))
//
// CHECK: the sum is unsigned and SUM_W = 24b. How many 8b exp values can you
//        add before it overflows? Is that enough for 1024 tokens?
// CHECK: what should mantissa / shift do when the sum is smaller than 2^8?
//        Look at mantissa8() in the model before you decide.
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
  // TODO: exp LUT (lut_rom), addressed by the raw 8 bits of z_i
  lut_rom  #(.INIT_HEX(EXP_HEX), .DEPTH(LUT_DEPTH), .WIDTH(LUT_W))numerator_rom(
    .clk_i (clk_i),
    .en_i (z_valid_i),
    .addr_i (z_i[DATA_W-1:0]),
    .data_o (e_o)
  );
  // TODO: e_valid: z_valid delayed by one cycle, so it lines up with e_o
  always_ff @( posedge clk_i or negedge rst_ni ) begin : enable_e_valid
    if (!rst_ni) e_valid_o <= 1'b0;
    else e_valid_o <= z_valid_i;
  end
  // TODO: running exp-sum: clear on row_start, add e_o when it is valid
  logic [SUM_W-1:0] sum_q;
  always_ff @( posedge clk_i or negedge rst_ni) begin : exp_sum
    if (!rst_ni) sum_q <= '0;
    else if(row_start_i) sum_q <= '0;
    else if(e_valid_o) sum_q <= sum_q + SUM_W'(e_o);
  end
  assign sum_o = sum_q;
  // ======================= denominator =====================================
  // THE IDEA: softmax output = partial / S, where S = sum of all e_i.
  // S is only known after the LAST score, so we cannot divide while streaming.
  // Dividing is expensive, so we multiply by a table value M ~ 1/S instead.
  //
  // But S can be any 24-bit number and a table can't have 16M entries. The
  // trick is "scientific notation in binary": write S = 1.m x 2^p
  //     p = position of the leading 1 of S
  //     m = the 8 bits right below that leading 1
  // Then 1/S = (1/1.m) x 2^-p. The table only has to answer "1/1.m" for the
  // 256 possible m, and the 2^-p part is just a right shift.
  //
  // Worked example, S = 1000 = 1111101000 (binary):
  //     leading 1 is at bit 9          -> p = 9
  //     the 8 bits below it: 11110100  -> m = 244
  //     M = RECIP_LUT[244] = 2^15/(256+244) = 66
  //     shift = RECIP_FRAC - LUT_W + p = 15 - 8 + 9 = 16
  //     partial * 66 / 2^16 = partial * 0.001007  ~  partial / 1000
  // Small S, e.g. S = 77 = 1001101: p = 6, pad on the right with zeros to
  //     get 8 bits: 001101 + 00 -> m = 00110100 = 52.

  // ---- step 1: p = position of the leading 1 of sum_q ---------------------
  logic [4:0] msb_pos;
  always_comb begin : find_largest_1
    msb_pos = 0;
    for (int i = 0; i < SUM_W ; i++ ) begin
      if(sum_q[i]) msb_pos = 5'(i);
    end
  end

  // ---- step 2: m = the 8 bits right below the leading 1 -------------------
  logic [LUT_W-1:0] mant;
  // TODO: always_comb, two cases, then keep only the low LUT_W bits (that
  //       also throws away the leading 1 itself):
  //         msb_pos >= LUT_W : sum_q >> (msb_pos - LUT_W)    sum is 256 or more
  //         otherwise        : sum_q << (LUT_W - msb_pos)    sum is below 256
  //       (the model does exactly this in mantissa8())
  always_comb begin : find_8bits
    if (LUT_W'(msb_pos) >= LUT_W) mant = sum_q >> msb_pos - $bits(msb_pos)'(LUT_W);
    else mant = $bits(mant)'(sum_q << LUT_W - LUT_W'(msb_pos));
  end

  // ---- step 3: M from the reciprocal table, read once per row -------------
  // TODO: a second lut_rom, like u_exp above, but INIT_HEX is RECIP_HEX.
  //       en_i   = the signal that says the row is finished
  //       addr_i = mant
  //       data_o = recip_o
  //       The ROM holds data_o until the next read, which is what makes
  //       recip_o stay put for the final scaling.

  // ---- step 4: shift, and the valid flag ----------------------------------
  logic             row_done_q;
  logic [4:0]       shift_q;
  // TODO: flop for row_done_q: row_done_i delayed one clock (async reset).
  // TODO: flop for shift_q: when row_done_i is high, load
  //         5'(RECIP_FRAC) - 5'(LUT_W) + msb_pos
  //       otherwise HOLD. (Here an enable IS right: this one must remember.)
  // TODO: assign recip_valid_o and recip_shift_o from those two.

  // ======================= final scale + requantise ========================
  // TODO: quant_unit: partial_i * M >> shift  -> score_o
  // TODO: score_valid: partial_valid delayed by one cycle

endmodule
