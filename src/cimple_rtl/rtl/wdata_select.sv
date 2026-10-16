// ---------------------------------------------------------------------------
// wdata_select.sv -- "Write data select" + LUT output "Reg" (Fig. 4, bottom)
//
// WHAT THE PAPER SHOWS (Section IV-A, C, D):
//   The softmax / quantiser output is sent to ONE of two places:
//     - the CIM write bus WBL (128b)   : encoder-only, results become WEIGHTS
//                                        (e.g. Q or V is stored back in SRAM)
//     - the input driver XINLB (64b)   : decoder-only, results become the next
//                                        ACTIVATIONS
//   The 4 lanes x 8b the quantiser makes per cycle are gathered into wide words
//   before they are used (128b = 16 INT8, 64b = 8 INT8 ... see the question
//   below).
//
// OPEN QUESTION:  [A]
//   4 lanes x 8b = 32b per cycle, but the bus is 128b. The paper does not say
//   how the 32b is gathered. A shift register that collects words until the
//   bus is full is the simplest answer. Decide, then write it in the design log.
//
// BEHAVIOUR TO IMPLEMENT:
//   1. `Reg`: collect incoming 4 x 8b words into one wide word.
//   2. When the wide word is full, raise `full_o`.
//   3. dest_i picks where the full word goes; the other output stays idle.
//   4. valid flags must be aligned with the data they describe.
// ---------------------------------------------------------------------------
module wdata_select
  import cimple_pkg::*;
(
  input  logic                          clk_i,
  input  logic                          rst_ni,

  // --- from quantiser / softmax -------------------------------------------
  input  logic signed [QOUT_W-1:0]      score_i [N_LANES],
  input  logic                          score_valid_i,

  // --- control ------------------------------------------------------------
  input  logic                          dest_i,       // 0: weights (WBL), 1: activations (XINLB)
  input  logic                          flush_i,      // push a partly full word out

  // --- to the CIM -----------------------------------------------------------
  output logic [128-1:0]                wbl_o,        // 128b write bus
  output logic                          wbl_valid_o,
  output logic [N_ROWS-1:0]             xin_o,        // 64b activation word
  output logic                          xin_valid_o,
  output logic                          full_o
);

  // TODO: the collecting register ("Reg")
  // TODO: full / flush logic
  // TODO: destination select

endmodule
