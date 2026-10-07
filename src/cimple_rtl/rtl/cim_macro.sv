// ---------------------------------------------------------------------------
// cim_macro.sv -- functional equivalent of the paper's 16-bitcell / 8-OAI macro
//
// PAPER:  16 8T SRAM bitcells arranged as two banks of 8, every two cells
//         sharing one OAI gate. The OAI is both the 1b multiplier (w AND x,
//         using the inverted stored value and the inverted activation) and the
//         bank selector -- only one bank drives the OAI at a time.
//
// HERE:   the storage is plain flops and the OAI is a plain AND. The *bits that
//         come out* are identical; only the circuit that produces them differs.
//         That is the deliberate trade in this re-implementation -- see
//         docs/01-spec-teardown.md section "What is and is not reproduced".
//
//         Keep this module as the single point of divergence. Swapping in a
//         real standard-cell array later means replacing only this file.
// ---------------------------------------------------------------------------
module cim_macro
  import cimple_pkg::*;
(
  input  logic                     clk_i,
  // --- write port (simultaneous MAC + write is supported, as in the paper) --
  input  logic                     we_i,
  input  logic                     wbank_i,
  input  logic [DATA_W-1:0]        wdata_i,
  // --- read / compute port -------------------------------------------------
  input  logic                     bank_sel_i,   // Block0 / Block1
  input  logic                     x_bit_i,      // one activation bit this cycle
  output logic signed [NIB_W-1:0]  p_msb_o,      // signed  MSB-nibble product
  output logic        [NIB_W-1:0]  p_lsb_o       // unsigned LSB-nibble product
);

  logic [N_BANKS-1:0][DATA_W-1:0] cells_q;

  always_ff @(posedge clk_i) begin
    if (we_i) cells_q[wbank_i] <= wdata_i;
  end

  logic [DATA_W-1:0] w;
  assign w = cells_q[bank_sel_i];

  // The 8 OAI outputs: 4 bits of MSB-nibble product, 4 of LSB-nibble product.
  // Treating the MSB nibble as SIGNED and the LSB nibble as UNSIGNED makes
  // {msb,lsb} exactly the two's-complement INT8 weight, so no extra sign
  // correction is needed downstream.
  assign p_msb_o = x_bit_i ? $signed(w[DATA_W-1 -: NIB_W]) : '0;
  assign p_lsb_o = x_bit_i ?         w[NIB_W-1 : 0]        : '0;

endmodule
