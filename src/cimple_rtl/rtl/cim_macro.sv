// ---------------------------------------------------------------------------
// cim_macro.sv -- the paper's 16-bitcell / 8-OAI macro, built structurally
//
// PAPER:  16 8T SRAM bitcells arranged as two banks of 8, every two cells
//         (one per bank, same bit position) sharing one OAI gate. The OAI is
//         both the 1b multiplier and the bank selector.
//
// HERE:   16 `bitcell` instances + 8 `oai221` instances, wired so that
//         the OAI computes   y = ~( w_sel[i] & x )   with
//             w_sel[i] = (q0[i] | sel) & (q1[i] | ~sel)
//         i.e. sel=0 picks bank 0, sel=1 picks bank 1. That is exactly an
//         OAI221 with A=(q0,sel) B=(q1,~sel) C=x -- one gate does select + AND.
//         A trailing inverter recovers the product bit; on silicon the adder
//         tree can absorb the inversion, in an FPGA it folds into the LUT.
//
//         [A] Which rail (Q or QB) the paper taps, and so whether its OAI sees
//         inverted operands, is not stated. The function is identical; only
//         the polarity bookkeeping differs.
//
//         The ONLY things that differ from silicon are inside `bitcell`.
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

  // Per-bank wordlines: the write decode, as the row driver would do it.
  logic [N_BANKS-1:0] wl;
  for (genvar b = 0; b < N_BANKS; b++) begin : g_wl
    assign wl[b] = we_i && (wbank_i == 1'(b));
  end

  // 2 banks x 8 bits of storage. Only Q is consumed here; QB is exposed by the
  // cell and left unconnected (synthesis prunes it).
  logic [N_BANKS-1:0][DATA_W-1:0] q;

  for (genvar b = 0; b < N_BANKS; b++) begin : g_bank
    for (genvar i = 0; i < DATA_W; i++) begin : g_bit
      bitcell u_cell (
        .clk_i,
        .we_i (wl[b]),
        .d_i  (wdata_i[i]),
        .q_o  (q[b][i]),
        .qb_o ()
      );
    end
  end

  // 8 OAIs: one per weight bit, shared by that bit's cell in each bank.
  logic [DATA_W-1:0] y_n, prod;
  logic              sel_n;
  assign sel_n = ~bank_sel_i;

  for (genvar i = 0; i < DATA_W; i++) begin : g_oai
    oai221 u_oai (
      .a1_i (q[0][i]), .a2_i (bank_sel_i),
      .b1_i (q[1][i]), .b2_i (sel_n),
      .c_i  (x_bit_i),
      .y_o  (y_n[i])
    );
  end
  assign prod = ~y_n;

  // Treating the MSB nibble as SIGNED and the LSB nibble as UNSIGNED makes
  // {msb,lsb} exactly the two's-complement INT8 weight, so no extra sign
  // correction is needed downstream.
  assign p_msb_o = prod[DATA_W-1 -: NIB_W];
  assign p_lsb_o = prod[NIB_W-1 : 0];

endmodule
