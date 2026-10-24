// ---------------------------------------------------------------------------
// cpl_bitcell.sv -- behavioural stand-in for the paper's 8T SRAM bitcell
//
// Copy of src/cimple_rtl/rtl/bitcell.sv (commit d195ccd), renamed for the
// VP1202 tree (fpga/docs/ARCH.md). Paper: Ahn et al., arXiv:2604.15944v2,
// Sec. IV-A / Fig. 5.
//
// WHAT THE REAL CELL GIVES THE ARCHITECTURE (and so what this must give too):
//   1. Storage with a write port (WL + BL).
//   2. A read tap that is ALWAYS ON and NON-DESTRUCTIVE: Q (and QB) are wired
//      straight into logic -- no read access, bitline, sense amp or read mux.
//      This is what makes "compute in memory" possible at all.
//   3. Write and read are independent, so a bank can be written while the
//      other bank is being computed on [P] ("simultaneous read and write").
//
// WHAT THIS DOES NOT GIVE:
//   Area, energy, read-disturb / write-contention behaviour. On the Versal
//   the cell is a fabric flip-flop with clock enable (FDRE), written on the
//   clock edge instead of by a level-sensitive wordline pulse [A]. Because
//   every bit is read every cycle the array cannot map to BRAM/LUTRAM; this is
//   near-memory compute, not CIM (ARCH.md §9).
//
// [A] No reset: SRAM content is undefined at power-up and the host always
//     loads weights (WLOAD) before use. Saves 32k reset nets.
// ---------------------------------------------------------------------------
module cpl_bitcell (
  input  logic clk_i,
  input  logic we_i,    // "wordline": write enable for this cell
  input  logic d_i,     // "bitline":  write data
  output logic q_o,     // stored value, always driven
  output logic qb_o     // complement, always driven
);

  logic q_q;

  always_ff @(posedge clk_i) begin
    if (we_i) q_q <= d_i;
  end

  assign q_o  = q_q;
  assign qb_o = ~q_q;

endmodule
