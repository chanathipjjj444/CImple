// ---------------------------------------------------------------------------
// bitcell.sv -- behavioural stand-in for the paper's 8T SRAM standard cell
//
// WHAT THE REAL CELL GIVES THE ARCHITECTURE (and so what this must give too):
//   1. Storage with a write port (WL + BL).
//   2. A read tap that is ALWAYS ON and NON-DESTRUCTIVE: Q (and QB) are wired
//      straight into logic. No read access, no bitline, no sense amp, no
//      read mux. This is what makes "compute in memory" possible at all.
//   3. Write and read are independent, so a bank can be written while the
//      other bank is being computed on.
//
// WHAT THIS DOES NOT GIVE:
//   Area, energy, or read-disturb / write-contention behaviour. On an FPGA the
//   cell is a flip-flop with a clock enable -- writes are edge-sampled, not a
//   level-sensitive wordline pulse.
//
// This is the narrowest swap point. Replacing this module with a wrapper
// around the real standard cell leaves cim_macro (and everything above it)
// untouched.
// ---------------------------------------------------------------------------
module bitcell (
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
