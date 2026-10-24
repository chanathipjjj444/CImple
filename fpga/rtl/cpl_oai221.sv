// ---------------------------------------------------------------------------
// cpl_oai221.sv -- Y = ~((A1|A2) & (B1|B2) & C), the standard OAI221 function
//
// Copy of src/cimple_rtl/rtl/oai221.sv (commit d195ccd), renamed for the
// VP1202 tree. Paper: Ahn et al., arXiv:2604.15944v2, Fig. 5 -- the OAI is
// both the 1b multiplier and the bank selector [P].
//
// Kept as its own module so cpl_macro reads as the paper's gate-level picture.
// On the Versal it is absorbed into one LUT together with the trailing
// inverter in cpl_macro.
// ---------------------------------------------------------------------------
module cpl_oai221 (
  input  logic a1_i, a2_i,
  input  logic b1_i, b2_i,
  input  logic c_i,
  output logic y_o
);
  assign y_o = ~((a1_i | a2_i) & (b1_i | b2_i) & c_i);
endmodule
