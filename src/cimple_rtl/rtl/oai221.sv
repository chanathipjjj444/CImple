// ---------------------------------------------------------------------------
// oai221.sv -- Y = ~((A1|A2) & (B1|B2) & C), the standard OAI221 function
//
// Kept as its own module so cim_macro reads as the paper's gate-level
// picture. On an FPGA this is absorbed into a LUT; on an ASIC flow it maps
// to the library OAI221 cell.
// ---------------------------------------------------------------------------
module oai221 (
  input  logic a1_i, a2_i,
  input  logic b1_i, b2_i,
  input  logic c_i,
  output logic y_o
);
  assign y_o = ~((a1_i | a2_i) & (b1_i | b2_i) & c_i);
endmodule
