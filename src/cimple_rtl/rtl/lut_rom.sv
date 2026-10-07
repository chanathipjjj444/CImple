// ---------------------------------------------------------------------------
// lut_rom.sv -- simple single-port ROM, contents from a $readmemh file.
//
// Both softmax tables use this. On FPGA it maps to a BRAM/LUTRAM; in an ASIC
// flow it synthesises to logic (the paper's e^x table is only 256 x 8b and
// costs 0.34% of total power, so a synthesised table is fine).
// ---------------------------------------------------------------------------
module lut_rom #(
  parameter int unsigned DEPTH    = 256,
  parameter int unsigned WIDTH    = 8,
  parameter string       INIT_HEX = ""
)(
  input  logic                        clk_i,
  input  logic                        en_i,
  input  logic [$clog2(DEPTH)-1:0]    addr_i,
  output logic [WIDTH-1:0]            data_o   // 1-cycle latency
);

  logic [WIDTH-1:0] mem [DEPTH];

  initial begin
    if (INIT_HEX != "") $readmemh(INIT_HEX, mem);
  end

  always_ff @(posedge clk_i) begin
    if (en_i) data_o <= mem[addr_i];
  end

endmodule
