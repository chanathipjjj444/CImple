// ---------------------------------------------------------------------------
// lut_rom.sv -- simple single-port ROM, contents from a $readmemh file.
//
// WHAT THE PAPER SHOWS:
//   "LUT e^x : 256 x 8b" (Fig. 4) and a reciprocal table M (Fig. 7). The
//   e^x table costs 0.34% of total power, so a tiny synthesised table is fine.
//   Both softmax tables are instances of this one module.
//
// WHERE THE CONTENTS COME FROM:
//   model/gen_luts.py writes build/exp_lut.hex and build/recip_lut.hex
//   (one hex value per line). `make luts` or `make test` runs it first.
//
// BEHAVIOUR TO IMPLEMENT:
//   1. Storage: DEPTH words of WIDTH bits.
//   2. Load it once at time 0 from INIT_HEX ($readmemh), only if INIT_HEX
//      is not the empty string.
//   3. Read: on a rising clock edge with en_i = 1, data_o <= mem[addr_i].
//      With en_i = 0, data_o HOLDS its last value.
//      -> 1 cycle of latency: the address goes in on edge N, the data is
//         visible right after edge N.
//
// CHECK: why registered and not combinational? On an FPGA this is what lets
//        the tool map it to a BRAM. And split_softmax counts on data_o holding
//        while en_i is low -- the reciprocal is read once and used later.
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
  logic [WIDTH-1:0] storage [DEPTH];
  initial begin
    if (INIT_HEX != "") $readmemh(INIT_HEX, storage);
  end
  always_ff @(posedge clk_i ) begin : load_bit
    if (en_i) begin
      data_o <= storage[addr_i];
    end
  end

endmodule
