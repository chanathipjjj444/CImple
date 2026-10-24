// ---------------------------------------------------------------------------
// cpl_lut.sv -- split-softmax lookup tables (exp LUT RAM + reciprocal ROM)
//
// Paper: Ahn et al., arXiv:2604.15944v2, Sec. IV-B, Fig. 7 ("EXP LUT",
// "1/x LUT"); contract fpga/docs/ARCH.md §2, §5 (LUTLOAD, EXP, NORM).
// Replaces src/cimple_rtl/rtl/lut_rom.sv, whose $readmemh hex files are not
// wanted in the Vivado flow.
//
// EXP LUT: 256 x 8b RAM. Its contents gm.build_exp_lut(score_scale) depend on
// the score scale of the attention head, so it is loaded at run time by
// LUTLOAD instead of being a ROM [A]. Write port = one 128b GBUF line per
// cycle: entries 16*exp_waddr_i + i <= exp_wdata_i[8*i +: 8], i = 0..15 [A].
// One replica per read port (N_RD = N_LANES = 4 by default) so each lane gets
// a private single-read-port memory [A]; every replica is stored as 16 words
// x 128b (Vivado: distributed RAM, ~10 RAM32M16 each) followed by a 16:1
// byte select.
//
// RECIP ROM: 256 x 8b, RECIP[m] = round(2^15 / (256 + m)) = gm.build_recip_lut()
// (cpl_pkg::recip_entry) [P][D]: 1/S for S = (1.m) * 2^msb. Filled by an
// initial loop over the constant function (no hex file) -- Vivado and yosys
// both infer an initialised ROM from this. One replica per read port [A].
//
// TIMING: both tables have a synchronous read with LUT_LAT = 1: an address
// presented in cycle n gives the data in cycle n + 1 (output registers, no
// reset, no enable). A write to the exp LUT in cycle n is visible to a read
// address presented in cycle n + 1 or later; a read of the same entry in
// cycle n returns the old content [D]. The exp LUT powers up as zeros in
// simulation (initial block; on the FPGA the LUTRAM INIT is zero as well).
// ---------------------------------------------------------------------------
module cpl_lut
  import cpl_pkg::*;
#(
  parameter int unsigned N_RD = N_LANES
)(
  input  logic                 clk_i,

  // ---- exp LUT write: one 128b line = 16 entries per cycle -------------------
  input  logic                 exp_we_i,
  input  logic [3:0]           exp_waddr_i,      // entries 16*a .. 16*a+15
  input  logic [WR_W-1:0]      exp_wdata_i,      // byte i -> entry 16*a + i

  // ---- read ports (1-cycle latency) ------------------------------------------
  input  logic [LUT_W-1:0]     exp_raddr_i   [N_RD],
  output logic [LUT_W-1:0]     exp_rdata_o   [N_RD],
  input  logic [LUT_W-1:0]     recip_raddr_i [N_RD],
  output logic [LUT_W-1:0]     recip_rdata_o [N_RD]
);

  localparam int unsigned N_WORDS = LUT_DEPTH / WR_BYTES;   // 16 words x 128b

  for (genvar l = 0; l < N_RD; l++) begin : g_port

    // ---- exp LUT replica l ---------------------------------------------------
    logic [WR_W-1:0] exp_mem [N_WORDS];

    initial begin
      for (int w = 0; w < N_WORDS; w++) exp_mem[w] = '0;
    end

    always_ff @(posedge clk_i) begin
      if (exp_we_i) exp_mem[exp_waddr_i] <= exp_wdata_i;
    end

    logic [WR_W-1:0] exp_word;
    assign exp_word = exp_mem[exp_raddr_i[l][LUT_W-1:4]];

    always_ff @(posedge clk_i) begin
      exp_rdata_o[l] <= exp_word[LUT_W*exp_raddr_i[l][3:0] +: LUT_W];
    end

    // ---- reciprocal ROM replica l ----------------------------------------------
    logic [LUT_W-1:0] recip_rom [LUT_DEPTH];

    initial begin
      for (int unsigned m = 0; m < LUT_DEPTH; m++) recip_rom[m] = recip_entry(m);
    end

    always_ff @(posedge clk_i) begin
      recip_rdata_o[l] <= recip_rom[recip_raddr_i[l]];
    end

  end

endmodule
