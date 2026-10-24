// ---------------------------------------------------------------------------
// cpl_imem.sv -- instruction memory: 256 instructions x 256b
//
// Contract: fpga/docs/ARCH.md §1, §3 (IMEM[i].word[w] at 0x010000 + 32*i +
// 4*w), §4 (256b instruction = 8 little-endian 32-bit words). The paper has
// no ISA; the instruction store is part of our host interface [A].
//
// One simple-dual-port array (UG901 "simple dual port, byte-write" template):
//   write port: one 32-bit host word per cycle with its 4 AXI byte strobes,
//               word w of instruction i = bits [32*w +: 32] [A]
//   read port : one whole 256b instruction, shared by the engine (fetch,
//               while busy) and the host (word read-back, while idle); the
//               top level muxes the read address by busy [A].
// 64 kb -> 2 RAMB36 (or 4 x RAMB36 in 512 x 72 SDP mode) on Versal.
//
// TIMING: read latency 1 (address + re_i in cycle n -> rdata_o in n + 1, held
// until the next read). A write in cycle n is visible to a read in n + 1.
// ---------------------------------------------------------------------------
module cpl_imem
  import cpl_pkg::*;
(
  input  logic                 clk_i,
  // ---- host write port (32-bit word + byte strobes) -------------------------
  input  logic                 we_i,
  input  logic [PC_W-1:0]      waddr_i,     // instruction index
  input  logic [2:0]           wword_i,     // word 0..7 inside the instruction
  input  logic [31:0]          wdata_i,
  input  logic [3:0]           wstrb_i,
  // ---- read port (whole instruction) ----------------------------------------
  input  logic                 re_i,
  input  logic [PC_W-1:0]      raddr_i,
  output logic [INSTR_W-1:0]   rdata_o
);

  localparam int unsigned NBYTE = INSTR_W / 8;   // 32 byte lanes

  (* ram_style = "block" *) logic [INSTR_W-1:0] mem [IMEM_DEPTH];

  initial begin   // power-up content: all END (opcode 0) [A]
    for (int i = 0; i < int'(IMEM_DEPTH); i++) mem[i] = '0;
  end

  // byte enables of the 32 byte lanes: strobes of word wword_i
  logic [NBYTE-1:0] be;
  assign be = we_i ? (NBYTE'(wstrb_i) << {wword_i, 2'b00}) : '0;

  always_ff @(posedge clk_i) begin
    for (int b = 0; b < int'(NBYTE); b++)
      if (be[b]) mem[waddr_i][8*b +: 8] <= wdata_i[8*(b%4) +: 8];
    if (re_i) rdata_o <= mem[raddr_i];
  end

endmodule
