// ---------------------------------------------------------------------------
// cpl_gbuf.sv -- global buffer: GBUF_BYTES of 128b lines, true dual port
//
// Paper: Ahn et al., arXiv:2604.15944v2, Fig. 1 / Sec. IV ("global buffer",
// 16 kb SRAM in silicon [P]); contract fpga/docs/ARCH.md §1, §2 (GBUF row),
// §3 (host window), §6. On the VP1202 the buffer is much larger (default
// 512 KiB) so that a whole attention head (X, W, Q, K, V^T, e, O) fits [A].
//
// Organisation: DEPTH = GBUF_BYTES/16 lines of 128b; byte b of line L is GBUF
// byte 16*L + b (little-endian inside the line) [A].
//   port A: read/write  -- engine activation / weight / LUT reads while busy,
//                          host (AXI) word reads and writes while idle
//   port B: write only  -- engine output writer
// Both ports have per-byte write enables (16 bits). The top level guarantees
// that A and B never write the same line in the same cycle and that the
// engine never reads a line on A in the cycle B writes it [A] (the engine
// orders its reads after its writes, cpl_engine.sv), so the collision
// behaviour of the RAM primitive never matters.
//
// Inference [A]: written after the UG901 "true dual port, byte-write enable"
// templates (one process per port on the shared array; plain `always`
// because SystemVerilog forbids two always_ff writing one variable).
// RAM_STYLE selects BRAM ("block", default; read-first template) or URAM
// ("ultra"; no-change template: URAM288 ports only support NO_CHANGE, so a
// port-A cycle with any byte enable set writes and does NOT read -- a_rdata_o
// is then undefined. Nobody uses it: host writes never use read data and the
// engine never writes port A, cpl_top.sv). 512 KiB = 128 RAMB36 or 16 URAM288
// (2 wide x 8 cascaded) on Versal.
//
// TIMING: read latency RD_LAT (1..3) cycles: address + a_en_i in cycle n
// -> a_rdata_o valid in cycle n + RD_LAT. RD_LAT = 2 adds an output register
// that Vivado absorbs into the BRAM DOB_REG / URAM OREG [A]; RD_LAT = 3 adds
// one more register after the depth mux / cascade (fabric, or the URAM
// cascade pipeline register) [A]. A write in cycle n (either port) is
// visible to a port-A read issued in cycle n + 1. BRAM: a port-A read during
// a port-A write of the same line returns the old data (unused).
// ---------------------------------------------------------------------------
module cpl_gbuf
  import cpl_pkg::*;
#(
  parameter int unsigned GBUF_BYTES = 512 * 1024,
  parameter int unsigned RD_LAT     = 2,
  parameter              RAM_STYLE  = "block",
  localparam int unsigned DEPTH     = GBUF_BYTES / LINE_BYTES,
  localparam int unsigned AW        = $clog2(DEPTH)
)(
  input  logic                    clk_i,
  // ---- port A: read / write -------------------------------------------------
  input  logic                    a_en_i,
  input  logic [LINE_BYTES-1:0]   a_we_i,      // byte enables (0 = read)
  input  logic [AW-1:0]           a_addr_i,
  input  logic [LINE_W-1:0]       a_wdata_i,
  output logic [LINE_W-1:0]       a_rdata_o,
  // ---- port B: write ----------------------------------------------------------
  input  logic [LINE_BYTES-1:0]   b_we_i,
  input  logic [AW-1:0]           b_addr_i,
  input  logic [LINE_W-1:0]       b_wdata_i
);

  if (RD_LAT < 1 || RD_LAT > 3) begin : g_bad_lat
    $error("cpl_gbuf: RD_LAT must be 1..3");
  end
  if (GBUF_BYTES % 1024 != 0 || GBUF_BYTES > (1 << 20)) begin : g_bad_size
    $error("cpl_gbuf: GBUF_BYTES must be a multiple of 1 KiB and <= 1 MiB (ARCH.md §3)");
  end

  logic [LINE_W-1:0] a_rd;

  if (RAM_STYLE == "ultra") begin : g_uram
    (* ram_style = "ultra" *) logic [LINE_W-1:0] mem [DEPTH];

    /* verilator lint_off MULTIDRIVEN */  // TDP template: one process per port
    always @(posedge clk_i) begin
      if (a_en_i) begin
        if (|a_we_i) begin                       // NO_CHANGE: write, no read
          for (int i = 0; i < LINE_BYTES; i++)
            if (a_we_i[i]) mem[a_addr_i][8*i +: 8] <= a_wdata_i[8*i +: 8];
        end else begin
          a_rd <= mem[a_addr_i];
        end
      end
    end
    always @(posedge clk_i) begin
      for (int i = 0; i < LINE_BYTES; i++)
        if (b_we_i[i]) mem[b_addr_i][8*i +: 8] <= b_wdata_i[8*i +: 8];
    end
    /* verilator lint_on MULTIDRIVEN */
  end else begin : g_bram
    (* ram_style = "block" *) logic [LINE_W-1:0] mem [DEPTH];

    // power-up content (BRAM INIT); the ISA never relies on it [A]
    initial begin
      for (int l = 0; l < int'(DEPTH); l++) mem[l] = '0;
    end

    /* verilator lint_off MULTIDRIVEN */  // TDP template: one process per port
    always @(posedge clk_i) begin
      if (a_en_i) begin
        for (int i = 0; i < LINE_BYTES; i++)
          if (a_we_i[i]) mem[a_addr_i][8*i +: 8] <= a_wdata_i[8*i +: 8];
        a_rd <= mem[a_addr_i];
      end
    end
    always @(posedge clk_i) begin
      for (int i = 0; i < LINE_BYTES; i++)
        if (b_we_i[i]) mem[b_addr_i][8*i +: 8] <= b_wdata_i[8*i +: 8];
    end
    /* verilator lint_on MULTIDRIVEN */
  end

  if (RD_LAT >= 2) begin : g_oreg
    logic [LINE_W-1:0] a_rd2;
    always_ff @(posedge clk_i) a_rd2 <= a_rd;   // BRAM DOB_REG / URAM OREG
    if (RD_LAT == 3) begin : g_creg
      logic [LINE_W-1:0] a_rd3;
      always_ff @(posedge clk_i) a_rd3 <= a_rd2; // after the depth mux / cascade
      assign a_rdata_o = a_rd3;
    end else begin : g_nocreg
      assign a_rdata_o = a_rd2;
    end
  end else begin : g_noreg
    assign a_rdata_o = a_rd;
  end

endmodule
