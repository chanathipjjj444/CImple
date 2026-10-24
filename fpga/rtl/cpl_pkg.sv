// ---------------------------------------------------------------------------
// cpl_pkg.sv -- parameters, ISA fields and opcodes of the CIMple-FPGA design
//
// Contract: fpga/docs/ARCH.md. Geometry and widths are the ones of
// src/cimple_rtl/rtl/cimple_pkg.sv (paper Fig. 4); the cpl_ prefix keeps this
// tree independent of the original so both can be compiled side by side.
// ---------------------------------------------------------------------------
package cpl_pkg;

  // ---- CIM array geometry [P][F] ------------------------------------------
  localparam int unsigned N_ROWS     = 64;   // activation inputs (64b XIN)
  localparam int unsigned N_PART     = 32;   // partitions = output columns
  localparam int unsigned N_BANKS    = 2;    // Block0 / Block1, one active
  localparam int unsigned DATA_W     = 8;    // INT8 weights / activations
  localparam int unsigned NIB_W      = 4;    // MSB / LSB nibble split

  // ---- datapath widths (Fig. 4) --------------------------------------------
  localparam int unsigned TREE_W     = 10;   // adder tree over 64 nibbles
  localparam int unsigned COMB_W     = 15;   // (msb << 4) + lsb
  localparam int unsigned MAC_W      = 23;   // after the 8-cycle shift-acc
  localparam int unsigned N_LANES    = 4;    // CIM OUT 23b x 4
  localparam int unsigned MUX_PHASES = N_PART / N_LANES;   // 8
  localparam int unsigned CNT_W      = 3;
  localparam int unsigned ACC_W      = 32;   // intermediate ACC / buffer
  localparam int unsigned QOUT_W     = 8;

  // ---- write port: 128b = 16 weights per cycle [P] -------------------------
  localparam int unsigned WR_W       = 128;
  localparam int unsigned WR_BYTES   = WR_W / DATA_W;       // 16
  localparam int unsigned N_WGRP     = N_ROWS / WR_BYTES;   // 4 row groups
  localparam int unsigned WGRP_W     = $clog2(N_WGRP);      // 2

  // ---- softmax --------------------------------------------------------------
  localparam int unsigned LUT_DEPTH  = 256;
  localparam int unsigned LUT_W      = 8;
  localparam int unsigned SUM_W      = 24;
  localparam int unsigned N_SUMS     = 32;
  localparam int unsigned RECIP_FRAC = 15;

  // ---- memories ---------------------------------------------------------------
  localparam int unsigned LINE_W     = 128;  // GBUF line
  localparam int unsigned LINE_BYTES = LINE_W / 8;
  localparam int unsigned INSTR_W    = 256;
  localparam int unsigned IMEM_DEPTH = 256;
  localparam int unsigned PC_W       = $clog2(IMEM_DEPTH);

  localparam int unsigned PART_AW    = $clog2(N_PART);   // 5
  /* verilator lint_off UNUSEDPARAM */  // informative (row address width)
  localparam int unsigned ROW_AW     = $clog2(N_ROWS);   // 6
  /* verilator lint_on UNUSEDPARAM */

  // ---- AXI4-Lite map (ARCH.md §3) ---------------------------------------------
  localparam int unsigned AXI_ADDR_W = 21;
  localparam logic [31:0] CPL_ID     = 32'h4350_4C31;    // "CPL1"
  localparam logic [AXI_ADDR_W-1:0] IMEM_BASE = 21'h01_0000;
  localparam logic [AXI_ADDR_W-1:0] GBUF_BASE = 21'h10_0000;

  // ---- ISA (ARCH.md §4) ----------------------------------------------------------
  typedef enum logic [3:0] {
    OP_END     = 4'd0,
    OP_WLOAD   = 4'd1,
    OP_MATVEC  = 4'd2,
    OP_LUTLOAD = 4'd3,
    OP_NOP     = 4'd4
  } opcode_e;

  typedef enum logic [1:0] {
    ACC_NONE  = 2'd0,
    ACC_FIRST = 2'd1,
    ACC_ADD   = 2'd2
  } acc_mode_e;

  typedef enum logic [2:0] {
    OUT_NONE  = 3'd0,
    OUT_Q8    = 3'd1,
    OUT_EXP   = 3'd2,
    OUT_NORM  = 3'd3,
    OUT_RAW32 = 3'd4
  } out_mode_e;

  typedef enum logic [7:0] {
    ERR_NONE    = 8'd0,
    ERR_OPCODE  = 8'd1,   // bad opcode / reserved encoding / pc overflow
    ERR_ALIGN   = 8'd2,
    ERR_RANGE   = 8'd3,   // GBUF address out of range
    ERR_PART    = 8'd4    // bad partition range or ncols > 32
  } err_e;

  // Decoded instruction. Field positions are normative (ARCH.md §4).
  typedef struct packed {
    logic [3:0]  opcode;
    logic        bank;
    logic [1:0]  acc_mode;
    logic [2:0]  out_mode;
    logic        act_unsigned;
    logic        sum_by_t;
    logic        sum_clr;
    logic        causal;
    logic [1:0]  rsvd0;
    logic [15:0] ntok;
    logic [31:0] w1;        // aaddr / gaddr
    logic [31:0] w2;        // astride / pstride
    logic [31:0] w3;        // oaddr / {np, p0}
    logic [15:0] tstride;
    logic [15:0] cstride;
    logic [15:0] scale;
    logic [4:0]  shift;
    logic [5:0]  ncols;
    logic [15:0] acc_base;
    logic [15:0] qbase;
    logic [15:0] kbase;
  } instr_t;

  function automatic instr_t decode(input logic [INSTR_W-1:0] raw);
    instr_t d;
    logic [31:0] w [8];
    for (int i = 0; i < 8; i++) w[i] = raw[32*i +: 32];
    d.opcode       = w[0][3:0];
    d.bank         = w[0][4];
    d.acc_mode     = w[0][6:5];
    d.out_mode     = w[0][9:7];
    d.act_unsigned = w[0][10];
    d.sum_by_t     = w[0][11];
    d.sum_clr      = w[0][12];
    d.causal       = w[0][13];
    d.rsvd0        = w[0][15:14];
    d.ntok         = w[0][31:16];
    d.w1           = w[1];
    d.w2           = w[2];
    d.w3           = w[3];
    d.tstride      = w[4][15:0];
    d.cstride      = w[4][31:16];
    d.scale        = w[5][15:0];
    d.shift        = w[5][20:16];
    d.ncols        = w[5][29:24];
    d.acc_base     = w[6][15:0];
    d.qbase        = w[7][15:0];
    d.kbase        = w[7][31:16];
    return d;
  endfunction

  // round(2^15 / (256 + m)) -- the reciprocal LUT, = gm.build_recip_lut()
  function automatic logic [LUT_W-1:0] recip_entry(input int unsigned m);
    int unsigned den;
    den = 256 + m;
    // round-half-up of 32768/den; no exact ties occur for den in [256, 511]
    return LUT_W'((2 * 32768 + den) / (2 * den));
  endfunction

endpackage
