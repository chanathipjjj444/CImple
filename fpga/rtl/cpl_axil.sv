// ---------------------------------------------------------------------------
// cpl_axil.sv -- AXI4-Lite slave: register file, IMEM window, GBUF window
//
// Contract: fpga/docs/ARCH.md §3 (normative register map and SLVERR rules).
// The paper's accelerator has no host bus; this is the VP1202 host
// interface, reached from the CIPS APU through a SmartConnect [A].
//
//   0x000000 ID       RO  "CPL1"
//   0x000004 CTRL     WO  bit0 START (pulse; the engine ignores it while busy)
//                         bit1 CLEAR (done / error / err_code)
//   0x000008 STATUS   RO  {8'b0, 8'b0, err_code, 5'b0, error, done, busy}
//   0x00000C PC_START RW  [7:0]
//   0x000010 PC, 0x14 CYCLES, 0x18 INSTRET, 0x1C PASSES  RO
//   0x000020 CFG      RO  {IMEM_DEPTH, GBUF_BYTES/1024}
//   0x010000 + 32*i + 4*w   IMEM[i].word[w]   RW (SLVERR while busy)
//   0x100000 + a            GBUF[a..a+3]      RW (SLVERR while busy)
// Unaligned or unmapped -> SLVERR (reads return 0, writes are dropped).
// Choices ARCH.md leaves open, identical to fpga/host/cpl/sim.py [A]:
//   * writes to RO registers are dropped with OKAY, CTRL reads 0 with OKAY;
//   * CTRL / PC_START only react to a write with wstrb[0] = 1;
//   * address bits are taken modulo 2^21 (the port is 21 bits wide).
//
// PROTOCOL [A]: AW, W and AR are captured in skid registers; *READY is
// "capture register empty" (a flip-flop, never a combinational function of
// *VALID), so AW and W may arrive in any order or cycle. One access unit
// executes one transaction at a time (a write when both AW and W are held
// and no B response is pending; otherwise a read when AR is held and no R
// response is pending). BVALID / RVALID (with BRESP / RRESP / RDATA) are
// registers held until BREADY / RREADY. AWPROT / ARPROT are ignored.
// The address is decoded when it is captured, so the execute cycle only
// sees flip-flops.
//
// TIMING: a write executes in the cycle after AW and W are both held and
// raises BVALID one cycle later; START / CLEAR are combinational pulses in
// the execute cycle, so the engine is busy from the next cycle and any
// later transaction sees busy_i = 1. Register reads: RVALID 2 cycles after
// the AR handshake; IMEM reads +1, GBUF reads +GBUF_RD_LAT (the latency seen
// on the gbuf_* port: cpl_gbuf RD_LAT + cpl_top's request register, 1..4).
// The memory ports driven here are only used while !busy_i; the top level
// gives them to the engine while it is busy.
// ---------------------------------------------------------------------------
module cpl_axil
  import cpl_pkg::*;
#(
  parameter  int unsigned GBUF_BYTES  = 512 * 1024,
  parameter  int unsigned GBUF_RD_LAT = 2,
  localparam int unsigned GAW         = $clog2(GBUF_BYTES / LINE_BYTES)
)(
  input  logic                    clk_i,
  input  logic                    rst_ni,

  // ---- AXI4-Lite slave --------------------------------------------------------
  input  logic [AXI_ADDR_W-1:0]   s_axil_awaddr,
  /* verilator lint_off UNUSEDSIGNAL */  // protection bits carry no meaning here
  input  logic [2:0]              s_axil_awprot,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic                    s_axil_awvalid,
  output logic                    s_axil_awready,
  input  logic [31:0]             s_axil_wdata,
  input  logic [3:0]              s_axil_wstrb,
  input  logic                    s_axil_wvalid,
  output logic                    s_axil_wready,
  output logic [1:0]              s_axil_bresp,
  output logic                    s_axil_bvalid,
  input  logic                    s_axil_bready,
  input  logic [AXI_ADDR_W-1:0]   s_axil_araddr,
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [2:0]              s_axil_arprot,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic                    s_axil_arvalid,
  output logic                    s_axil_arready,
  output logic [31:0]             s_axil_rdata,
  output logic [1:0]              s_axil_rresp,
  output logic                    s_axil_rvalid,
  input  logic                    s_axil_rready,

  // ---- run control / status (cpl_engine) ----------------------------------------
  output logic                    start_o,
  output logic                    clear_o,
  output logic [PC_W-1:0]         pc_start_o,
  input  logic                    busy_i,
  input  logic                    done_i,
  input  logic                    error_i,
  input  logic [7:0]              err_code_i,
  input  logic [PC_W-1:0]         pc_i,
  input  logic [31:0]             cycles_i,
  input  logic [31:0]             instret_i,
  input  logic [31:0]             passes_i,

  // ---- IMEM host port -------------------------------------------------------------
  output logic                    imem_we_o,
  output logic [PC_W-1:0]         imem_waddr_o,
  output logic [2:0]              imem_wword_o,
  output logic [31:0]             imem_wdata_o,
  output logic [3:0]              imem_wstrb_o,
  output logic                    imem_re_o,
  output logic [PC_W-1:0]         imem_raddr_o,
  input  logic [INSTR_W-1:0]      imem_rdata_i,

  // ---- GBUF host port (port A while idle) -------------------------------------------
  output logic                    gbuf_en_o,
  output logic [LINE_BYTES-1:0]   gbuf_we_o,
  output logic [GAW-1:0]          gbuf_addr_o,
  output logic [LINE_W-1:0]       gbuf_wdata_o,
  input  logic [LINE_W-1:0]       gbuf_rdata_i
);

  localparam logic [1:0] RESP_OKAY   = 2'b00;
  localparam logic [1:0] RESP_SLVERR = 2'b10;

  // register indices (address[5:2])
  localparam logic [3:0] R_ID = 4'd0, R_CTRL = 4'd1, R_STATUS = 4'd2, R_PC_START = 4'd3,
                         R_PC = 4'd4, R_CYCLES = 4'd5, R_INSTRET = 4'd6, R_PASSES = 4'd7,
                         R_CFG = 4'd8;

  typedef enum logic [1:0] { K_BAD, K_REG, K_IMEM, K_GBUF } kind_e;

  // ARCH.md §3 address decode (done at capture time)
  function automatic kind_e decode_addr(input logic [AXI_ADDR_W-1:0] a);
    if (a[1:0] != 2'b00)                                   return K_BAD;
    if (a[AXI_ADDR_W-1:6] == '0 && a[5:2] <= R_CFG)        return K_REG;
    if (a >= IMEM_BASE && 32'(a - IMEM_BASE) < 32 * IMEM_DEPTH) return K_IMEM;  // 8 KiB
    if (a >= GBUF_BASE && 32'(a - GBUF_BASE) < GBUF_BYTES)      return K_GBUF;
    return K_BAD;
  endfunction

  // ===========================================================================
  // capture registers
  // ===========================================================================
  logic                  aw_full, w_full, ar_full;
  /* verilator lint_off UNUSEDSIGNAL */  // [1:0] / high bits only matter to decode_addr at capture
  logic [AXI_ADDR_W-1:0] aw_addr, ar_addr;
  /* verilator lint_on UNUSEDSIGNAL */
  kind_e                 aw_kind, ar_kind;
  logic [31:0]           w_data;
  logic [3:0]            w_strb;

  assign s_axil_awready = !aw_full;
  assign s_axil_wready  = !w_full;
  assign s_axil_arready = !ar_full;

  if (GBUF_RD_LAT < 1 || GBUF_RD_LAT > 4) begin : g_bad_lat
    $error("cpl_axil: GBUF_RD_LAT must be 1..4");
  end

  // ===========================================================================
  // access unit
  // ===========================================================================
  typedef enum logic [0:0] { A_IDLE, A_RDWAIT } astate_e;
  astate_e   ast;
  logic [2:0] rd_cnt;        // remaining read latency
  kind_e      rd_kind;
  logic [2:0] rd_sel;        // IMEM: word in instruction; GBUF: word in line

  logic wr_go, rd_go;
  assign wr_go = (ast == A_IDLE) && aw_full && w_full && !s_axil_bvalid;
  assign rd_go = (ast == A_IDLE) && !wr_go && ar_full && !s_axil_rvalid;

  logic [3:0] aw_reg, ar_reg;
  assign aw_reg = aw_addr[5:2];
  assign ar_reg = ar_addr[5:2];

  // ---- write side effects (combinational in the execute cycle) ---------------------
  logic wr_ctrl;
  assign wr_ctrl  = wr_go && aw_kind == K_REG && aw_reg == R_CTRL && w_strb[0];
  assign start_o  = wr_ctrl && w_data[0];
  assign clear_o  = wr_ctrl && w_data[1];

  assign imem_we_o    = wr_go && aw_kind == K_IMEM && !busy_i;
  assign imem_waddr_o = aw_addr[12:5];               // IMEM_BASE is 8 KiB aligned
  assign imem_wword_o = aw_addr[4:2];
  assign imem_wdata_o = w_data;
  assign imem_wstrb_o = w_strb;

  logic gbuf_wr, gbuf_rd;
  assign gbuf_wr      = wr_go && aw_kind == K_GBUF && !busy_i;
  assign gbuf_rd      = rd_go && ar_kind == K_GBUF && !busy_i;
  assign gbuf_en_o    = gbuf_wr || gbuf_rd;
  assign gbuf_we_o    = gbuf_wr ? (LINE_BYTES'(w_strb) << {aw_addr[3:2], 2'b00}) : '0;
  assign gbuf_addr_o  = gbuf_wr ? aw_addr[GAW+3:4] : ar_addr[GAW+3:4];
  assign gbuf_wdata_o = {4{w_data}};

  assign imem_re_o    = rd_go && ar_kind == K_IMEM && !busy_i;
  assign imem_raddr_o = ar_addr[12:5];

  // ---- register read mux --------------------------------------------------------------
  logic [31:0] reg_rdata;
  always_comb begin
    unique case (ar_reg)
      R_ID:       reg_rdata = CPL_ID;
      R_STATUS:   reg_rdata = {16'b0, err_code_i, 5'b0, error_i, done_i, busy_i};
      R_PC_START: reg_rdata = 32'(pc_start_o);
      R_PC:       reg_rdata = 32'(pc_i);
      R_CYCLES:   reg_rdata = cycles_i;
      R_INSTRET:  reg_rdata = instret_i;
      R_PASSES:   reg_rdata = passes_i;
      R_CFG:      reg_rdata = {16'(IMEM_DEPTH), 16'(GBUF_BYTES / 1024)};
      default:    reg_rdata = '0;                  // CTRL (write-only)
    endcase
  end

  // ===========================================================================
  // sequential part
  // ===========================================================================
  // synchronous reset (rst_ni comes from proc_sys_reset, synchronous to clk_i)
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      aw_full       <= 1'b0;
      w_full        <= 1'b0;
      ar_full       <= 1'b0;
      aw_addr       <= '0;
      ar_addr       <= '0;
      aw_kind       <= K_BAD;
      ar_kind       <= K_BAD;
      w_data        <= '0;
      w_strb        <= '0;
      ast           <= A_IDLE;
      rd_cnt        <= '0;
      rd_kind       <= K_BAD;
      rd_sel        <= '0;
      s_axil_bvalid <= 1'b0;
      s_axil_bresp  <= RESP_OKAY;
      s_axil_rvalid <= 1'b0;
      s_axil_rresp  <= RESP_OKAY;
      s_axil_rdata  <= '0;
      pc_start_o    <= '0;
    end else begin
      // ---- capture ----------------------------------------------------------
      if (s_axil_awvalid && !aw_full) begin
        aw_full <= 1'b1;
        aw_addr <= s_axil_awaddr;
        aw_kind <= decode_addr(s_axil_awaddr);
      end
      if (s_axil_wvalid && !w_full) begin
        w_full <= 1'b1;
        w_data <= s_axil_wdata;
        w_strb <= s_axil_wstrb;
      end
      if (s_axil_arvalid && !ar_full) begin
        ar_full <= 1'b1;
        ar_addr <= s_axil_araddr;
        ar_kind <= decode_addr(s_axil_araddr);
      end

      // ---- responses accepted -------------------------------------------------
      if (s_axil_bvalid && s_axil_bready) s_axil_bvalid <= 1'b0;
      if (s_axil_rvalid && s_axil_rready) s_axil_rvalid <= 1'b0;

      // ---- write execute ------------------------------------------------------
      if (wr_go) begin
        aw_full       <= 1'b0;
        w_full        <= 1'b0;
        s_axil_bvalid <= 1'b1;
        unique case (aw_kind)
          K_REG: begin
            s_axil_bresp <= RESP_OKAY;               // RO registers: dropped, OKAY [A]
            if (aw_reg == R_PC_START && w_strb[0]) pc_start_o <= w_data[PC_W-1:0];
          end
          K_IMEM, K_GBUF: s_axil_bresp <= busy_i ? RESP_SLVERR : RESP_OKAY;
          default:        s_axil_bresp <= RESP_SLVERR;
        endcase
      end

      // ---- read execute -------------------------------------------------------
      if (rd_go) begin
        if (ar_kind == K_REG) begin
          ar_full       <= 1'b0;
          s_axil_rvalid <= 1'b1;
          s_axil_rresp  <= RESP_OKAY;
          s_axil_rdata  <= reg_rdata;
        end else if (ar_kind == K_BAD || busy_i) begin
          ar_full       <= 1'b0;
          s_axil_rvalid <= 1'b1;
          s_axil_rresp  <= RESP_SLVERR;
          s_axil_rdata  <= '0;
        end else begin
          ast     <= A_RDWAIT;
          rd_kind <= ar_kind;
          rd_cnt  <= (ar_kind == K_IMEM) ? 3'd1 : 3'(GBUF_RD_LAT);
          rd_sel  <= (ar_kind == K_IMEM) ? ar_addr[4:2] : {1'b0, ar_addr[3:2]};
        end
      end

      if (ast == A_RDWAIT) begin
        rd_cnt <= rd_cnt - 3'd1;
        if (rd_cnt == 3'd1) begin                    // memory data valid this cycle
          ast           <= A_IDLE;
          ar_full       <= 1'b0;
          s_axil_rvalid <= 1'b1;
          s_axil_rresp  <= RESP_OKAY;
          s_axil_rdata  <= (rd_kind == K_IMEM) ? imem_rdata_i[32*rd_sel +: 32]
                                               : gbuf_rdata_i[32*rd_sel[1:0] +: 32];
        end
      end
    end
  end

endmodule
