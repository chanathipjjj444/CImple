// ---------------------------------------------------------------------------
// cpl_top.sv -- CIMple-FPGA accelerator top level (AMD Versal Premium VP1202)
//
// Paper: Ahn et al., arXiv:2604.15944v2, Fig. 1 (CIM core, intermediate
// buffer, quantiser, LUT softmax, input and global buffer). Contract:
// fpga/docs/ARCH.md (§1 block diagram, §3 AXI4-Lite map, §5 ISA semantics).
//
//   s_axil --> cpl_axil --+-- regs / START / CLEAR --> cpl_engine (cpl_core, cpl_post)
//                         +-- IMEM host port --------> cpl_imem <-- engine fetch
//                         +-- GBUF host port --+
//                                             mux(busy) -> cpl_gbuf port A <-- engine reads
//                                                          cpl_gbuf port B <-- engine writes
//
// Ownership [A]: while the engine is busy it owns IMEM's read port and GBUF
// port A; cpl_axil answers host IMEM / GBUF accesses with SLVERR then
// (ARCH.md §3), so the muxes below never drop a live host request. Busy
// rises in the cycle after the START write executes and every AXI
// transaction is executed after the previous one, so a host access can
// never be in flight across the hand-over.
//
// Parameters:
//   GBUF_BYTES     global buffer size (multiple of 1 KiB, <= 1 MiB)    [A]
//   ACC_DEPTH      ACC buffer entries (power of two >= 8)               [A]
//   PIPE_TREE      0..2 pipeline stages in the CIM adder trees          [A]
//                  (cpl_partition; default 2)
//   RND_FIX        true round-half-away in the quantiser (ARCH.md §9)  [A]
//   GBUF_RD_LAT    GBUF RAM read latency 1..3 (2: + BRAM/URAM output   [A]
//                  reg, 3: + register after the depth mux / cascade)
//   GBUF_REQ_REG   1: register the port-A request (en / we / addr /     [A]
//                  wdata) after the host/engine mux, so the busy decode
//                  and the mux are not in the path to the ~128 BRAMs
//   GBUF_RAM_STYLE "block" (BRAM) or "ultra" (URAM) for the GBUF        [A]
// Port-A read latency seen by cpl_axil / cpl_engine = GBUF_RD_LAT +
// GBUF_REQ_REG (default 3 + 1 = 4). Port B is not delayed, so an engine read
// still sees every write committed before it was issued; the host side
// executes one AXI transaction at a time, so delaying its requests by one
// cycle keeps their order [A].
// irq_o = done | error (level; cleared by CTRL.CLEAR or the next START) [A].
//
// The X_INTERFACE_* attributes let IP integrator infer s_axil as an
// AXI4-Lite slave with clk_i / rst_ni; fpga/vivado/cpl_top_bd.v (a Verilog
// shim, needed because a module reference top cannot be SystemVerilog)
// carries the same attributes.
// ---------------------------------------------------------------------------
module cpl_top
  import cpl_pkg::*;
#(
  parameter int unsigned GBUF_BYTES     = 512 * 1024,
  parameter int unsigned ACC_DEPTH      = 1024,
  parameter int unsigned PIPE_TREE      = 2,
  parameter bit          RND_FIX        = 1'b0,
  parameter int unsigned GBUF_RD_LAT    = 3,
  parameter bit          GBUF_REQ_REG   = 1'b1,
  parameter              GBUF_RAM_STYLE = "block"
)(
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk_i CLK" *)
  (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axil, ASSOCIATED_RESET rst_ni" *)
  input  logic                  clk_i,
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst_ni RST" *)
  (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
  input  logic                  rst_ni,

  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil AWADDR" *)
  (* X_INTERFACE_PARAMETER = "PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 21, HAS_PROT 1, HAS_WSTRB 1, HAS_BRESP 1, HAS_RRESP 1, MAX_BURST_LENGTH 1, NUM_READ_OUTSTANDING 1, NUM_WRITE_OUTSTANDING 1" *)
  input  logic [AXI_ADDR_W-1:0] s_axil_awaddr,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil AWPROT" *)
  input  logic [2:0]            s_axil_awprot,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil AWVALID" *)
  input  logic                  s_axil_awvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil AWREADY" *)
  output logic                  s_axil_awready,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil WDATA" *)
  input  logic [31:0]           s_axil_wdata,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil WSTRB" *)
  input  logic [3:0]            s_axil_wstrb,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil WVALID" *)
  input  logic                  s_axil_wvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil WREADY" *)
  output logic                  s_axil_wready,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil BRESP" *)
  output logic [1:0]            s_axil_bresp,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil BVALID" *)
  output logic                  s_axil_bvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil BREADY" *)
  input  logic                  s_axil_bready,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil ARADDR" *)
  input  logic [AXI_ADDR_W-1:0] s_axil_araddr,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil ARPROT" *)
  input  logic [2:0]            s_axil_arprot,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil ARVALID" *)
  input  logic                  s_axil_arvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil ARREADY" *)
  output logic                  s_axil_arready,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil RDATA" *)
  output logic [31:0]           s_axil_rdata,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil RRESP" *)
  output logic [1:0]            s_axil_rresp,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil RVALID" *)
  output logic                  s_axil_rvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil RREADY" *)
  input  logic                  s_axil_rready,

  (* X_INTERFACE_INFO = "xilinx.com:signal:interrupt:1.0 irq_o INTERRUPT" *)
  (* X_INTERFACE_PARAMETER = "SENSITIVITY LEVEL_HIGH" *)
  output logic                  irq_o
);

  localparam int unsigned GAW    = $clog2(GBUF_BYTES / LINE_BYTES);
  localparam int unsigned GA_LAT = GBUF_RD_LAT + (GBUF_REQ_REG ? 1 : 0);  // port-A latency

  // ---- run control / status ------------------------------------------------------
  logic            start, clear, busy, done, error;
  logic [PC_W-1:0] pc_start, pc;
  logic [7:0]      err_code;
  logic [31:0]     cycles, instret, passes;

  // ---- IMEM ------------------------------------------------------------------------
  logic                 h_imem_we, h_imem_re, e_imem_re;
  logic [PC_W-1:0]      h_imem_waddr, h_imem_raddr, e_imem_raddr;
  logic [2:0]           h_imem_wword;
  logic [31:0]          h_imem_wdata;
  logic [3:0]           h_imem_wstrb;
  logic [INSTR_W-1:0]   imem_rdata;

  // ---- GBUF --------------------------------------------------------------------------
  logic                  h_gbuf_en, e_ga_en;
  logic [LINE_BYTES-1:0] h_gbuf_we, e_gb_we;
  logic [GAW-1:0]        h_gbuf_addr, e_ga_addr, e_gb_addr;
  logic [LINE_W-1:0]     h_gbuf_wdata, e_gb_wdata, ga_rdata;

  cpl_axil #(
    .GBUF_BYTES  (GBUF_BYTES),
    .GBUF_RD_LAT (GA_LAT)
  ) u_axil (
    .clk_i,
    .rst_ni,
    .s_axil_awaddr,
    .s_axil_awprot,
    .s_axil_awvalid,
    .s_axil_awready,
    .s_axil_wdata,
    .s_axil_wstrb,
    .s_axil_wvalid,
    .s_axil_wready,
    .s_axil_bresp,
    .s_axil_bvalid,
    .s_axil_bready,
    .s_axil_araddr,
    .s_axil_arprot,
    .s_axil_arvalid,
    .s_axil_arready,
    .s_axil_rdata,
    .s_axil_rresp,
    .s_axil_rvalid,
    .s_axil_rready,
    .start_o      (start),
    .clear_o      (clear),
    .pc_start_o   (pc_start),
    .busy_i       (busy),
    .done_i       (done),
    .error_i      (error),
    .err_code_i   (err_code),
    .pc_i         (pc),
    .cycles_i     (cycles),
    .instret_i    (instret),
    .passes_i     (passes),
    .imem_we_o    (h_imem_we),
    .imem_waddr_o (h_imem_waddr),
    .imem_wword_o (h_imem_wword),
    .imem_wdata_o (h_imem_wdata),
    .imem_wstrb_o (h_imem_wstrb),
    .imem_re_o    (h_imem_re),
    .imem_raddr_o (h_imem_raddr),
    .imem_rdata_i (imem_rdata),
    .gbuf_en_o    (h_gbuf_en),
    .gbuf_we_o    (h_gbuf_we),
    .gbuf_addr_o  (h_gbuf_addr),
    .gbuf_wdata_o (h_gbuf_wdata),
    .gbuf_rdata_i (ga_rdata)
  );

  cpl_imem u_imem (
    .clk_i,
    .we_i    (h_imem_we),
    .waddr_i (h_imem_waddr),
    .wword_i (h_imem_wword),
    .wdata_i (h_imem_wdata),
    .wstrb_i (h_imem_wstrb),
    .re_i    (busy ? e_imem_re    : h_imem_re),
    .raddr_i (busy ? e_imem_raddr : h_imem_raddr),
    .rdata_o (imem_rdata)
  );

  // ---- GBUF port A: host / engine mux (+ optional request register) --------------
  logic                  ga_en_m,  ga_en;
  logic [LINE_BYTES-1:0] ga_we_m,  ga_we;
  logic [GAW-1:0]        ga_addr_m, ga_addr;
  logic [LINE_W-1:0]     ga_wdata;

  assign ga_en_m   = busy ? e_ga_en   : h_gbuf_en;
  assign ga_we_m   = busy ? '0        : h_gbuf_we;  // the engine never writes port A
  assign ga_addr_m = busy ? e_ga_addr : h_gbuf_addr;

  if (GBUF_REQ_REG) begin : g_areq
    // synchronous reset on the enables only (rst_ni from proc_sys_reset)
    always_ff @(posedge clk_i) begin
      if (!rst_ni) begin
        ga_en <= 1'b0;
        ga_we <= '0;
      end else begin
        ga_en <= ga_en_m;
        ga_we <= ga_we_m;
      end
    end
    always_ff @(posedge clk_i) begin
      ga_addr  <= ga_addr_m;
      ga_wdata <= h_gbuf_wdata;
    end
  end else begin : g_noareq
    assign ga_en    = ga_en_m;
    assign ga_we    = ga_we_m;
    assign ga_addr  = ga_addr_m;
    assign ga_wdata = h_gbuf_wdata;
  end

  cpl_gbuf #(
    .GBUF_BYTES (GBUF_BYTES),
    .RD_LAT     (GBUF_RD_LAT),
    .RAM_STYLE  (GBUF_RAM_STYLE)
  ) u_gbuf (
    .clk_i,
    .a_en_i    (ga_en),
    .a_we_i    (ga_we),
    .a_addr_i  (ga_addr),
    .a_wdata_i (ga_wdata),
    .a_rdata_o (ga_rdata),
    .b_we_i    (e_gb_we),
    .b_addr_i  (e_gb_addr),
    .b_wdata_i (e_gb_wdata)
  );

  cpl_engine #(
    .GBUF_BYTES  (GBUF_BYTES),
    .ACC_DEPTH   (ACC_DEPTH),
    .PIPE_TREE   (PIPE_TREE),
    .RND_FIX     (RND_FIX),
    .GBUF_RD_LAT (GA_LAT)
  ) u_engine (
    .clk_i,
    .rst_ni,
    .start_i      (start),
    .clear_i      (clear),
    .pc_start_i   (pc_start),
    .busy_o       (busy),
    .done_o       (done),
    .error_o      (error),
    .err_code_o   (err_code),
    .pc_o         (pc),
    .cycles_o     (cycles),
    .instret_o    (instret),
    .passes_o     (passes),
    .imem_re_o    (e_imem_re),
    .imem_raddr_o (e_imem_raddr),
    .imem_rdata_i (imem_rdata),
    .ga_en_o      (e_ga_en),
    .ga_addr_o    (e_ga_addr),
    .ga_rdata_i   (ga_rdata),
    .gb_we_o      (e_gb_we),
    .gb_addr_o    (e_gb_addr),
    .gb_wdata_o   (e_gb_wdata)
  );

  assign irq_o = done | error;

endmodule
