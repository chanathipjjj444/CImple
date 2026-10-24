// ---------------------------------------------------------------------------
// cpl_engine.sv -- instruction sequencer + MAC/drain/output datapath control
//
// Paper: Ahn et al., arXiv:2604.15944v2, Fig. 1 / Fig. 4 (CIM core, input
// buffer XIN, intermediate buffer, quantiser, softmax, global buffer),
// Sec. IV-A (8-cycle bit-serial MAC, 32 -> 4 output counter), Sec. IV-C..E
// (mappings, executed as programs of the ISA below). The paper has no ISA
// and no controller description; everything about sequencing is [A].
// Contract: fpga/docs/ARCH.md §4 (format) and §5 (normative semantics).
//
// Contains cpl_core (the CIM array) and cpl_post (ACC buffer, quantiser,
// split softmax). IMEM and GBUF are outside (cpl_top) and reached through
// the imem_* / ga_* (GBUF port A, reads) / gb_* (GBUF port B, writes) ports.
//
// INSTRUCTION FLOW (one instruction at a time) [A]:
//   FETCH  IMEM read at PC                       (IMEM latency 1)
//   DEC    ins_q <= cpl_pkg::decode(...)
//   CK1    error 1/2/4 flags; saturated multiplier operands
//   CK2    products  (n-1)*stride               (1 DSP58 each, registered)
//   CK3    last byte addresses amax / omax       (registered)
//   EXEC   error decision (priority 1 > 2 > 4 > 3, = sim.py), dispatch
//   ...    WLOAD / LUTLOAD / MATVEC execution, then RET (INSTRET++, PC++)
// Every error, including error 3 of MATVEC, is detected in EXEC before the
// instruction changes any state [A] (ARCH.md §5 allows error 3 later; doing
// it up front makes the RTL state after any error equal to the simulator's).
// Range arithmetic: every operand >= 2^20 is saturated to 2^20 before the
// multiply; since GBUF_BYTES <= 2^20 this never changes the error decision,
// and in an error-free instruction every saturated value is exact [D].
// Fetch past instruction 255 (a non-END instruction at 255 retires) ->
// error 1 with PC = 255, as in sim.py [A].
//
// WLOAD: one 128b GBUF line per cycle -> cpl_core 128b write port
//   (partition p0+i, row group g = line g of the partition's 64 bytes).
// LUTLOAD: 16 lines -> cpl_post exp-LUT write port (line k = entries 16k..).
// MATVEC, per activation vector t (ARCH.md §5 steps 1-4), four overlapped units:
//   fetch : 4 GBUF line reads -> XPF (prefetch buffer, 64 x 8b)
//   MAC   : XIN <= XPF; 8 cycles, XIN shifts left, x_bits = MSBs (MSB first)
//           [P]; one pass starts at most every PASS_GAP = 15 cycles, because
//           the core's accumulators are single-buffered: the 8-phase drain of
//           pass t must end before pass t+1 changes them (cpl_core TIMING) [D]
//   drain : MAC_LAT cycles after mac_first, 8 cycles of 4 lanes (phase k =
//           columns 8l+k) -> cpl_post with (phase, t) [F]
//   writer: cpl_post's results (POST_LAT later) fill a 32-column token buffer
//           (NBUF = 3 of them); a full token is written to GBUF port B in
//           increasing column order p (ARCH.md §5: later writes win):
//             cstride == 1 and a byte mode: the <= 32 contiguous bytes are
//               shifted into a 3-line window and written as <= 3 line writes
//               with byte enables ("fast path");
//             otherwise one column per cycle (RAW32: 4 bytes, 2 cycles when
//               they straddle a line), byte enables + replicated data.
// HAZARDS [A]:
//   * x_{t+1} must see the writes of step t. In EXEC the engine compares the
//     activation byte range [aaddr, amax] with the output byte range
//     [oaddr, omax] of the whole instruction. Disjoint -> no read of this
//     instruction can alias any of its writes, and x_{t+1} is prefetched
//     freely; overlapping -> the fetch of x_{t+1} waits until every write of
//     steps <= t has been committed to the RAM ("serial mode").
//   * A token buffer is reused only after its writes were issued
//     (issued - freed < NBUF at pass start), so no back-pressure into
//     cpl_post is ever needed.
//   * "freed" is counted one cycle after the last write of a token is
//     registered, i.e. when the RAM performs it; a port-A read issued from
//     then on sees the data, and never collides with a port-B write of the
//     same line. (The fetch issues its first read one cycle after f_start,
//     so this is one cycle of extra margin, not a requirement.)
//   * The next instruction starts only after all writes are in the RAM and
//     cpl_post is idle (its configuration inputs come from ins_q and its
//     ACC / exp-sum hazards rely on this, cpl_post.sv HAZARDS).
//
// TIMING (informative): instruction overhead 7 cycles (FETCH..EXEC + RET).
// MATVEC: >= 15 cycles per pass in the prefetch case with fast-path or no
// output (measured 15.10 over 400 passes; paper: 16 = 8 MAC + 8 drain [P]
// -- the last 2 drain cycles overlap the next pass's first MAC cycles,
// cpl_core TIMING); serial (aliasing) mode ~44 cycles; a generic-stride token costs
// ncols (x1..2 for RAW32) writer cycles, which then bounds the pass rate.
// WLOAD: 4*np + GBUF_RD_LAT + 2 cycles; LUTLOAD: 16 + GBUF_RD_LAT + 2.
// CYCLES counts every cycle from START to done / error.
//
// [A] Fanout: bank_sel / act_unsigned (dedicated copies of the ins_q
//     fields, bank_q / uns_q, with max_fanout), x_bits (XIN flip-flops),
//     mac_en / mac_first and out_phase are all driven straight from
//     registers, as cpl_core asks; Vivado replicates them.
// RESET [A]: synchronous, active low (rst_ni comes from proc_sys_reset,
//     synchronous to clk_i). Only control state is reset; ins_q, the
//     writer's data registers (win_q / wmask_q / gb_wdata_q) and the token
//     snapshot are plain registers in their own processes, so rst_ni is not
//     folded into their clock enables.
// GBUF_RD_LAT is the read latency seen on the ga_* port (cpl_top: RAM
//     latency + its optional request register), 1..4.
// ---------------------------------------------------------------------------
module cpl_engine
  import cpl_pkg::*;
#(
  parameter  int unsigned GBUF_BYTES  = 512 * 1024,
  parameter  int unsigned ACC_DEPTH   = 1024,
  parameter  int unsigned PIPE_TREE   = 2,
  parameter  bit          RND_FIX     = 1'b0,
  parameter  int unsigned GBUF_RD_LAT = 2,
  localparam int unsigned GAW         = $clog2(GBUF_BYTES / LINE_BYTES)
)(
  input  logic                    clk_i,
  input  logic                    rst_ni,

  // ---- run control / status ---------------------------------------------------
  input  logic                    start_i,      // pulse; ignored while busy
  input  logic                    clear_i,      // pulse; clears done / error / err_code
  input  logic [PC_W-1:0]         pc_start_i,
  output logic                    busy_o,
  output logic                    done_o,
  output logic                    error_o,
  output logic [7:0]              err_code_o,
  output logic [PC_W-1:0]         pc_o,
  output logic [31:0]             cycles_o,
  output logic [31:0]             instret_o,
  output logic [31:0]             passes_o,

  // ---- IMEM read port (latency 1) -------------------------------------------------
  output logic                    imem_re_o,
  output logic [PC_W-1:0]         imem_raddr_o,
  input  logic [INSTR_W-1:0]      imem_rdata_i,

  // ---- GBUF port A: reads (latency GBUF_RD_LAT) -------------------------------------
  output logic                    ga_en_o,
  output logic [GAW-1:0]          ga_addr_o,
  input  logic [LINE_W-1:0]       ga_rdata_i,

  // ---- GBUF port B: writes ---------------------------------------------------------
  output logic [LINE_BYTES-1:0]   gb_we_o,
  output logic [GAW-1:0]          gb_addr_o,
  output logic [LINE_W-1:0]       gb_wdata_o
);

  localparam int unsigned BAW      = GAW + 4;                   // GBUF byte address
  localparam int unsigned MAC_LAT  = DATA_W + PIPE_TREE;
  localparam int unsigned PASS_GAP = MAC_LAT + MUX_PHASES - 1 - PIPE_TREE;  // 15 [D]
  localparam int unsigned NBUF     = 3;                         // token buffers [A]
  localparam int unsigned SAT      = 1 << 20;                   // >= GBUF_BYTES
  localparam int unsigned XL       = N_ROWS / LINE_BYTES;       // 4 lines per x_t

  if (GBUF_RD_LAT < 1 || GBUF_RD_LAT > 4) begin : g_bad_lat
    $error("cpl_engine: GBUF_RD_LAT must be 1..4");
  end

  // saturate a 32-bit address / stride to 21 bits (value >= 2^20 -> 2^20)
  function automatic logic [20:0] sat20(input logic [31:0] x);
    return (x[31:20] != '0) ? 21'(SAT) : {1'b0, x[19:0]};
  endfunction

  // ===========================================================================
  // state
  // ===========================================================================
  typedef enum logic [3:0] {
    S_IDLE, S_FETCH, S_DEC, S_CK1, S_CK2, S_CK3, S_EXEC,
    S_WL, S_LUT, S_RDWAIT, S_MV, S_RET
  } state_e;

  state_e          st;
  /* verilator lint_off UNUSEDSIGNAL */  // w0[15:14] reserved: ignored, as sim.py [A]
  instr_t          ins_q;
  /* verilator lint_on UNUSEDSIGNAL */
  logic [PC_W-1:0] pc_q;
  // fanout copies of ins_q.bank / ins_q.act_unsigned for the CIM array
  // (bank_sel reaches ~18k OAI LUTs) [A]
  (* max_fanout = 512 *) logic bank_q;
  (* max_fanout = 512 *) logic uns_q;
  logic            done_q, error_q;
  logic [7:0]      err_q;
  logic [31:0]     cyc_q, iret_q, pass_q;
  logic            sclr_q;                      // exp-sum clear pulse -> cpl_post

  assign busy_o     = (st != S_IDLE);
  assign done_o     = done_q;
  assign error_o    = error_q;
  assign err_code_o = err_q;
  assign pc_o       = pc_q;
  assign cycles_o   = cyc_q;
  assign instret_o  = iret_q;
  assign passes_o   = pass_q;

  assign imem_re_o    = (st == S_FETCH);
  assign imem_raddr_o = pc_q;

  logic is_wl, is_lut, is_mv;
  assign is_wl  = (ins_q.opcode == OP_WLOAD);
  assign is_lut = (ins_q.opcode == OP_LUTLOAD);
  assign is_mv  = (ins_q.opcode == OP_MATVEC);

  // WLOAD fields (w3)
  logic [4:0] wl_p0;
  logic [5:0] wl_np;
  assign wl_p0 = ins_q.w3[4:0];
  assign wl_np = ins_q.w3[13:8];

  logic mv_writes;     // MATVEC writes output bytes
  assign mv_writes = (ins_q.out_mode != OUT_NONE) && (ins_q.ncols != '0);

  // ===========================================================================
  // checks: CK1 operands + flags, CK2 products, CK3 sums
  // ===========================================================================
  logic        e1_q, e2_q, e4_q;
  logic [15:0] op_n1, op_n2;                    // ntok-1 (or np-1), ntok-1
  logic [20:0] op_s1;                           // sat(astride / pstride)
  logic [15:0] op_ts, op_cs;                    // tstride, cstride
  logic [4:0]  op_nc;                           // ncols-1
  logic [20:0] sw1_q, sw3_q;                    // sat(w1), sat(w3)
  logic [36:0] m1_q;
  logic [31:0] m2_q;
  logic [20:0] m3_q;
  logic [39:0] amax_q, omax_q;                  // last byte touched (inclusive)

  always_ff @(posedge clk_i) begin
    if (st == S_CK1) begin
      e1_q  <= (ins_q.opcode > OP_NOP) ||
               (is_mv && (ins_q.acc_mode == 2'd3 || ins_q.out_mode > OUT_RAW32));
      e2_q  <= (is_wl  && (ins_q.w1[3:0] != '0 || ins_q.w2[3:0] != '0)) ||
               (is_lut &&  ins_q.w1[3:0] != '0) ||
               (is_mv  && (ins_q.w1[3:0] != '0 || ins_q.w2[3:0] != '0));
      e4_q  <= (is_wl && (wl_np == '0 || (7'(wl_p0) + 7'(wl_np)) > 7'(N_PART))) ||
               (is_mv && ins_q.ncols > 6'(N_PART));
      op_n1 <= is_mv ? ins_q.ntok - 16'd1 : is_wl ? 16'(wl_np - 6'd1) : 16'd0;
      op_n2 <= ins_q.ntok - 16'd1;
      op_s1 <= sat20(ins_q.w2);
      op_ts <= ins_q.tstride;
      op_cs <= ins_q.cstride;
      op_nc <= 5'(ins_q.ncols - 6'd1);
      sw1_q <= sat20(ins_q.w1);
      sw3_q <= sat20(ins_q.w3);
    end
    if (st == S_CK2) begin
      m1_q <= 37'(op_n1) * 37'(op_s1);
      m2_q <= 32'(op_n2) * 32'(op_ts);
      m3_q <= 21'(op_nc) * 21'(op_cs);
    end
    if (st == S_CK3) begin
      amax_q <= 40'(sw1_q) + 40'(m1_q) + (is_lut ? 40'(LUT_DEPTH - 1) : 40'(N_ROWS - 1));
      omax_q <= 40'(sw3_q) + 40'(m2_q) + 40'(m3_q) +
                ((ins_q.out_mode == OUT_RAW32) ? 40'd3 : 40'd0);
    end
  end

  logic rng_a, rng_o, e3, mv_alias;
  assign rng_a = (amax_q >= 40'(GBUF_BYTES));
  assign rng_o = (omax_q >= 40'(GBUF_BYTES));
  assign e3    = ((is_wl || is_lut) && rng_a) ||
                 (is_mv && ins_q.ntok != '0 && (rng_a || (mv_writes && rng_o)));
  // output byte range overlaps the activation byte range (exact when no error)
  assign mv_alias = mv_writes && (40'(sw3_q) <= amax_q) && (40'(sw1_q) <= omax_q);

  logic [7:0] exec_err;
  always_comb begin
    if      (e1_q) exec_err = ERR_OPCODE;
    else if (e2_q) exec_err = ERR_ALIGN;
    else if (e4_q) exec_err = ERR_PART;
    else if (e3)   exec_err = ERR_RANGE;
    else           exec_err = ERR_NONE;
  end

  // ===========================================================================
  // GBUF port A read issue + response tags
  // ===========================================================================
  typedef enum logic [1:0] { T_WL, T_LUT, T_X } tag_e;

  logic [GAW-1:0] ra_q;                         // read address presented this cycle
  logic           rd_issue;
  tag_e           rd_tag;
  logic [6:0]     rd_idx;

  logic           rt_vld [GBUF_RD_LAT];
  tag_e           rt_tag [GBUF_RD_LAT];
  logic [6:0]     rt_idx [GBUF_RD_LAT];

  assign ga_en_o   = rd_issue;
  assign ga_addr_o = ra_q;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      for (int i = 0; i < int'(GBUF_RD_LAT); i++) rt_vld[i] <= 1'b0;
    end else begin
      rt_vld[0] <= rd_issue;
      for (int i = 1; i < int'(GBUF_RD_LAT); i++) rt_vld[i] <= rt_vld[i-1];
    end
  end

  always_ff @(posedge clk_i) begin
    rt_tag[0] <= rd_tag;
    rt_idx[0] <= rd_idx;
    for (int i = 1; i < int'(GBUF_RD_LAT); i++) begin
      rt_tag[i] <= rt_tag[i-1];
      rt_idx[i] <= rt_idx[i-1];
    end
  end

  logic       rsp_vld;
  tag_e       rsp_tag;
  logic [6:0] rsp_idx;
  assign rsp_vld = rt_vld[GBUF_RD_LAT-1];
  assign rsp_tag = rt_tag[GBUF_RD_LAT-1];
  assign rsp_idx = rt_idx[GBUF_RD_LAT-1];

  logic rd_inflight;
  always_comb begin
    rd_inflight = 1'b0;
    for (int i = 0; i < int'(GBUF_RD_LAT); i++) rd_inflight |= rt_vld[i];
  end

  // ---- registered write ports of the core (WLOAD) and the exp LUT (LUTLOAD) -----------
  logic               cw_we_q;
  logic [PART_AW-1:0] cw_part_q;
  logic [WGRP_W-1:0]  cw_grp_q;
  logic [LINE_W-1:0]  cw_data_q;                // shared by the LUT write port
  logic               lw_we_q;
  logic [3:0]         lw_addr_q;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      cw_we_q <= 1'b0;
      lw_we_q <= 1'b0;
    end else begin
      cw_we_q <= rsp_vld && rsp_tag == T_WL;
      lw_we_q <= rsp_vld && rsp_tag == T_LUT;
    end
  end

  always_ff @(posedge clk_i) begin
    cw_part_q <= rsp_idx[6:2];
    cw_grp_q  <= rsp_idx[1:0];
    lw_addr_q <= rsp_idx[3:0];
    cw_data_q <= ga_rdata_i;
  end

  // ===========================================================================
  // MATVEC: fetch -> XPF -> XIN -> MAC -> drain
  // ===========================================================================
  logic [15:0]    f_t;                          // next token to fetch
  logic           f_iss;                        // issuing the 4 lines of f_t
  logic [1:0]     f_j;
  logic [GAW-1:0] f_line, f_str;                // line of x_{f_t}, astride in lines
  logic           xpf_busy;                     // XPF filling or full
  logic [2:0]     xpf_got;                      // lines received
  logic [LINE_W-1:0] xpf [XL];
  (* max_fanout = 64 *) logic [DATA_W-1:0] xin [N_ROWS];   // x_bits: 256 OAI loads each

  logic [15:0]    mv_issued, mv_freed;
  logic [3:0]     since_q;                      // cycles since the last mac_first (sat 15)
  logic           mac_act, mac_first_q;
  logic [2:0]     mac_k;
  logic [15:0]    pass_t;

  logic [MAC_LAT-2:0] fst_sr;                   // mac_first delay line
  logic           dr_act;
  logic [2:0]     dr_k;
  logic [15:0]    dr_t;

  logic           mv_alias_q, mv_fast_q, mv_wr_q, mv_w4_q;

  logic f_start, m_issue;
  assign f_start = (st == S_MV) && !f_iss && !xpf_busy && (f_t != ins_q.ntok) &&
                   (!mv_alias_q || mv_freed == f_t);
  assign m_issue = (st == S_MV) && (xpf_got == 3'(XL)) && (since_q >= 4'(PASS_GAP - 1)) &&
                   ((mv_issued - mv_freed) < 16'(NBUF));

  // ---- read issue mux (WLOAD / LUTLOAD / MATVEC fetch) -------------------------------
  logic [4:0] cnt_i;                            // WLOAD partition / LUTLOAD line counter
  logic [1:0] cnt_g;                            // WLOAD row group
  logic [GAW-1:0] wl_pbase, wl_pstr;

  always_comb begin
    rd_issue = 1'b0;
    rd_tag   = T_X;
    rd_idx   = '0;
    unique case (st)
      S_WL:  begin rd_issue = 1'b1; rd_tag = T_WL;  rd_idx = {5'(wl_p0 + cnt_i), cnt_g}; end
      S_LUT: begin rd_issue = 1'b1; rd_tag = T_LUT; rd_idx = {3'b0, cnt_i[3:0]}; end
      S_MV:  begin rd_issue = f_iss; rd_tag = T_X;  rd_idx = {5'b0, f_j}; end
      default: ;
    endcase
  end

  // ---- XPF fill / XIN shift --------------------------------------------------------------
  always_ff @(posedge clk_i) begin
    if (rsp_vld && rsp_tag == T_X) xpf[rsp_idx[1:0]] <= ga_rdata_i;
    if (m_issue) begin
      for (int r = 0; r < int'(N_ROWS); r++)
        xin[r] <= xpf[r / LINE_BYTES][8*(r % LINE_BYTES) +: 8];
    end else if (mac_act) begin
      for (int r = 0; r < int'(N_ROWS); r++) xin[r] <= {xin[r][DATA_W-2:0], 1'b0};
    end
  end

  logic [N_ROWS-1:0] x_bits;
  always_comb begin
    for (int r = 0; r < int'(N_ROWS); r++) x_bits[r] = xin[r][DATA_W-1];
  end

  // ===========================================================================
  // cpl_core + cpl_post
  // ===========================================================================
  logic signed [MAC_W-1:0] cim_out [N_LANES];

  cpl_core #(.PIPE_TREE(PIPE_TREE)) u_core (
    .clk_i,
    .rst_ni,
    .we_i           (cw_we_q),
    .wbank_i        (ins_q.bank),
    .wpart_i        (cw_part_q),
    .wgrp_i         (cw_grp_q),
    .wdata_i        (cw_data_q),
    .bank_sel_i     (bank_q),
    .x_bits_i       (x_bits),
    .mac_en_i       (mac_act),
    .mac_first_i    (mac_first_q),
    .act_unsigned_i (uns_q),
    .out_phase_i    (dr_k),
    .cim_out_o      (cim_out)
  );

  logic                     po_valid;
  logic [CNT_W-1:0]         po_phase;
  logic [N_LANES-1:0][31:0] po_data;
  logic                     post_busy;
  /* verilator lint_off UNUSEDSIGNAL */  // t / lane enables are implied by the token order and ncols
  logic [15:0]              po_t;
  logic [N_LANES-1:0]       po_en;
  /* verilator lint_on UNUSEDSIGNAL */

  cpl_post #(.ACC_DEPTH(ACC_DEPTH), .RND_FIX(RND_FIX)) u_post (
    .clk_i,
    .rst_ni,
    .acc_mode_i    (ins_q.acc_mode),
    .out_mode_i    (ins_q.out_mode),
    .scale_i       (ins_q.scale),
    .shift_i       (ins_q.shift),
    .ncols_i       (ins_q.ncols),
    .sum_by_t_i    (ins_q.sum_by_t),
    .causal_i      (ins_q.causal),
    .qbase_i       (ins_q.qbase),
    .kbase_i       (ins_q.kbase),
    .acc_base_i    (ins_q.acc_base),
    .sum_clr_i     (sclr_q),
    .lut_we_i      (lw_we_q),
    .lut_waddr_i   (lw_addr_q),
    .lut_wdata_i   (cw_data_q),
    .in_valid_i    (dr_act),
    .in_cim_i      (cim_out),
    .in_phase_i    (dr_k),
    .in_t_i        (dr_t),
    .out_valid_o   (po_valid),
    .out_t_o       (po_t),
    .out_phase_o   (po_phase),
    .out_lane_en_o (po_en),
    .out_data_o    (po_data),
    .out_busy_o    (post_busy)
  );

  // ===========================================================================
  // output writer: token buffers -> GBUF port B
  // ===========================================================================
  typedef enum logic [1:0] { W_IDLE, W_FWIN, W_FLINE, W_GEN } wstate_e;

  logic [31:0]      cbuf [NBUF][N_PART];        // v[p] (bytes in [7:0]) per token
  logic [NBUF-1:0]  cfull;
  logic [1:0]       fptr;
  (* max_fanout = 64 *) logic [1:0] rptr;
  wstate_e          wst;
  logic [BAW-1:0]   w_base;                     // oaddr + t*tstride of the token being written
  logic [BAW-1:0]   w_o;                        // generic path: oaddr + t*tstride + p*cstride
  (* max_fanout = 64 *) logic [4:0] w_p;
  logic             w_half;
  logic [1:0]       w_j;
  logic [3*LINE_W-1:0]     win_q;               // fast path: 3-line window
  logic [3*LINE_BYTES-1:0] wmask_q;
  logic [GAW-1:0]   wline_q;
  logic             free_pend;

  logic [LINE_BYTES-1:0] gb_we_q;
  logic [GAW-1:0]        gb_addr_q;
  logic [LINE_W-1:0]     gb_wdata_q;

  assign gb_we_o    = gb_we_q;
  assign gb_addr_o  = gb_addr_q;
  assign gb_wdata_o = gb_wdata_q;

  function automatic logic [1:0] ptr_inc(input logic [1:0] p);
    return (p == 2'(NBUF - 1)) ? 2'd0 : p + 2'd1;
  endfunction

  // ---- fill (cpl_post output: lane l, phase k -> column 8l+k) -------------------------
  always_ff @(posedge clk_i) begin
    if (po_valid) begin
      for (int l = 0; l < int'(N_LANES); l++)
        cbuf[fptr][l*MUX_PHASES + int'(po_phase)] <= po_data[l];
    end
  end

  // ---- token snapshot [A] -----------------------------------------------------------
  // Copied from cbuf[rptr] in every W_IDLE cycle, i.e. also on the edge that
  // leaves W_IDLE (cfull[rptr] = 1: the token is complete, and it is not
  // refilled before w_free), so the writer states read a 32-entry register
  // instead of the 96:1 cbuf mux.
  logic [31:0] wtok [N_PART];
  always_ff @(posedge clk_i) begin
    if (wst == W_IDLE) wtok <= cbuf[rptr];
  end

  // ---- fast path window (combinational, registered in W_FWIN) ----------------------------
  logic [2*LINE_W-1:0]   fb_bytes;
  logic [N_PART-1:0]     fb_mask;
  always_comb begin
    for (int p = 0; p < int'(N_PART); p++) begin
      fb_bytes[8*p +: 8] = wtok[p][7:0];
      fb_mask[p]         = (6'(p) < ins_q.ncols);
    end
  end

  // ---- generic path: column value, rotated word, byte mask ----------------------------
  logic [31:0] g_v, g_word, g_rot;
  logic [19:0] g_m20;
  always_comb begin
    g_v    = wtok[w_p];
    g_word = mv_w4_q ? g_v : {4{g_v[7:0]}};
    unique case (w_o[1:0])                      // rotate left by w_o[1:0] bytes
      2'd0: g_rot = g_word;
      2'd1: g_rot = {g_word[23:0], g_word[31:24]};
      2'd2: g_rot = {g_word[15:0], g_word[31:16]};
      default: g_rot = {g_word[7:0], g_word[31:8]};
    endcase
    g_m20  = (mv_w4_q ? 20'hF : 20'h1) << w_o[3:0];
  end

  logic w_free;                                 // token written: release its buffer
  logic fill_done;
  assign fill_done = po_valid && (po_phase == 3'(MUX_PHASES - 1));

  // ===========================================================================
  // main sequential block
  // ===========================================================================
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      st          <= S_IDLE;
      pc_q        <= '0;
      done_q      <= 1'b0;
      error_q     <= 1'b0;
      err_q       <= ERR_NONE;
      cyc_q       <= '0;
      iret_q      <= '0;
      pass_q      <= '0;
      sclr_q      <= 1'b0;
      ra_q        <= '0;
      cnt_i       <= '0;
      cnt_g       <= '0;
      wl_pbase    <= '0;
      wl_pstr     <= '0;
      f_t         <= '0;
      f_iss       <= 1'b0;
      f_j         <= '0;
      f_line      <= '0;
      f_str       <= '0;
      xpf_busy    <= 1'b0;
      xpf_got     <= '0;
      mv_issued   <= '0;
      mv_freed    <= '0;
      since_q     <= '1;
      mac_act     <= 1'b0;
      mac_first_q <= 1'b0;
      mac_k       <= '0;
      pass_t      <= '0;
      fst_sr      <= '0;
      dr_act      <= 1'b0;
      dr_k        <= '0;
      dr_t        <= '0;
      mv_alias_q  <= 1'b0;
      mv_fast_q   <= 1'b0;
      mv_wr_q     <= 1'b0;
      mv_w4_q     <= 1'b0;
      cfull       <= '0;
      fptr        <= '0;
      rptr        <= '0;
      wst         <= W_IDLE;
      w_base      <= '0;
      w_o         <= '0;
      w_p         <= '0;
      w_half      <= 1'b0;
      w_j         <= '0;
      wline_q     <= '0;
      free_pend   <= 1'b0;
      gb_we_q     <= '0;
      gb_addr_q   <= '0;
    end else begin
      sclr_q  <= 1'b0;
      gb_we_q <= '0;
      if (busy_o) cyc_q <= cyc_q + 32'd1;

      if (clear_i) begin
        done_q  <= 1'b0;
        error_q <= 1'b0;
        err_q   <= ERR_NONE;
      end

      // ======================= instruction sequencer ===========================
      unique case (st)
        S_IDLE: if (start_i) begin
          st      <= S_FETCH;
          pc_q    <= pc_start_i;
          done_q  <= 1'b0;
          error_q <= 1'b0;
          err_q   <= ERR_NONE;
          cyc_q   <= '0;
          iret_q  <= '0;
          pass_q  <= '0;
          sclr_q  <= 1'b1;                       // START clears the 32 exp-sums
        end
        S_FETCH: st <= S_DEC;
        S_DEC: st <= S_CK1;                      // ins_q: see below
        S_CK1: st <= S_CK2;
        S_CK2: st <= S_CK3;
        S_CK3: st <= S_EXEC;
        S_EXEC: begin
          if (exec_err != ERR_NONE) begin
            st      <= S_IDLE;
            error_q <= 1'b1;
            err_q   <= exec_err;
          end else begin
            unique case (ins_q.opcode)
              OP_END: begin
                st     <= S_IDLE;
                done_q <= 1'b1;
                iret_q <= iret_q + 32'd1;
              end
              OP_WLOAD: begin
                st       <= S_WL;
                ra_q     <= ins_q.w1[BAW-1:4];
                wl_pbase <= ins_q.w1[BAW-1:4];
                wl_pstr  <= ins_q.w2[BAW-1:4];
                cnt_i    <= '0;
                cnt_g    <= '0;
              end
              OP_LUTLOAD: begin
                st    <= S_LUT;
                ra_q  <= ins_q.w1[BAW-1:4];
                cnt_i <= '0;
              end
              OP_MATVEC: begin
                sclr_q     <= ins_q.sum_clr;
                st         <= (ins_q.ntok == '0) ? S_RET : S_MV;
                f_t        <= '0;
                f_iss      <= 1'b0;
                f_line     <= ins_q.w1[BAW-1:4];
                f_str      <= ins_q.w2[BAW-1:4];
                xpf_busy   <= 1'b0;
                xpf_got    <= '0;
                mv_issued  <= '0;
                mv_freed   <= '0;
                since_q    <= '1;
                mv_alias_q <= mv_alias;
                mv_wr_q    <= mv_writes;
                mv_fast_q  <= (ins_q.cstride == 16'd1) && (ins_q.out_mode != OUT_RAW32);
                mv_w4_q    <= (ins_q.out_mode == OUT_RAW32);
                fptr       <= '0;
                rptr       <= '0;
                wst        <= W_IDLE;
                w_base     <= BAW'(ins_q.w3);
              end
              default: st <= S_RET;              // NOP
            endcase
          end
        end
        S_WL: begin
          if (cnt_g == 2'(XL - 1)) begin
            cnt_g    <= '0;
            cnt_i    <= cnt_i + 5'd1;
            wl_pbase <= wl_pbase + wl_pstr;
            ra_q     <= wl_pbase + wl_pstr;
            if (cnt_i == 5'(wl_np - 6'd1)) st <= S_RDWAIT;
          end else begin
            cnt_g <= cnt_g + 2'd1;
            ra_q  <= ra_q + GAW'(1);
          end
        end
        S_LUT: begin
          cnt_i <= cnt_i + 5'd1;
          ra_q  <= ra_q + GAW'(1);
          if (cnt_i == 5'(LUT_DEPTH / LINE_BYTES - 1)) st <= S_RDWAIT;
        end
        S_RDWAIT: if (!rd_inflight && !cw_we_q && !lw_we_q) st <= S_RET;
        S_MV: if (mv_freed == ins_q.ntok && !free_pend && !post_busy) st <= S_RET;
        S_RET: begin
          iret_q <= iret_q + 32'd1;
          if (pc_q == PC_W'(IMEM_DEPTH - 1)) begin
            st      <= S_IDLE;                   // fetch past 255 [A]
            error_q <= 1'b1;
            err_q   <= ERR_OPCODE;
          end else begin
            st   <= S_FETCH;
            pc_q <= pc_q + PC_W'(1);
          end
        end
        default: st <= S_IDLE;
      endcase

      // ======================= MATVEC: fetch ===================================
      if (f_start) begin
        f_iss    <= 1'b1;
        f_j      <= '0;
        ra_q     <= f_line;
        xpf_busy <= 1'b1;
      end
      if (st == S_MV && f_iss) begin
        f_j  <= f_j + 2'd1;
        ra_q <= ra_q + GAW'(1);
        if (f_j == 2'(XL - 1)) begin
          f_iss  <= 1'b0;
          f_t    <= f_t + 16'd1;
          f_line <= f_line + f_str;
        end
      end
      if (rsp_vld && rsp_tag == T_X) xpf_got <= xpf_got + 3'd1;

      // ======================= MATVEC: MAC =====================================
      mac_first_q <= m_issue;
      if (m_issue) begin
        mac_act   <= 1'b1;
        mac_k     <= '0;
        pass_t    <= mv_issued;
        mv_issued <= mv_issued + 16'd1;
        pass_q    <= pass_q + 32'd1;
        since_q   <= '0;
        xpf_busy  <= 1'b0;
        xpf_got   <= '0;
      end else begin
        if (since_q != '1) since_q <= since_q + 4'd1;
        if (mac_act) begin
          mac_k <= mac_k + 3'd1;
          if (mac_k == 3'(DATA_W - 1)) mac_act <= 1'b0;
        end
      end

      // ======================= MATVEC: drain ===================================
      fst_sr <= {fst_sr[MAC_LAT-3:0], mac_first_q};
      if (fst_sr[MAC_LAT-2]) begin             // cycle c0 + MAC_LAT - 1
        dr_act <= 1'b1;
        dr_k   <= '0;
        dr_t   <= pass_t;
      end else if (dr_act) begin
        dr_k <= dr_k + 3'd1;
        if (dr_k == 3'(MUX_PHASES - 1)) dr_act <= 1'b0;
      end

      // ======================= MATVEC: writer ==================================
      free_pend <= w_free;
      if (free_pend) mv_freed <= mv_freed + 16'd1;

      begin
        logic [NBUF-1:0] cf;
        cf = cfull;
        if (fill_done) cf[fptr] = 1'b1;
        if (w_free)    cf[rptr] = 1'b0;
        cfull <= cf;
      end
      if (fill_done) fptr <= ptr_inc(fptr);
      if (w_free) begin
        rptr   <= ptr_inc(rptr);
        w_base <= w_base + BAW'(ins_q.tstride);
      end

      unique case (wst)
        W_IDLE: if (st == S_MV && cfull[rptr] && mv_wr_q) begin
          if (mv_fast_q) wst <= W_FWIN;
          else begin
            wst    <= W_GEN;
            w_p    <= '0;
            w_o    <= w_base;
            w_half <= 1'b0;
          end
        end
        W_FWIN: begin                            // win_q / wmask_q: see below
          wline_q <= w_base[BAW-1:4];
          w_j     <= '0;
          wst     <= W_FLINE;
        end
        W_FLINE: begin
          if (wmask_q[LINE_BYTES*w_j +: LINE_BYTES] != '0) begin
            gb_we_q    <= wmask_q[LINE_BYTES*w_j +: LINE_BYTES];
            gb_addr_q  <= wline_q + GAW'(w_j);
          end
          if (w_j == 2'd2 || (wmask_q >> (LINE_BYTES * (32'(w_j) + 1))) == '0) wst <= W_IDLE;
          else w_j <= w_j + 2'd1;
        end
        W_GEN: begin
          if (!w_half) begin
            gb_we_q   <= g_m20[LINE_BYTES-1:0];
            gb_addr_q <= w_o[BAW-1:4];
          end else begin
            gb_we_q   <= {12'b0, g_m20[19:16]};
            gb_addr_q <= w_o[BAW-1:4] + GAW'(1);
          end
          if (!w_half && g_m20[19:16] != '0) begin
            w_half <= 1'b1;                      // straddles a line: second cycle
          end else begin
            w_half <= 1'b0;
            if (6'(w_p) == ins_q.ncols - 6'd1) wst <= W_IDLE;
            else begin
              w_p <= w_p + 5'd1;
              w_o <= w_o + BAW'(ins_q.cstride);
            end
          end
        end
        default: wst <= W_IDLE;
      endcase
    end
  end

  // ---- registers without reset (own process: rst_ni stays out of their CE) ------------
  /* verilator lint_off UNUSEDSIGNAL */  // only .bank / .act_unsigned read directly
  instr_t dec_w;
  /* verilator lint_on UNUSEDSIGNAL */
  assign dec_w = decode(imem_rdata_i);

  always_ff @(posedge clk_i) begin
    if (st == S_DEC) begin
      ins_q  <= dec_w;
      bank_q <= dec_w.bank;
      uns_q  <= dec_w.act_unsigned;
    end
    if (wst == W_FWIN) begin
      win_q   <= {{LINE_W{1'b0}}, fb_bytes} << {w_base[3:0], 3'b000};
      wmask_q <= {{LINE_BYTES{1'b0}}, fb_mask} << w_base[3:0];
    end
    // data of the port-B write registered in this cycle (gb_we_q says
    // whether there is one; the value is don't-care otherwise)
    if (wst == W_FLINE) gb_wdata_q <= win_q[LINE_W*w_j +: LINE_W];
    if (wst == W_GEN)   gb_wdata_q <= {4{g_rot}};
  end

  // a token is released in the cycle its last write is registered (or at once
  // if the instruction writes nothing)
  always_comb begin
    w_free = 1'b0;
    unique case (wst)
      W_IDLE:  w_free = (st == S_MV) && cfull[rptr] && !mv_wr_q;
      W_FLINE: w_free = (w_j == 2'd2) || ((wmask_q >> (LINE_BYTES * (32'(w_j) + 1))) == '0);
      W_GEN:   w_free = !(!w_half && g_m20[19:16] != '0) && (6'(w_p) == ins_q.ncols - 6'd1);
      default: w_free = 1'b0;
    endcase
  end

endmodule
