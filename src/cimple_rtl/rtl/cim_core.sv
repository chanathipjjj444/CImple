// ---------------------------------------------------------------------------
// cim_core.sv -- 32kb CIM core: 32 partitions + output select
//
// One 64-wide INT8 activation vector is broadcast to all 32 partitions;
// each partition holds a different 64-weight column, so one 8-cycle pass
// produces 32 dot products of length 64 (= 2048 MACs / 8 cycles).
//
// The 32 results leave through 4 lanes under a 3b counter ("MUX 32x8",
// "COUNTER 3b", "23b x 4" in Fig. 4): 8 phases x 4 lanes = 32.
// ---------------------------------------------------------------------------
module cim_core
  import cimple_pkg::*;
(
  input  logic                     clk_i,
  input  logic                     rst_ni,
  // --- weight write port ---------------------------------------------------
  input  logic                     we_i,
  input  logic                     wbank_i,
  input  logic [PART_AW-1:0]       wpart_i,
  input  logic [ROW_AW-1:0]        waddr_i,
  input  logic [DATA_W-1:0]        wdata_i,
  // --- compute -------------------------------------------------------------
  input  logic                     bank_sel_i,
  input  logic [N_ROWS-1:0]        x_bits_i,
  input  logic                     mac_en_i,
  input  logic                     mac_first_i,
  // --- output select -------------------------------------------------------
  input  logic                     out_adv_i,     // advance the 3b counter
  input  logic                     out_clr_i,     // reset counter to phase 0
  output logic [CNT_W-1:0]         out_phase_o,
  output logic signed [MAC_W-1:0]  cim_out_o [N_LANES]
);

  logic signed [MAC_W-1:0] acc [N_PART];

  for (genvar p = 0; p < N_PART; p++) begin : g_part
    cim_partition u_part (
      .clk_i,
      .rst_ni,
      .we_i       (we_i && (wpart_i == PART_AW'(p))),
      .wbank_i,
      .waddr_i,
      .wdata_i,
      .bank_sel_i,
      .x_bits_i,
      .mac_en_i,
      .mac_first_i,
      .acc_o      (acc[p])
    );
  end

  logic [CNT_W-1:0] phase_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if      (!rst_ni)   phase_q <= '0;
    else if (out_clr_i) phase_q <= '0;
    else if (out_adv_i) phase_q <= phase_q + 1'b1;
  end
  assign out_phase_o = phase_q;

  // lane l, phase k  ->  partition l*MUX_PHASES + k
  for (genvar l = 0; l < N_LANES; l++) begin : g_lane
    assign cim_out_o[l] = acc[l*MUX_PHASES + int'(phase_q)];
  end

endmodule
