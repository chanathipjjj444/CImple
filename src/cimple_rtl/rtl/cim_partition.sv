// ---------------------------------------------------------------------------
// cim_partition.sv -- one of the 32 CIM partitions
//
// 64 macros -> two adder trees (MSB nibbles, LSB nibbles) -> "<<4" combine
// -> "<<1 & ACC" bit-serial shift-accumulator.
//
// Activations arrive MSB-FIRST, one bit per row per cycle, 8 cycles per
// INT8 x INT8 dot product ("accumulates partial products over 8 cycles").
// Cycle 0 carries the sign bit, so its partial sum is SUBTRACTED:
//     x = -x7*2^7 + x6*2^6 + ... + x0
//   acc <- -P7 ;  acc <- 2*acc + P6 ; ... ; acc <- 2*acc + P0
// ---------------------------------------------------------------------------
module cim_partition
  import cimple_pkg::*;
(
  input  logic                    clk_i,
  input  logic                    rst_ni,
  // --- write port ----------------------------------------------------------
  input  logic                    we_i,
  input  logic                    wbank_i,
  input  logic [ROW_AW-1:0]       waddr_i,
  input  logic [DATA_W-1:0]       wdata_i,
  // --- compute port --------------------------------------------------------
  input  logic                    bank_sel_i,
  input  logic [N_ROWS-1:0]       x_bits_i,     // one bit per row(xin), this cycle
  input  logic                    mac_en_i,     // advance the shift-accumulator
  input  logic                    mac_first_i,  // this cycle carries the sign bit
  output logic signed [MAC_W-1:0] acc_o
);

  logic signed [NIB_W-1:0] p_msb [N_ROWS];
  logic        [NIB_W-1:0] p_lsb [N_ROWS];

  for (genvar r = 0; r < N_ROWS; r++) begin : g_macro
    cim_macro u_macro (
      .clk_i,
      .we_i       (we_i && (waddr_i == ROW_AW'(r))),
      .wbank_i,
      .wdata_i,
      .bank_sel_i,
      .x_bit_i    (x_bits_i[r]),
      .p_msb_o    (p_msb[r]),
      .p_lsb_o    (p_lsb[r])
    );
  end

  // --- adder trees ---------------------------------------------------------
  // Written as a reduction; synthesis infers the balanced tree. Range checks:
  //   MSB tree: 64 * [-8, 7]  -> [-512, 448]  -> 10b signed   ("10b x 32")
  //   LSB tree: 64 * [ 0,15]  -> [   0, 960]  -> 10b unsigned ("10b x 32")

  logic signed [TREE_W-1:0] sum_msb;
  logic [TREE_W-1: 0] sum_lsb;

  always_comb begin
    automatic logic signed [TREE_W-1:0] acc_m = '0;
    automatic logic [TREE_W-1:0] acc_l = '0;
    for (int unsigned i = 0; i < N_ROWS;i++ ) begin
      acc_m += TREE_W'($signed(p_msb[i]));
      acc_l += TREE_W'((p_lsb[i]));
    end
    sum_msb = acc_m;
    sum_lsb = acc_l;
  end

  // --- "<<4" combine -------------------------------------------------------
  //   (sum_msb << 4) + sum_lsb  ->  [-8192, 8128] -> 15b signed ("15b x 32")

  logic signed [COMB_W-1: 0] comb_val;
  assign comb_val = (COMB_W'($signed(sum_msb) <<< NIB_W)) + COMB_W'($signed({1'b0, sum_lsb}));
  // --- "<<1 & ACC" bit-serial shift-accumulator ----------------------------
  logic signed [MAC_W-1:0] acc_q;
  always_ff @( posedge clk_i or negedge rst_ni ) begin
    if (!rst_ni) begin
      acc_q <= '0;
    end else if (mac_en_i) begin
      if (mac_first_i) acc_q <= -MAC_W'(comb_val);
      else acc_q <= (acc_q <<< 1) + MAC_W'(comb_val);
    end
  end

  assign acc_o = acc_q;
endmodule
