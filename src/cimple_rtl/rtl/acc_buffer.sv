// ---------------------------------------------------------------------------
// acc_buffer.sv -- intermediate ACC + Buffer after "CIM OUT" (Fig. 4, blue)
//
// WHAT THE PAPER SHOWS:
//   CIM OUT delivers 23b x 4 lanes. The ACC widens that to 32b x 4 and adds to
//   the value already stored, so a dot product longer than the 64 rows of one
//   pass (or an A'V sum over several tiles) can be built up over several
//   passes. The stacked Buffer keeps those partial sums between passes.
//
// WHAT THE PAPER DOES NOT SAY:  [A]
//   - Buffer depth. It bounds the largest head dimension you can process
//     without spilling to the global buffer. Make it a parameter.
//   - Whether "first pass" overwrites or clears-then-adds. Pick one, record it.
//
// BEHAVIOUR TO IMPLEMENT:
//   acc_en_i = 0           : hold
//   acc_en_i = 1, acc_first_i = 1 : buf[addr] <= sign-extended cim_in
//   acc_en_i = 1, acc_first_i = 0 : buf[addr] <= buf[addr] + sign-extended cim_in
//   rd_o always shows buf[addr_i] for the 4 lanes (needed by the quant mux).
//
// CHECK: 23b -> 32b must SIGN-extend. Think about what happens to a negative
//        partial sum if you zero-extend it.
// ---------------------------------------------------------------------------

module acc_buffer
  import cimple_pkg::*;
#(
  parameter int unsigned DEPTH = 64          // [A] number of 4-lane entries
)(
  input  logic                          clk_i,
  input  logic                          rst_ni,

  // --- from the CIM core ---------------------------------------------------
  input  logic signed [MAC_W-1:0]       cim_in_i [N_LANES],
  input  logic                          acc_en_i,     // take cim_in this cycle
  input  logic                          acc_first_i,  // overwrite, do not add
  input  logic [$clog2(DEPTH)-1:0]      addr_i,

  // --- to the quant mux ----------------------------------------------------
  output logic signed [ACC_W-1:0]       rd_o [N_LANES]
);

  logic [ACC_W-1:0] buff [DEPTH][N_LANES];
  always_ff @( posedge clk_i ) begin
    if (acc_en_i) begin
      for (int l = 0;l < N_LANES ;l++ ) begin
        if (acc_first_i) buff[addr_i][l] <= ACC_W'(cim_in_i[l]);
        else buff[addr_i][l] <= buff[addr_i][l] + ACC_W'(cim_in_i[l]);
      end
    end
  end

  

  for (genvar l =0 ;l < N_LANES ;l++ ) begin: g_assign
    assign rd_o[l] = buff[addr_i][l];
  end

endmodule
