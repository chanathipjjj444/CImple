// ---------------------------------------------------------------------------
// cim_top.sv -- the whole of Fig. 4: CIM core + ACC/Buffer + quant + softmax
//
// DATAFLOW (follow the arrows in the paper's Fig. 4):
//
//   XIN --> cim_core --cim_out(23b x4)--> acc_buffer --(32b x4)--+
//                |                                               |
//                +----------(23b x4 sign-extended)---------> quant mux
//                                                                |
//                                                       quant_unit x4 (32b -> 8b)
//                                                                |
//                                                  split_softmax (LUT e^x)
//                                                                |
//                                                          wdata_select
//                                                          /          \
//                                                 WBL (weights)   XINLB (activations)
//
// MODES:  [P] "The output of the CIM core is sent either to the intermediate
//             accumulator or to the quantization unit based on the mode of
//             operation."
//   mode_i = 0 : core output goes straight to quant
//   mode_i = 1 : core output is accumulated, quant reads the buffer
//
// WHAT THIS MODULE OWNS:
//   - instantiation + wiring only. NO control decisions here: every enable,
//     address and select comes in from the sequencer (cimple_seq.sv).
//
// WIRING TO WATCH:
//   - cim_out is 23b, quant_unit takes ACC_W (32b): sign-extend when it takes
//     the direct path.
//   - quant_unit is one per lane, so instantiate N_LANES of them in a
//     generate loop (like g_part in cim_core).
//   - split_softmax takes one score per cycle here, but there are 4 lanes.
//     Decide: 4 softmax instances, or 1 instance fed lane by lane. [A]
// ---------------------------------------------------------------------------
module cim_top
  import cimple_pkg::*;
(
  input  logic                          clk_i,
  input  logic                          rst_ni,

  // --- weight load (from the global buffer) -------------------------------
  input  logic                          we_i,
  input  logic                          wbank_i,
  input  logic [PART_AW-1:0]            wpart_i,
  input  logic [ROW_AW-1:0]             waddr_i,
  input  logic [DATA_W-1:0]             wdata_i,

  // --- activations ----------------------------------------------------------
  input  logic [N_ROWS-1:0]             x_bits_i,

  // --- control from the sequencer -------------------------------------------
  input  logic                          bank_sel_i,
  input  logic                          mac_en_i,
  input  logic                          mac_first_i,
  input  logic                          out_adv_i,
  input  logic                          out_clr_i,
  input  logic                          mode_i,         // see MODES above
  input  logic                          acc_en_i,
  input  logic                          acc_first_i,
  input  logic [5:0]                    acc_addr_i,
  input  logic [15:0]                   scale_i,        // quant multiplier
  input  logic [4:0]                    shift_i,        // quant shift
  input  logic                          row_start_i,    // softmax: clear sum
  input  logic                          row_done_i,     // softmax: last score seen
  input  logic                          dest_i,         // wdata_select

  // --- results --------------------------------------------------------------
  output logic [128-1:0]                wbl_o,
  output logic                          wbl_valid_o,
  output logic [N_ROWS-1:0]             xin_o,
  output logic                          xin_valid_o
);

  // TODO: cim_core
  // TODO: acc_buffer
  // TODO: quant mux (mode_i) + N_LANES x quant_unit
  // TODO: split_softmax
  // TODO: wdata_select

endmodule
