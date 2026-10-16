// ---------------------------------------------------------------------------
// cimple_seq.sv -- the mapping sequencer (the part the paper only describes
//                  in prose, Section IV-C / IV-D / IV-E)
//
// JOB: turn "run one attention layer" into the per-cycle control signals that
//      cim_top needs. The datapath already works; this makes it an accelerator.
//
// ATTENTION STAGES (paper Fig. 3):
//   1. WEIGHT PROJECTION   Q,K,V = X * Wq,Wk,Wv   weights stationary in the CIM
//   2. QK^T                Q (or K^T) stored in the CIM, the other streamed in
//   3. SOFTMAX             row-wise, via split_softmax
//   4. A' V                softmax output multiplied by V
//
// TWO MAPPINGS:
//   ENCODER-ONLY  [P] Q and V stay in the CIM, K^T is streamed as input;
//                 whole sequence processed in parallel; softmax result is
//                 written back as WEIGHTS (dest_i = 0).
//   DECODER-ONLY  [P] CIM stores K^T and V; the input buffer holds Q and QK^T;
//                 ONE token at a time; softmax result becomes the next
//                 ACTIVATIONS (dest_i = 1); old K,V come from a cache.
//
// DECISIONS YOU HAVE TO MAKE AND WRITE DOWN (design-decision log, M1):  [A]
//   - Bank policy: is bank 1 double-buffering the next weight tile, or does
//     it hold K vs V? The paper does not say.
//   - Tiling: head dim vs the 64-row x 32-column array. What happens when
//     d_model/h is not 64?
//   - How many cycles between the last score and the reciprocal scaling.
//
// STATE MACHINE (suggested starting point, change it if you can justify it):
//
//   IDLE -> LOAD_W -> PROJECT -> QKT -> SOFTMAX -> AV -> DRAIN -> DONE
//
//   LOAD_W   write weights: we_i, wbank_i, wpart_i, waddr_i, wdata_i
//   PROJECT  8 cycles of bit-serial MAC (mac_en_i, mac_first_i on cycle 0),
//            then walk out_phase 0..7 to read all 32 results (out_adv_i)
//   QKT      same, with the other operand streaming in
//   SOFTMAX  row_start_i ... z stream ... row_done_i
//   AV       MAC with the exp values as activations
//   DRAIN    flush wdata_select, wait for valid flags
//
// COUNTERS YOU WILL NEED:
//   - 3b bit counter (8 cycles per MAC)
//   - 3b output-phase counter mirrors cim_core.out_phase_o
//   - tile counters (row tile, column tile)
//   - token counter (decoder-only)
//
// VERIFY AGAINST: a numpy attention reference with the SAME quantisation,
//   one layer of a small BERT or TinyLlama, not a toy.
// ---------------------------------------------------------------------------
module cimple_seq
  import cimple_pkg::*;
(
  input  logic                          clk_i,
  input  logic                          rst_ni,

  // --- command from outside -------------------------------------------------
  input  logic                          start_i,
  input  logic                          model_type_i,   // 0: encoder-only, 1: decoder-only
  input  logic [15:0]                   n_tokens_i,
  input  logic [15:0]                   d_head_i,
  output logic                          busy_o,
  output logic                          done_o,

  // --- control to cim_top (same names as cim_top ports) ---------------------
  output logic                          we_o,
  output logic                          wbank_o,
  output logic [PART_AW-1:0]            wpart_o,
  output logic [ROW_AW-1:0]             waddr_o,
  output logic                          bank_sel_o,
  output logic                          mac_en_o,
  output logic                          mac_first_o,
  output logic                          out_adv_o,
  output logic                          out_clr_o,
  output logic                          mode_o,
  output logic                          acc_en_o,
  output logic                          acc_first_o,
  output logic [5:0]                    acc_addr_o,
  output logic                          row_start_o,
  output logic                          row_done_o,
  output logic                          dest_o,

  // --- feedback from the datapath -------------------------------------------
  input  logic [CNT_W-1:0]              out_phase_i,
  input  logic                          wbl_valid_i,
  input  logic                          xin_valid_i
);

  // TODO: state encoding (typedef enum logic [N:0])
  // TODO: state register (always_ff)
  // TODO: next-state logic (always_comb)
  // TODO: counters
  // TODO: output decode per state

endmodule
