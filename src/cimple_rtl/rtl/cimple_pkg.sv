// ---------------------------------------------------------------------------
// cimple_pkg.sv -- parameters for the CIMple self-attention datapath
//
// Reconstructed from: Ahn, Tao, Gomony, Geilen, Corporaal,
//   "CIMple: Standard-cell SRAM-based CIM with LUT-based split softmax
//    for attention acceleration", arXiv:2604.15944v2.
//
// See docs/01-spec-teardown.md for how every width below was derived from
// the annotations in Fig. 4 of the paper.
// ---------------------------------------------------------------------------
package cimple_pkg;

  // ---- CIM array geometry -------------------------------------------------
  // One partition = 64 macros. One macro = 16 bitcells + 8 OAIs = two INT8
  // weights (one per bank) sharing one activation bit. 64 macros x 2 banks
  // x 8b = 1024b = the paper's "two SRAM blocks of 512 bits".
  parameter int unsigned N_ROWS  = 64;   // input channels  (XIN width, 64b)
  parameter int unsigned N_PART  = 32;   // output channels ("32 partitions")
  parameter int unsigned N_BANKS = 2;    // Block0 / Block1, one active at a time
  parameter int unsigned DATA_W  = 8;    // INT8 weights and activations
  parameter int unsigned NIB_W   = 4;    // MSB / LSB nibble split

  // Total array = 2 banks * 32 part * 64 rows * 8b = 32768b = 32kb. [paper]

  // ---- Datapath widths (match Fig. 4 annotations) -------------------------
  parameter int unsigned TREE_W = 10;    // adder tree over 64 nibbles  -> "10b x 32"
  parameter int unsigned COMB_W = 15;    // (msb<<4) + lsb              -> "15b x 32"
  parameter int unsigned MAC_W  = 23;    // after 8-cycle shift-acc     -> "23b x 32"

  parameter int unsigned N_LANES    = 4;               // "23b x 4" at CIM OUT
  parameter int unsigned MUX_PHASES = N_PART/N_LANES;  // 8  -> 3b COUNTER
  parameter int unsigned CNT_W      = 3;

  parameter int unsigned ACC_W   = 32;   // intermediate accumulator / buffer
  parameter int unsigned QOUT_W  = 8;    // quantiser output (INT8)

  // ---- Softmax LUTs -------------------------------------------------------
  parameter int unsigned LUT_DEPTH = 256;  // "LUT e^x : 256 x 8b"
  parameter int unsigned LUT_W     = 8;
  parameter int unsigned SUM_W     = 24;   // exp-sum accumulator
  // Reciprocal LUT is indexed by the 8 bits below the leading 1 of the sum.
  parameter int unsigned RECIP_FRAC = 15;  // recip_lut[m] ~= 2^15 / (256+m)

  parameter int unsigned ROW_AW  = $clog2(N_ROWS);
  parameter int unsigned PART_AW = $clog2(N_PART);

endpackage
