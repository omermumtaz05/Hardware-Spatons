
module LUT_fxp#(
  parameter BIT_WIDTH = 32,          // FP32 output width (unchanged for pipeline)
  parameter IND_WIDTH = 8,
  parameter Q_WIDTH   = 16,          // Q8.8 stored width per coordinate
  parameter LUT_FILE  = "data_files/lut_quant.hex"
)(
  input  [IND_WIDTH-1:0] x_ind,
  input  [IND_WIDTH-1:0] y_ind,
  input  clock,
  input  reset,

  input stall,
  output logic [Q_WIDTH-1:0] x_out,   // FP32 out — pipeline unchanged
  output logic [Q_WIDTH-1:0] y_out
);

  // LUT now stores 32-bit entries: {y_q88[15:0], x_q88[15:0]}  -> HALF the size
  logic [2*Q_WIDTH-1:0] LUT [0:(1 << (2*IND_WIDTH)) - 1];

  initial $readmemh(LUT_FILE, LUT);

  // registered read, NO reset (BRAM inference rule)
  logic [2*Q_WIDTH-1:0] read_word;
  always_ff @(posedge clock)
      if(!stall)
        read_word <= LUT[{y_ind, x_ind}];

  // split into two Q8.8 coordinates
  logic [Q_WIDTH-1:0] x_q88, y_q88;
  assign x_out = read_word[Q_WIDTH-1:0];
  assign y_out = read_word[2*Q_WIDTH-1:Q_WIDTH];





endmodule