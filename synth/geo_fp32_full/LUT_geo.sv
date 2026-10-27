module MLP_geo_LUT#(
  parameter BIT_WIDTH = 32,
  parameter IND_WIDTH = 8,
  parameter LUT_FILE = "data_files/lut.hex"
)(
  
  input [IND_WIDTH - 1:0] x_ind,
  input [IND_WIDTH - 1:0] y_ind,
  input clock,
  input reset,
  input stall,
  
  output logic [BIT_WIDTH - 1:0] x_out,
  output logic [BIT_WIDTH - 1:0] y_out

);
  

  
  logic [2*BIT_WIDTH - 1:0] LUT [0:(1 << (2*IND_WIDTH)) - 1]; // initialize reg array of 64 bit width and 256^2 no. of elements
  
  initial $readmemh(LUT_FILE, LUT); // initialize reg with preloaded values
                    
  //logic [2*BIT_WIDTH - 1:0] entry; 
  
  logic [2*BIT_WIDTH - 1:0] read; 

  always_ff @(posedge clock)
    if(!stall)
          read <= LUT[{y_ind, x_ind}];   // one read, full 64-bit word

  assign x_out = read[  BIT_WIDTH-1 : 0];
  assign y_out = read[2*BIT_WIDTH-1 : BIT_WIDTH];

  
  
endmodule
