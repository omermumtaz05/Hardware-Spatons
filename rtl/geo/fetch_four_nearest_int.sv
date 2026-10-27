module ffn_int #(
    parameter int IND_WIDTH  = 8,
    parameter int IMG_WIDTH  = 256,
    parameter int IMG_HEIGHT = 256
) (
    input  logic [IND_WIDTH-1:0]   x_int,
    input  logic [IND_WIDTH-1:0]   y_int,
 
    output logic [2*IND_WIDTH-1:0] A_00,   // (x  , y  )
    output logic [2*IND_WIDTH-1:0] A_01,   // (x  , y+1)
    output logic [2*IND_WIDTH-1:0] A_10,   // (x+1, y  )
    output logic [2*IND_WIDTH-1:0] A_11    // (x+1, y+1)
);
 
    localparam logic [IND_WIDTH-1:0] XMAX = IND_WIDTH'(IMG_WIDTH  - 1);
    localparam logic [IND_WIDTH-1:0] YMAX = IND_WIDTH'(IMG_HEIGHT - 1);
 
    logic [IND_WIDTH-1:0] x_plus1, y_plus1;
 
    assign x_plus1 = (x_int >= XMAX) ? XMAX : x_int + 1'b1;
    assign y_plus1 = (y_int >= YMAX) ? YMAX : y_int + 1'b1;
 
    // pack {y, x}: y in the high half, x in the low half
    assign A_00 = {y_int,   x_int};
    assign A_01 = {y_plus1, x_int};
    assign A_10 = {y_int,   x_plus1};
    assign A_11 = {y_plus1, x_plus1};
 
endmodule