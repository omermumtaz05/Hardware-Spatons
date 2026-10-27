import fp32_pkg::*;

module fetch_four_near #(
    parameter BIT_WIDTH = 32
) (
    input  logic [BIT_WIDTH-1:0]   x_int,
    input  logic [BIT_WIDTH-1:0]   y_int,

    output logic [2*BIT_WIDTH-1:0] A_00,
    output logic [2*BIT_WIDTH-1:0] A_01,
    output logic [2*BIT_WIDTH-1:0] A_10,
    output logic [2*BIT_WIDTH-1:0] A_11
);

    // FP32 constant 1.0 = 0x3F800000
    localparam logic [BIT_WIDTH-1:0] FP_ONE = 32'h3F80_0000;

    //for clamping
    localparam FP_MAX_X = 32'h437F0000; // 255.0
    localparam FP_MAX_Y = 32'h437F0000;

    logic [BIT_WIDTH-1:0] x_plus1; // logic values for x + 1 and y + 1
    logic [BIT_WIDTH-1:0] y_plus1;

    logic boundary_x, boundary_y;

    assign boundary_x = (x_int >= FP_MAX_X);
    assign boundary_y = (y_int >= FP_MAX_Y);

    // x+1 and y+1 in FP32
    // clamped if at or beyond the boundary
    assign x_plus1 = boundary_x ? FP_MAX_X : fp32_add(x_int, FP_ONE);
    assign y_plus1 = boundary_y ? FP_MAX_Y : fp32_add(y_int, FP_ONE);

    // pack {y, x}: y in high half, x in low half
    assign A_00 = {y_int,   x_int};    // (x,   y)
    assign A_01 = {y_plus1, x_int};    // (x,   y+1)
    assign A_10 = {y_int,   x_plus1};  // (x+1, y)
    assign A_11 = {y_plus1, x_plus1};  // (x+1, y+1)

endmodule