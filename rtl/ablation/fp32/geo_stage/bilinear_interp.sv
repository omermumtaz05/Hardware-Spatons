import fp32_pkg::*;

module bilinear_interp #(
    parameter BIT_WIDTH = 32
) (
    input  logic [BIT_WIDTH-1:0] P_00 [2:0],
    input  logic [BIT_WIDTH-1:0] P_01 [2:0],
    input  logic [BIT_WIDTH-1:0] P_10 [2:0],
    input  logic [BIT_WIDTH-1:0] P_11 [2:0],

    input  logic [BIT_WIDTH-1:0] dx,
    input  logic [BIT_WIDTH-1:0] dy,

    output logic [BIT_WIDTH-1:0] P_out [2:0]
);

    localparam logic [BIT_WIDTH-1:0] FP_ONE = 32'h3F80_0000;

    // weights — shared across all three channels, compute once
    logic [BIT_WIDTH-1:0] one_m_dx, one_m_dy;
    logic [BIT_WIDTH-1:0] w00, w01, w10, w11;

    assign one_m_dx = fp32_sub(FP_ONE, dx);   // 1 - dx
    assign one_m_dy = fp32_sub(FP_ONE, dy);   // 1 - dy

    assign w00 = fp32_mul(one_m_dx, one_m_dy);  // (1-dx)(1-dy)
    assign w10 = fp32_mul(dx,       one_m_dy);  //   dx (1-dy)
    assign w01 = fp32_mul(one_m_dx, dy);        // (1-dx)  dy
    assign w11 = fp32_mul(dx,       dy);        //   dx    dy

    // per-channel accumulation
    genvar c;
    generate
        for (c = 0; c < 3; c++) begin : channel

            //intermediary buffers for getting sums
            logic [BIT_WIDTH-1:0] t00, t01, t10, t11;
            logic [BIT_WIDTH-1:0] sum0, sum1;

            //assign mult values to be summed later
            assign t00 = fp32_mul(w00, P_00[c]);
            assign t10 = fp32_mul(w10, P_10[c]);
            assign t01 = fp32_mul(w01, P_01[c]);
            assign t11 = fp32_mul(w11, P_11[c]);

            //sum0 and sum1 computed in parralel then combined
            assign sum0 = fp32_add(t00, t10);
            assign sum1 = fp32_add(t01, t11);
            assign P_out[c] = fp32_add(sum0, sum1);
        end
    endgenerate

endmodule