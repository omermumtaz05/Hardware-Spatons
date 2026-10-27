
module bilinear_interp_int #(
    parameter BIT_WIDTH_FP = 32,
    parameter BIT_WIDTH_INT = 8,
    parameter BIT_WIDTH_D = 8
) (

    input clock,
    input reset,
    input stall,

    input  logic [BIT_WIDTH_INT-1:0] P_00 [2:0],
    input  logic [BIT_WIDTH_INT-1:0] P_01 [2:0],
    input  logic [BIT_WIDTH_INT-1:0] P_10 [2:0],
    input  logic [BIT_WIDTH_INT-1:0] P_11 [2:0],

    input  logic [BIT_WIDTH_D-1:0] dx,
    input  logic [BIT_WIDTH_D-1:0] dy,

    output logic [BIT_WIDTH_INT-1:0] P_out [2:0]
);

    localparam int WW    = 2*BIT_WIDTH_D + 1;                    // 17
    localparam int AW    = 2*BIT_WIDTH_D + BIT_WIDTH_INT + 1;    // 25
    localparam int SHIFT = 2*BIT_WIDTH_D;                        // 16
    localparam logic [AW-1:0] HALF = AW'(1) << (SHIFT-1);        // 32768

    // ---- stage 1: corner weights, and forward the pixels one cycle ----
    // mdx needs BIT_WIDTH_D+1 bits: 256-0 = 256 does not fit in 8.
    logic [BIT_WIDTH_D:0] mdx, mdy;
    assign mdx = (BIT_WIDTH_D+1)'(1 << BIT_WIDTH_D) - {1'b0, dx};
    assign mdy = (BIT_WIDTH_D+1)'(1 << BIT_WIDTH_D) - {1'b0, dy};

    logic [WW-1:0]            w00_q, w01_q, w10_q, w11_q;
    logic [BIT_WIDTH_INT-1:0] p00_1 [2:0], p01_1 [2:0], p10_1 [2:0], p11_1 [2:0];

    always_ff @(posedge clock) begin
        if (reset) begin
            w00_q <= '0; w01_q <= '0; w10_q <= '0; w11_q <= '0;
            for (int c = 0; c < 3; c++) begin
                p00_1[c] <= '0; p01_1[c] <= '0; p10_1[c] <= '0; p11_1[c] <= '0;
            end
        end else if (!stall) begin
            w00_q <= mdx          * mdy;            // (1-dx)(1-dy)
            w10_q <= {1'b0, dx}   * mdy;            //   dx (1-dy)
            w01_q <= mdx          * {1'b0, dy};     // (1-dx)  dy
            w11_q <= {1'b0, dx}   * {1'b0, dy};     //   dx    dy
            for (int c = 0; c < 3; c++) begin
                p00_1[c] <= P_00[c]; p01_1[c] <= P_01[c];
                p10_1[c] <= P_10[c]; p11_1[c] <= P_11[c];
            end
        end
    end

    // ---- stage 2: 12 products + adder tree, per channel ----
    logic [AW-1:0] acc_q [2:0];
    logic [AW-1:0] m00_q [2:0], m01_q[2:0], m10_q[2:0], m11_q [2:0];

    always_ff @(posedge clock) begin
        if (reset)
            for (int c = 0; c < 3; c++) 
            begin
                m00_q[c] <= '0; m01_q[c] <= '0; 
                m10_q[c] <= '0; m11_q[c] <= '0;

            end
        else if (!stall)    
            
            for (int c = 0; c < 3; c++)
            begin
                m00_q[c] <= w00_q * p00_1[c]; m01_q[c] <= w01_q * p01_1[c]; 
                m10_q[c] <= w10_q * p10_1[c]; m11_q[c] <= w11_q * p11_1[c];

            end
    end

    always_ff @(posedge clock) begin
        if (reset)
            for (int c = 0; c < 3; c++) 
            begin
                acc_q[c] <= '0;
            end
        else if (!stall)    
            
            for (int c = 0; c < 3; c++)
            begin
                acc_q[c] <= AW'(m00_q[c]) + AW'(m01_q[c])
                          + AW'(m10_q[c]) + AW'(m11_q[c]);
            end
    end
    
    // ---- stage 3: round to nearest, normalise ----
    // Worst case (65536*255 + 32768) >> 16 == 255 exactly, so no clamp.
    always_ff @(posedge clock) begin
        if (reset)
            for (int c = 0; c < 3; c++) P_out[c] <= '0;
        else if (!stall)
            for (int c = 0; c < 3; c++)
                P_out[c] <= BIT_WIDTH_INT'((acc_q[c] + HALF) >> SHIFT);
    end

endmodule