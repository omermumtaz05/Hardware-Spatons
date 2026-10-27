module patch_extractor_int #(
    parameter int PIX_W =           8,
    parameter int IMG_WIDTH      = 256,
    parameter int IMG_HEIGHT     = 256,
    parameter int NUM_LINES      = 4,
    parameter int IN_DIM         = 29
) (
    input  logic clock,
    input  logic reset,

    input  logic [$clog2(NUM_LINES+1)-1:0] line_count,
    output logic                           rd_line_consume,
    output logic [$clog2(IMG_WIDTH)-1:0]   rd_col,
    output logic [$clog2(NUM_LINES)-1:0]   rd_line_sel,
    input  logic [PIX_W-1:0]                  rd_pixel [3],

    output logic                           out_valid,
    input  logic                           out_ready,
    output logic [PIX_W-1:0]                  f [IN_DIM-1:0],
    output logic                           frame_done
);

    localparam int XW    = $clog2(IMG_WIDTH);
    localparam int YW    = $clog2(IMG_HEIGHT);
    localparam int LUT_N = (IMG_WIDTH > IMG_HEIGHT) ? IMG_WIDTH : IMG_HEIGHT;


    logic [PIX_W-1:0] win [0:26];          // 9 slots x 3 ch, raster order
    logic [XW-1:0] x;
    logic [YW-1:0] y;
    logic [3:0]    t;
    logic [3:0]    ta, tc;
    logic [1:0]    wr_idx, wc_idx;
    logic [2:0]    need;
    logic          lines_ok;

    typedef enum logic [1:0] {S_WAIT, S_READ, S_OUT, S_DONE} state_t;
    state_t state;

    assign ta = (t < 4'd9) ? t : 4'd0;
    assign tc = t - 4'd1;

    always_comb begin
        unique case (ta)
            4'd0: begin wr_idx = 2'd0; wc_idx = 2'd0; end
            4'd1: begin wr_idx = 2'd0; wc_idx = 2'd1; end
            4'd2: begin wr_idx = 2'd0; wc_idx = 2'd2; end
            4'd3: begin wr_idx = 2'd1; wc_idx = 2'd0; end
            4'd4: begin wr_idx = 2'd1; wc_idx = 2'd1; end
            4'd5: begin wr_idx = 2'd1; wc_idx = 2'd2; end
            4'd6: begin wr_idx = 2'd2; wc_idx = 2'd0; end
            4'd7: begin wr_idx = 2'd2; wc_idx = 2'd1; end
            default: begin wr_idx = 2'd2; wc_idx = 2'd2; end
        endcase
    end

    // reflection: mirror about edge, edge not repeated
    always_comb begin
        unique case (wc_idx)
            2'd0:    rd_col = (x == '0) ? XW'(1) : x - 1'b1;
            2'd1:    rd_col = x;
            default: rd_col = (x == XW'(IMG_WIDTH-1)) ? XW'(IMG_WIDTH-2) : x + 1'b1;
        endcase
    end

    always_comb begin
        if (y == '0)
            rd_line_sel = (wr_idx == 2'd1) ? 2'd0 : 2'd1;
        else if (y == YW'(IMG_HEIGHT-1))
            rd_line_sel = (wr_idx == 2'd1) ? 2'd1 : 2'd0;
        else
            rd_line_sel = wr_idx;
    end

    always_comb begin
        need     = (y == '0 || y == YW'(IMG_HEIGHT-1)) ? 3'd2 : 3'd3;
        lines_ok = (line_count >= need[$clog2(NUM_LINES+1)-1:0]);
    end

    // data for address sent at t-1 arrives now
    always_ff @(posedge clock) begin
        if (state == S_READ && t != 4'd0) begin
            win[tc*3 + 0] <= rd_pixel[0];
            win[tc*3 + 1] <= rd_pixel[1];
            win[tc*3 + 2] <= rd_pixel[2];
        end
    end

    always_ff @(posedge clock) begin
        if (reset) begin
            state <= S_WAIT; x <= '0; y <= '0; t <= '0;
            out_valid <= 1'b0; rd_line_consume <= 1'b0; frame_done <= 1'b0;
        end else begin
            rd_line_consume <= 1'b0;
            unique case (state)
                S_WAIT: if (lines_ok) begin t <= '0; state <= S_READ; end

                S_READ: if (t == 4'd9) begin
                            out_valid <= 1'b1; state <= S_OUT;
                        end else t <= t + 1'b1;

                S_OUT: if (out_ready) begin
                    out_valid <= 1'b0;
                    if (x == XW'(IMG_WIDTH-1)) begin
                        x <= '0;
                        if (y != '0) rd_line_consume <= 1'b1;
                        if (y == YW'(IMG_HEIGHT-1)) begin
                            frame_done <= 1'b1; state <= S_DONE;
                        end else begin
                            y <= y + 1'b1; state <= S_WAIT;
                        end
                    end else begin
                        x <= x + 1'b1; t <= '0; state <= S_READ;
                    end
                end

                default: ;
            endcase
        end
    end

    always_comb begin
        f[0] = PIX_W'(x);  f[1] = PIX_W'(y);
        for (int c = 0; c < 3; c++) begin
            f[2  + c] = win[4*3 + c];   // center
            f[5  + c] = win[0*3 + c];   // NW
            f[8  + c] = win[1*3 + c];   // N
            f[11 + c] = win[2*3 + c];   // NE
            f[14 + c] = win[3*3 + c];   // W
            f[17 + c] = win[5*3 + c];   // E
            f[20 + c] = win[6*3 + c];   // SW
            f[23 + c] = win[7*3 + c];   // S
            f[26 + c] = win[8*3 + c];   // SE
        end
    end

endmodule