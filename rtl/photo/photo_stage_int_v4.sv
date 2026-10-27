
module photo_stage_int_v4 #(
    parameter int IN_DIM     = 29,
    parameter int BW         = 32,   // internal zi width cap inside the MLP
    parameter int FEAT_W     = 8,    // uint8 features and pixels
    parameter int IDXW       = 12,   // delta-LUT address width; must match both children
    parameter int SIDE_DEPTH = 8     // >= max pixels in flight in the MLP
) (
    input  logic clock,
    input  logic reset,

    input  logic in_valid,
    output logic in_ready,
    input  logic [FEAT_W-1:0] f [IN_DIM-1:0],

    output logic out_valid,
    input  logic out_ready,
    output logic [FEAT_W-1:0] final_rgb [2:0]
);

    // ---- MLP handshake wires ----
    logic nn_iv, nn_ir, nn_ov, nn_or;
    logic [IDXW-1:0] z_idx [2:0];

    // ---- side fifo wires (carries centre RGB across MLP flight) ----
    logic sf_full, sf_empty, sf_wr, sf_rd;
    logic [3*FEAT_W-1:0] sf_wdata, sf_rdata;
    logic [FEAT_W-1:0]   center_u8 [2:0];

    // input accepted only if BOTH the MLP and the side fifo can take it
    assign in_ready = nn_ir && !sf_full;
    assign nn_iv    = in_valid && in_ready;
    assign sf_wr    = in_valid && in_ready;

    // centre RGB lives at f[2],f[3],f[4] of the 29-vector
    assign sf_wdata = {f[4], f[3], f[2]};

    nn_pipelined_int_v4 #(
        .IN_DIM(IN_DIM), .FEAT_W(FEAT_W), .BW(BW), .IDXW(IDXW)
    ) u_nn (
        .clock(clock), .reset(reset),
        .in_valid(nn_iv), .in_ready(nn_ir), .f(f),
        .out_valid(nn_ov), .out_ready(nn_or), .z_idx(z_idx)
    );

    fifo_residual #(.WIDTH(3*FEAT_W), .DEPTH(SIDE_DEPTH)) u_side_fifo (
        .clock(clock), .reset(reset),
        .wr_en(sf_wr), .wr_data(sf_wdata), .full(sf_full),
        .rd_en(sf_rd), .rd_data(sf_rdata), .empty(sf_empty)
    );

    // unpack FWFT head -- order must mirror sf_wdata above
    assign center_u8[0] = sf_rdata[1*FEAT_W-1 -: FEAT_W];   // f[2]
    assign center_u8[1] = sf_rdata[2*FEAT_W-1 -: FEAT_W];   // f[3]
    assign center_u8[2] = sf_rdata[3*FEAT_W-1 -: FEAT_W];   // f[4]

    // ---- post_mlp_int has ONE registered stage (the ROM read), so the stage
    //      carries a one-deep skid register and its own valid. ----
    logic p_valid, accept;

    assign accept = nn_ov && !sf_empty && (!p_valid || out_ready);

    always_ff @(posedge clock) begin
        if (reset)          p_valid <= 1'b0;
        else if (accept)    p_valid <= 1'b1;
        else if (out_ready) p_valid <= 1'b0;
    end

    assign nn_or = accept;      // drain the MLP...
    assign sf_rd = accept;      // ...and pop the matching centre together

    post_mlp_int #(
        .PIX_W(FEAT_W), .IDXW(IDXW)
    ) u_post (
        .clock(clock), .en(accept),
        .z_idx(z_idx), .center_u8(center_u8), .final_rgb(final_rgb)
    );

    assign out_valid = p_valid;

endmodule