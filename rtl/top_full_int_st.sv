// top_full_int.sv
//
// Full pipeline: integer geo, integer line buffer, integer patch extractor,
// integer MLP front end. post_mlp is still FP32 -- that is the only float
// left, and the only conversion in the design sits on the 3-wide residual
// path inside photo_stage_int.
//
//   top_geo_int --u8--> inter_line_buffer --u8--> patch_extractor_int
//        --u8 f[29]--> photo_stage_int { nn_pipelined_int --fp32 z-->
//                                        post_mlp <--fp32-- u8_to_fp32 }
//
// NO conversion tables anywhere:
//   - dequant_lut.hex   DELETED. uint8 v means v/255; in Q0.24 that is
//                       {v,v,v} (v * 0x010101 = v * 65793 = v/255 * 2^24).
//                       Byte replication. Zero logic. nn_pipelined_int does
//                       this in place of feat_to_fixed.
//   - coord_norm.hex    DELETED. linspace(0,1) over 256 points IS x/255, the
//                       same mapping, so the quantised coord feature is the
//                       raw index. patch_extractor_int drops its coord ROM
//                       and emits f[0]=x, f[1]=y directly.
//
//   *** Both identities assume IMG_WIDTH = IMG_HEIGHT = 256. If either
//       dimension changes, coord normalisation stops being /255 and the
//       coord path needs a real scale again. ***
//
// Accuracy vs the fp32 build: the old path was fp32(v/255) -> feat_to_fixed
// (truncating shift) -> Q0.24. The new path is exact byte replication. They
// differ by at most 1 LSB at 2^-24, which the 12-bit requant erases. Expect
// bit-identical pixels, not bit-identical intermediates.
// ---------------------------------------------------------------------------

module top_full_int_st # (
    parameter int BIT_WIDTH_FP   = 32,   // fp32 width, post_mlp side only
    parameter int BIT_WIDTH_UINT = 8,    // pixel width: geo -> buffer -> PE -> MLP
    parameter int IND_WIDTH      = 8,
    parameter int IMG_WIDTH      = 256,
    parameter int IMG_HEIGHT     = 256,
    parameter int NUM_LINES      = 4,
    parameter int WINDOW         = 3,
    parameter int IN_DIM         = 29,

    parameter LUT_FILE = "data_files/lut_quant.hex",   // geometry LUT, Q8.8
    parameter IMAGE_EE = "data_files/rom_ee.hex",
    parameter IMAGE_EO = "data_files/rom_eo.hex",
    parameter IMAGE_OE = "data_files/rom_oe.hex",
    parameter IMAGE_OO = "data_files/rom_oo.hex"
) (
    input  logic clock,
    input  logic reset,
    output logic frame_done,

    output logic                    out_valid,
    input  logic                    out_ready,
    output logic [BIT_WIDTH_UINT-1:0] final_rgb [2:0]
);

    // =====================================================================
    // geo stage -- integer
    // =====================================================================
    logic                      geo_valid;
    logic [IND_WIDTH-1:0]      out_x, out_y;
    logic [BIT_WIDTH_UINT-1:0] P_out [2:0];
    logic                      out_row_last;

    logic wr_line_ready, wr_line_valid, stall_geo, frame_over;

    // latch once geo has committed its last line, so coord_gen's wrap cannot
    // push a second frame's lines into the buffer on top of unread ones
    always_ff @(posedge clock)
        if (reset)                            frame_over <= 1'b0;
        else if (frame_done && wr_line_ready) frame_over <= 1'b1;

    assign wr_line_valid = geo_valid && out_row_last && !frame_over;
    assign stall_geo     = wr_line_valid && !wr_line_ready;

    top_geo_int #(
        .BIT_WIDTH_FP(BIT_WIDTH_FP), .BIT_WIDTH_UINT(BIT_WIDTH_UINT),
        .IND_WIDTH(IND_WIDTH), .IMG_WIDTH(IMG_WIDTH), .IMG_HEIGHT(IMG_HEIGHT),
        .LUT_FILE(LUT_FILE),
        .IMAGE_EE(IMAGE_EE), .IMAGE_EO(IMAGE_EO),
        .IMAGE_OE(IMAGE_OE), .IMAGE_OO(IMAGE_OO)
    ) u_geo (
        .clock(clock), .reset(reset),
        .stall(stall_geo),
        .out_valid(geo_valid), .out_x(out_x), .out_y(out_y),
        .P_out(P_out),               
                                  
        .frame_done(frame_done),
        .out_row_last(out_row_last)
    );

    // =====================================================================
    // inter-line buffer -- uint8 in, uint8 out.
    // RTL unchanged: BIT_WIDTH_FP is just a data width. 4x less line RAM.
    // =====================================================================
    logic                           wr_pixel_en;
    logic [BIT_WIDTH_UINT-1:0]      wr_pixel [3];
    logic [$clog2(NUM_LINES+1)-1:0] line_count;

    logic                           rd_line_consume;
    logic [$clog2(IMG_WIDTH)-1:0]   rd_col;
    logic [$clog2(NUM_LINES)-1:0]   rd_line_sel;
    logic [BIT_WIDTH_UINT-1:0]      rd_pixel [3];

    assign wr_pixel_en = geo_valid && !frame_over;
    assign wr_pixel[0] = P_out[0];
    assign wr_pixel[1] = P_out[1];
    assign wr_pixel[2] = P_out[2];

    inter_line_buffer #(
        .BIT_WIDTH_FP(BIT_WIDTH_UINT),        // data width only: 32b -> 8b
        .IMG_WIDTH(IMG_WIDTH),
        .NUM_LINES(NUM_LINES), .WINDOW(WINDOW)
    ) u_inter_line_buffer (
        .clock(clock), .reset(reset),
        .wr_line_valid(wr_line_valid),
        .wr_line_ready(wr_line_ready),
        .wr_pixel_en(wr_pixel_en),
        .wr_col(out_x[$clog2(IMG_WIDTH)-1:0]),
        .wr_pixel(wr_pixel),
        .rd_lines_valid(),                    // line_count is what PE uses
        .rd_line_consume(rd_line_consume),
        .rd_col(rd_col),
        .rd_line_sel(rd_line_sel),
        .rd_pixel(rd_pixel),
        .line_count(line_count)
    );

    // =====================================================================
    // patch extractor -- uint8 in, uint8 f[29] out, no coord ROM
    // =====================================================================
    logic                      pe_valid, photo_ready;
    logic [BIT_WIDTH_UINT-1:0] f [IN_DIM-1:0];

    patch_extractor_int #(
        .PIX_W(BIT_WIDTH_UINT),
        .IMG_WIDTH(IMG_WIDTH), .IMG_HEIGHT(IMG_HEIGHT),
        .NUM_LINES(NUM_LINES), .IN_DIM(IN_DIM)
    ) u_pe (
        .clock(clock), .reset(reset),
        .line_count(line_count), .rd_line_consume(rd_line_consume),
        .rd_col(rd_col), .rd_line_sel(rd_line_sel), .rd_pixel(rd_pixel),
        .out_valid(pe_valid), .out_ready(photo_ready), .f(f),
        .frame_done()                  // geo's frame_done is the exported one
    );

    // =====================================================================
    // photo stage -- uint8 feature port, fp32 out. Internally:
    //   nn_pipelined_int : u8 -> Q0.24 by replication, fp32 z out (unchanged)
    //   side FIFO        : now 24b instead of 96b
    //   u8_to_fp32 x3    : the only float conversion left, on the residual
    //   post_mlp         : untouched, still fp32
    // =====================================================================
    photo_stage_int_v4 #(
        .IN_DIM(IN_DIM), .FEAT_W(BIT_WIDTH_UINT), .BW(BIT_WIDTH_FP),
        .SIDE_DEPTH(8)
    ) u_photo (
        .clock(clock), .reset(reset),
        .in_valid(pe_valid), .in_ready(photo_ready), .f(f),
        .out_valid(out_valid), .out_ready(out_ready), .final_rgb(final_rgb)
    );

endmodule