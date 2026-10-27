module top_full_synth # (
    parameter BIT_WIDTH_FP   = 32,
    parameter BIT_WIDTH_UINT = 8,
    parameter IND_WIDTH  = 8,
    parameter IMG_WIDTH  = 256,
    parameter IMG_HEIGHT = 256,
    parameter NUM_LINES  = 4,
    parameter WINDOW     = 3,
    parameter IN_DIM     = 29,
    parameter LUT_FILE = "data_files/lut_quant.hex",
    parameter IMAGE_EE = "data_files/rom_ee.hex",
    parameter IMAGE_EO = "data_files/rom_eo.hex",
    parameter IMAGE_OE = "data_files/rom_oe.hex",
    parameter IMAGE_OO = "data_files/rom_oo.hex"
) (
    input  logic clock,
    input  logic reset,

    output logic                        frame_done,
    output logic                        out_valid,
    input  logic                        out_ready,
    output logic [3*BIT_WIDTH_UINT-1:0] final_rgb_flat
);

    logic out_ready_r;
    always_ff @(posedge clock)
        out_ready_r <= out_ready;

    logic                      out_valid_c, frame_done_c;
    logic [BIT_WIDTH_UINT-1:0] final_rgb [2:0];

    top_full_int_st #(
        .BIT_WIDTH_FP(BIT_WIDTH_FP), .BIT_WIDTH_UINT(BIT_WIDTH_UINT),
        .IND_WIDTH(IND_WIDTH), .IMG_WIDTH(IMG_WIDTH), .IMG_HEIGHT(IMG_HEIGHT),
        .NUM_LINES(NUM_LINES), .WINDOW(WINDOW), .IN_DIM(IN_DIM),
        .LUT_FILE(LUT_FILE),
        .IMAGE_EE(IMAGE_EE), .IMAGE_EO(IMAGE_EO),
        .IMAGE_OE(IMAGE_OE), .IMAGE_OO(IMAGE_OO)
    ) u_top_full (
        .clock(clock), .reset(reset),
        .frame_done(frame_done_c),
        .out_valid(out_valid_c),
        .out_ready(out_ready_r),
        .final_rgb(final_rgb)
    );

    always_ff @(posedge clock) begin
        for (int j = 0; j < 3; j++)
            final_rgb_flat[j*BIT_WIDTH_UINT +: BIT_WIDTH_UINT] <= final_rgb[j];
        out_valid  <= out_valid_c;
        frame_done <= frame_done_c;
    end

endmodule