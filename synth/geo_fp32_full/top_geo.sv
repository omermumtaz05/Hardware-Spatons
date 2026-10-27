import fp32_pkg::*;

module top_geo # (
    parameter BIT_WIDTH_FP   = 32,
    parameter BIT_WIDTH_UINT = 8,
    parameter IND_WIDTH = 8,
    parameter IMG_WIDTH = 256,
    parameter IMG_HEIGHT = 256,
    parameter LUT_FILE = "data_files/lut_quant.hex",
    parameter DEQUANT_LUT = "data_files/dequant_lut.hex",
    parameter IMAGE_FILE = "data_files/image.hex",

    parameter IMAGE_EE = "data_files/rom_ee.hex",
    parameter IMAGE_EO = "data_files/rom_eo.hex",
    parameter IMAGE_OE = "data_files/rom_oe.hex",
    parameter IMAGE_OO = "data_files/rom_oo.hex"
) (
    input clock,
    input reset,
    input stall,                                     // NEW: freeze whole geo pipe

    output logic out_valid,
    output logic [IND_WIDTH - 1:0] out_x, out_y,
    output logic [BIT_WIDTH_FP-1:0] P_out [2:0],
    output logic frame_done,
    output logic out_row_last                        // NEW: this valid pixel is last-in-row
);

// ---- stall fans out to every register below ----

// valid chain
// valid chain: add a third stage
logic valid_1, valid_2, valid_3;
always_ff @(posedge clock) begin
    if (reset)       begin valid_1 <= 1'b0; valid_2 <= 1'b0; valid_3 <= 1'b0; end
    else if (!stall) begin valid_1<=1'b1; valid_2<=valid_1; valid_3<=valid_2; end
end

// coord pipeline (delay-matched to P_out)
logic [IND_WIDTH-1:0] x_ind, y_ind;
logic [IND_WIDTH-1:0] x_ind_1, y_ind_1, x_ind_2, y_ind_2, x_ind_3, y_ind_3;

always_ff @(posedge clock)
    if (reset)       begin x_ind_1 <= '0; y_ind_1 <= '0; end
    else if (!stall) begin x_ind_1 <= x_ind;   y_ind_1 <= y_ind;   end

always_ff @(posedge clock)
    if (reset)       begin x_ind_2 <= '0; y_ind_2 <= '0; end
    else if (!stall) begin x_ind_2 <= x_ind_1; y_ind_2 <= y_ind_1; end

always_ff @(posedge clock)
    if (reset)       begin x_ind_3<='0; y_ind_3<='0; end
    else if (!stall) begin x_ind_3<=x_ind_2; y_ind_3<=y_ind_2; end

// row-last marker, delay-matched alongside the coords (same 2-stage delay)
logic row_last_0, row_last_1, row_last_2, row_last_3;

always_ff @(posedge clock)
    if (reset)       begin row_last_1 <= 1'b0; row_last_2 <= 1'b0; row_last_3 <= 1'b0; end
    else if (!stall) begin row_last_1 <= row_last_0; row_last_2 <= row_last_1; row_last_3 <= row_last_2; end

logic [BIT_WIDTH_FP - 1:0] x_LUT, y_LUT;
logic [BIT_WIDTH_FP-1:0] dx_dd, dy_dd;

logic [BIT_WIDTH_FP - 1:0] dx_d, dy_d;
logic [BIT_WIDTH_FP - 1:0] dx, dy;
logic [BIT_WIDTH_FP - 1:0] x_int, y_int;
logic [2*BIT_WIDTH_FP - 1:0] A_00, A_01, A_10, A_11;
logic [BIT_WIDTH_FP - 1:0] P_00 [2:0], P_01 [2:0], P_10 [2:0], P_11 [2:0];

coord_gen #(.IMG_HEIGHT(IMG_HEIGHT), .IMG_WIDTH(IMG_WIDTH), .IND_WIDTH(IND_WIDTH))
u_coord_gen (
    .clock(clock), .reset(reset), .stall(stall),     // NEW: stall in
    .x_out(x_ind), .y_out(y_ind),
    .frame_done(frame_done),
    .row_last(row_last_0)                             // NEW: row-boundary marker
);

MLP_geo_LUT #(.BIT_WIDTH(BIT_WIDTH_FP), .IND_WIDTH(IND_WIDTH), .LUT_FILE("data_files/lut_fp32.hex"))
u_geo_lut (
    .x_ind(x_ind), .y_ind(y_ind), .clock(clock), .reset(reset), .stall(stall),
    .x_out(x_LUT), .y_out(y_LUT)
);

coordsplit_format #(.BIT_WIDTH(BIT_WIDTH_FP)) u_coord_split (
    .x_in(x_LUT), .y_in(y_LUT),
    .x_int(x_int), .y_int(y_int), .dx(dx), .dy(dy)
);

fetch_four_near #(.BIT_WIDTH(BIT_WIDTH_FP)) u_fetch_four (
    .x_int(x_int), .y_int(y_int),
    .A_00(A_00), .A_01(A_01), .A_10(A_10), .A_11(A_11)
);

input_buffer_parity #(BIT_WIDTH_FP, BIT_WIDTH_UINT, IMG_WIDTH, IMG_HEIGHT, DEQUANT_LUT, IMAGE_EE, IMAGE_EO, IMAGE_OE, IMAGE_OO)
u_input_buffer (
    .clock(clock), .reset(reset), .stall(stall),
    .addr_00(A_00), .addr_01(A_01), .addr_10(A_10), .addr_11(A_11), 
    .P_00(P_00), .P_01(P_01), .P_10(P_10), .P_11(P_11)
);

always_ff @(posedge clock)
    if (reset)       begin dx_d<='0; dy_d<='0; end
    else if (!stall) begin dx_d<=dx; dy_d<=dy;  end

always_ff @(posedge clock)
    if (reset)       begin dx_dd<='0; dy_dd<='0; end
    else if (!stall) begin dx_dd<=dx_d; dy_dd<=dy_d; end
bilinear_interp #(.BIT_WIDTH(BIT_WIDTH_FP)) u_bilinear (
    .P_00(P_00), .P_01(P_01), .P_10(P_10), .P_11(P_11),
    .dx(dx_dd), .dy(dy_dd), .P_out(P_out)
);

assign out_x = x_ind_3;  assign out_y = y_ind_3;
assign out_valid = valid_3;  assign out_row_last = row_last_3;

endmodule