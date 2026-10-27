// top_geo_int.sv -- geometric stage, integer datapath
//
// Datapath is now:  coord_gen -> LUT_fxp (Q8.8) -> csf_int -> ffn_int
//                   -> input_buffer_parity_int (uint8) -> bilinear_interp_int (uint8)
//
// Outputs BOTH:
//   P_out_int [2:0]  : the raw uint8 result out of bilinear_interp_int
//   P_out     [2:0]  : the same pixel dequantised to FP32, so inter_line_buffer /
//                      patch_extractor / photo_stage_int keep seeing FP32 and need
//                      zero changes.
//
// LATENCY CHANGED: 2 -> 6 cycles.  See LAT below.

module top_geo_int #(
    parameter int BIT_WIDTH_FP   = 32,   // fp32 output width
    parameter int BIT_WIDTH_UINT = 8,    // pixel width in the int datapath
    parameter int BIT_WIDTH_D    = 8,    // fractional bits of the Q8.8 coord
    parameter int Q_WIDTH        = 16,   // Q8.8 coord width  (INT + D)
    parameter int IND_WIDTH      = 8,    // integer coord width
    parameter int IMG_WIDTH      = 256,
    parameter int IMG_HEIGHT     = 256,

    parameter LUT_FILE    = "data_files/lut_quant.hex",
   // parameter DEQUANT_LUT = "data_files/dequant_lut.hex",
    parameter IMAGE_FILE  = "data_files/image.hex",   // unused, kept for port-compat

    parameter IMAGE_EE = "data_files/rom_ee.hex",
    parameter IMAGE_EO = "data_files/rom_eo.hex",
    parameter IMAGE_OE = "data_files/rom_oe.hex",
    parameter IMAGE_OO = "data_files/rom_oo.hex"
) (
    input  logic clock,
    input  logic reset,
    input  logic stall,

    output logic                      out_valid,
    output logic [IND_WIDTH-1:0]      out_x, out_y,

    output logic [BIT_WIDTH_UINT-1:0]   P_out     [2:0],   

    output logic                      frame_done,
    output logic                      out_row_last
);

    // ------------------------------------------------------------------
    // Pipeline latency budget, in registers, from x_ind to P_out_int
    //   LUT_fxp                 : 1  (read_word)
    //   csf_int                 : 0  (pure slice)
    //   ffn_int                 : 0  (pure comb)
    //   input_buffer_parity_int : 2  (word_* then ch_*)   <-- was 1
    //   bilinear_interp_int     : 3  (weights, acc, round) <-- was 0
    // ------------------------------------------------------------------
    localparam int LAT    = 7;
    localparam int IB_LAT = 2;   // dx/dy must be delayed this much to meet the pixels

    // ---------------- coord generator ----------------
    logic [IND_WIDTH-1:0] x_ind, y_ind;
    logic                 row_last_0, frame_done_0;

    coord_gen #(.IMG_HEIGHT(IMG_HEIGHT), .IMG_WIDTH(IMG_WIDTH), .IND_WIDTH(IND_WIDTH))
    u_coord_gen (
        .clock(clock), .reset(reset), .stall(stall),
        .x_out(x_ind), .y_out(y_ind),
        .frame_done(frame_done_0),
        .row_last(row_last_0)
    );

    // ---------------- side-channel delay line (LAT deep) ----------------
    logic                 valid_d   [1:LAT];
    logic [IND_WIDTH-1:0] x_d       [1:LAT];
    logic [IND_WIDTH-1:0] y_d       [1:LAT];
    logic                 rowlast_d [1:LAT];
    logic                 fdone_d   [1:LAT];

    always_ff @(posedge clock) begin
        if (reset) begin
            for (int i = 1; i <= LAT; i++) begin
                valid_d[i]   <= 1'b0;
                x_d[i]       <= '0;
                y_d[i]       <= '0;
                rowlast_d[i] <= 1'b0;
                fdone_d[i]   <= 1'b0;
            end
        end else if (!stall) begin
            valid_d[1]   <= 1'b1;          // pipe is fed every non-stalled cycle
            x_d[1]       <= x_ind;
            y_d[1]       <= y_ind;
            rowlast_d[1] <= row_last_0;
            fdone_d[1]   <= frame_done_0;
            for (int i = 2; i <= LAT; i++) begin
                valid_d[i]   <= valid_d[i-1];
                x_d[i]       <= x_d[i-1];
                y_d[i]       <= y_d[i-1];
                rowlast_d[i] <= rowlast_d[i-1];
                fdone_d[i]   <= fdone_d[i-1];
            end
        end
    end

    // ---------------- geometry LUT: Q8.8 out ----------------
    logic [Q_WIDTH-1:0] x_q88, y_q88;

    LUT_fxp #(
        .BIT_WIDTH(BIT_WIDTH_FP), .IND_WIDTH(IND_WIDTH),
        .Q_WIDTH(Q_WIDTH), .LUT_FILE(LUT_FILE)
    ) u_geo_lut (
        .x_ind(x_ind), .y_ind(y_ind),
        .clock(clock), .reset(reset), .stall(stall),
        .x_out(x_q88), .y_out(y_q88)
    );

    // ---------------- split Q8.8 into int part + fraction ----------------
    logic [IND_WIDTH-1:0]   x_int, y_int;
    logic [BIT_WIDTH_D-1:0] dx, dy;

    csf_int #(
        .Q_WIDTH(Q_WIDTH), .BIT_WIDTH_INT(IND_WIDTH), .BIT_WIDTH_D(BIT_WIDTH_D)
    ) u_coord_split (
        .x_in(x_q88), .y_in(y_q88),
        .x_int(x_int), .y_int(y_int),
        .dx(dx), .dy(dy)
    );

    // ---------------- four neighbour addresses, packed {y,x} ----------------
    logic [2*IND_WIDTH-1:0] A_00, A_01, A_10, A_11;

    ffn_int #(
        .IND_WIDTH(IND_WIDTH), .IMG_WIDTH(IMG_WIDTH), .IMG_HEIGHT(IMG_HEIGHT)
    ) u_fetch_four (
        .x_int(x_int), .y_int(y_int),
        .A_00(A_00), .A_01(A_01), .A_10(A_10), .A_11(A_11)
    );

    // ---------------- pixel fetch, uint8 out ----------------
    logic [BIT_WIDTH_UINT-1:0] P_00 [2:0], P_01 [2:0], P_10 [2:0], P_11 [2:0];

    // NOTE: named params only. The old positional #(BIT_WIDTH_FP, ...) form no
    // longer lines up, input_buffer_parity_int dropped DEQUANT_LUT and takes
    // IND_WIDTH where BIT_WIDTH_FP used to sit.
    input_buffer_parity_int #(
        .BIT_WIDTH_UINT(BIT_WIDTH_UINT),
        .IND_WIDTH(IND_WIDTH),
        .IMG_WIDTH(IMG_WIDTH), .IMG_HEIGHT(IMG_HEIGHT),
        .IMAGE_EE(IMAGE_EE), .IMAGE_EO(IMAGE_EO),
        .IMAGE_OE(IMAGE_OE), .IMAGE_OO(IMAGE_OO)
    ) u_input_buffer (
        .clock(clock), .reset(reset), .stall(stall),
        .addr_00(A_00), .addr_01(A_01), .addr_10(A_10), .addr_11(A_11),
        .P_00(P_00), .P_01(P_01), .P_10(P_10), .P_11(P_11)
    );

    // ---------------- dx/dy delayed to meet the pixels at bilinear ----------------
    // bilinear_interp_int registers dx/dy and P_xx on the SAME edge, so dx must be
    // aligned to the input buffer's IB_LAT, not IB_LAT-1.
    logic [BIT_WIDTH_D-1:0] dx_d [1:IB_LAT];
    logic [BIT_WIDTH_D-1:0] dy_d [1:IB_LAT];

    always_ff @(posedge clock) begin
        if (reset) begin
            for (int i = 1; i <= IB_LAT; i++) begin dx_d[i] <= '0; dy_d[i] <= '0; end
        end else if (!stall) begin
            dx_d[1] <= dx;  dy_d[1] <= dy;
            for (int i = 2; i <= IB_LAT; i++) begin
                dx_d[i] <= dx_d[i-1];
                dy_d[i] <= dy_d[i-1];
            end
        end
    end

    // ---------------- bilinear, integer ----------------
    bilinear_interp_int #(
        .BIT_WIDTH_FP(BIT_WIDTH_FP),
        .BIT_WIDTH_INT(BIT_WIDTH_UINT),
        .BIT_WIDTH_D(BIT_WIDTH_D)
    ) u_bilinear (
        .clock(clock), .reset(reset), .stall(stall),
        .P_00(P_00), .P_01(P_01), .P_10(P_10), .P_11(P_11),
        .dx(dx_d[IB_LAT]), .dy(dy_d[IB_LAT]),
        .P_out(P_out)
    );

    // ---------------- uint8 -> FP32 for the inter-line buffer ----------------
    // Same 256-entry table the old input_buffer_parity used, so the numbers the
    // photo stage sees are bit-identical to the fp32 build. Combinational: adds
    // no latency, and it is 3 lookups instead of the 12 the old buffer did.
    /*
    logic [BIT_WIDTH_FP-1:0] dequant_LUT [0:(1<<BIT_WIDTH_UINT)-1];
    initial $readmemh(DEQUANT_LUT, dequant_LUT);

    always_comb
        for (int c = 0; c < 3; c++)
            P_out[c] = dequant_LUT[P_out_int[c]];
*/

    // ---------------- delay-matched sidebands ----------------
    assign out_x        = x_d[LAT];
    assign out_y        = y_d[LAT];
    assign out_valid    = valid_d[LAT];
    assign out_row_last = rowlast_d[LAT];
    assign frame_done   = fdone_d[LAT];

endmodule