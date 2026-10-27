`default_nettype none

module geo_synth # (
    parameter BIT_WIDTH_FP = 32,
    parameter BIT_WIDTH_UINT = 8,
    parameter IND_WIDTH = 8,
    parameter IMG_WIDTH = 256,
    parameter IMG_HEIGHT = 256

) (
    input logic clock,
    input logic reset,

    output logic out_valid,
    output logic [IND_WIDTH - 1:0] out_x, out_y,
    output logic [3*BIT_WIDTH_FP - 1:0] P_out_flat,
    output logic frame_done,
    output logic out_row_last


);  

    logic [IND_WIDTH - 1:0] x_c, y_c;
    logic [BIT_WIDTH_FP - 1:0] p_out [2:0];
    logic fd, orl, ov;
    

    top_geo u_geo (.clock(clock), .reset(reset), .stall(1'b0),
                    .out_valid(ov), .out_x(x_c), .out_y(y_c), .P_out(p_out),
                    .frame_done(fd), .out_row_last(orl));


    always_ff @ (posedge clock)
        begin
            out_valid <= ov;
            out_x <= x_c;
            out_y <= y_c;
            P_out_flat <= {p_out[2], p_out[1], p_out[0]};
            frame_done <= fd;
            out_row_last <= orl;
        end

endmodule

`default_nettype wire