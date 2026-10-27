// synth/nn_pipe/nn_pipe_synth_top.sv  — synthesis wrapper, not for sim
module nn_pipe_synth_top (
    input  logic clock,
    input  logic reset,
    input  logic in_valid,
    output logic in_ready,
    input  logic [29*32-1:0] f_flat,
    output logic out_valid,
    input  logic out_ready,
    output logic [3*32-1:0]  z_flat
);
    logic [31:0] f [28:0];
    logic [31:0] z [2:0];

    genvar i;
    generate
        for (i = 0; i < 29; i++) begin : g_in
            assign f[i] = f_flat[i*32 +: 32];
        end
        for (i = 0; i < 3; i++) begin : g_out
            assign z_flat[i*32 +: 32] = z[i];
        end
    endgenerate

    nn_pipelined_quant #(
        .SCALES ("../../data_files/a12_wscales.hex"),
        .L1_W_Q ("../../data_files/a12_layer1_w_q.hex"),
        .L1_B   ("../../data_files/a12_layer1_b_q.hex"),
        .L2_W_Q ("../../data_files/a12_layer2_w_q.hex"),
        .L2_B   ("../../data_files/a12_layer2_b_q.hex"),
        .L3_W_Q ("../../data_files/a12_layer3_w_q.hex"),
        .L3_B   ("../../data_files/a12_layer3_b_q.hex"),
        .RS_W_Q ("../../data_files/a12_residual_w_q.hex"),
        .RS_B   ("../../data_files/a12_residual_b_q.hex")
    ) u_nn (
        .clock, .reset,
        .in_valid, .in_ready, .f,
        .out_valid, .out_ready, .z
    );

endmodule