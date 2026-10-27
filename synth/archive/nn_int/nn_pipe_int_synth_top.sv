// synth/nn_pipe_int/nn_pipe_int_synth_top.sv  — synthesis only, not for sim
module nn_pipe_int_synth_top (
    input  logic clock,
    input  logic reset,
    input  logic in_valid,
    output logic in_ready,
    input  logic [29*32-1:0] f_flat,
    output logic out_valid,
    input  logic out_ready,
    output logic [3*32-1:0]  z_flat
);
    // ---- input registers ----
    logic [29*32-1:0] f_flat_r;
    logic             in_valid_r, out_ready_r;
    always_ff @(posedge clock) begin
        f_flat_r    <= f_flat;
        in_valid_r  <= in_valid;
        out_ready_r <= out_ready;
    end

    logic [31:0] f [28:0];
    logic [31:0] z [2:0];

    genvar i;
    generate
        for (i = 0; i < 29; i++) begin : g_in
            assign f[i] = f_flat_r[i*32 +: 32];
        end
    endgenerate

    logic in_ready_c, out_valid_c;

    nn_pipelined_int u_nn (
        .clock, .reset,
        .in_valid  (in_valid_r),
        .in_ready  (in_ready_c),
        .f         (f),
        .out_valid (out_valid_c),
        .out_ready (out_ready_r),
        .z         (z)
    );

    // ---- output registers ----
    always_ff @(posedge clock) begin
        for (int j = 0; j < 3; j++)
            z_flat[j*32 +: 32] <= z[j];
        in_ready  <= in_ready_c;
        out_valid <= out_valid_c;
    end
endmodule