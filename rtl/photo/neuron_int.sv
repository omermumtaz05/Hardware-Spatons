// neuron_int.sv  -- INTEGER DATAPATH
//
// The fp32 dequant chain (sint_to_fp32 -> fp32_mul -> fp32_add) is GONE.
// That chain was ~1450 ALUTs x 102 instances ~= 148k ALUTs, which is the
// entire reason this design did not fit.
//
// The neuron now emits the raw integer accumulator. Everything downstream
// (requantize, x M, + B_int, ReLU) is layer-level, because the requantize
// needs max|acc| across ALL neurons in the layer -- a single neuron cannot
// compute it.
//
// What remains here is a thin wrapper around dot_prod_int. It is kept as a
// separate module only to preserve the existing instance hierarchy and the
// done_bus handshake in layer_quant. If you would rather flatten it, delete
// this file and instantiate dot_prod_int directly in the generate block.

module neuron_int #(
    parameter int BW_ACC     = 28,
    parameter int BW_W       = 10,
    parameter int BW_ACT     = 12,
    parameter int INPUT_SIZE = 29,
    parameter int NUM_FOLDS  = 4
) (
    input  logic clock,
    input  logic reset,

    input  logic start,

    // Pre-muxed activations for the current cycle, shared across the whole
    // layer -- see the header of dot_prod_int.sv for why this is no longer
    // the full vector.
    input  logic signed [BW_ACT-1:0] A_SEL [NUM_FOLDS],
    input  logic signed [BW_W-1:0]   W [0:INPUT_SIZE-1],

    output logic done,
    output logic signed [BW_ACC-1:0] acc_out
);

    logic signed [BW_ACC-1:0] dot_out;
    logic                     dot_done;

    dot_prod_int #(
        .INPUT_SIZE (INPUT_SIZE),
        .NUM_FOLDS  (NUM_FOLDS),
        .BW_W       (BW_W),
        .BW_ACT     (BW_ACT),
        .BW_ACC     (BW_ACC)
    ) dp_f (
        .clock    (clock),
        .reset    (reset),
        .start    (start),
        .ACT_SEL  (A_SEL),
        .W_Q      (W),
        .dot_prod (dot_out),
        .done     (dot_done)
    );

    always_ff @(posedge clock) begin
        if (reset) begin
            acc_out <= '0;
            done    <= 1'b0;
        end else begin
            done <= dot_done;
            if (dot_done)
                acc_out <= dot_out;
        end
    end

endmodule