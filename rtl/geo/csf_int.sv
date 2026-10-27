// csf_int.sv -- coordinate split, PURE BIT SLICE
//
// Was fp32_floor + fp32_sub. The input is Q8.8, so the integer part is the
// top BIT_WIDTH_INT bits and the fraction is the bottom BIT_WIDTH_D bits.
// The old version built a float and tore it apart to recover fields that
// were already sitting in the raw word.
//
// No import needed -- there is no arithmetic left in this module.

module csf_int #(
    parameter int Q_WIDTH       = 16,   // Q8.8 coordinate
    parameter int BIT_WIDTH_INT = 8,    // integer bits
    parameter int BIT_WIDTH_D   = 8     // fractional bits
) (
    input  logic [Q_WIDTH-1:0]       x_in,
    input  logic [Q_WIDTH-1:0]       y_in,

    output logic [BIT_WIDTH_D-1:0]   dx,
    output logic [BIT_WIDTH_D-1:0]   dy,

    output logic [BIT_WIDTH_INT-1:0] x_int,
    output logic [BIT_WIDTH_INT-1:0] y_int
);

    // If these don't add up the slices silently misalign and every
    // coordinate is off by a power of two. Catch it at elaboration.
    initial begin
        if (BIT_WIDTH_INT + BIT_WIDTH_D != Q_WIDTH)
            $fatal(1, "csf_int: BIT_WIDTH_INT(%0d) + BIT_WIDTH_D(%0d) != Q_WIDTH(%0d)",
                   BIT_WIDTH_INT, BIT_WIDTH_D, Q_WIDTH);
    end

    assign x_int = x_in[Q_WIDTH-1 -: BIT_WIDTH_INT];   // [15:8]
    assign y_int = y_in[Q_WIDTH-1 -: BIT_WIDTH_INT];

    assign dx    = x_in[BIT_WIDTH_D-1:0];              // [7:0]
    assign dy    = y_in[BIT_WIDTH_D-1:0];

endmodule