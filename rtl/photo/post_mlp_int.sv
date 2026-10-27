// post_mlp_int.sv  --  ALL-INTEGER post-MLP pixel chain.  NO fp32 ANYWHERE.
//
// Replaces post_mlp.sv. Same function, same 1-cycle latency, same `en`
// contract -- so photo_stage_int_v2's accept/p_valid handshake is UNCHANGED.
//
// ===========================================================================
// WHAT THE FP32 VERSION COMPUTED
//
//   delta = 4*sigmoid(z) - 2                     z = LAMBDA3 * zi  (real)
//   final = clamp(center + delta, 0, 1)          center, final in [0,1]
//
// and it burned, per channel, per pixel:
//   fp32_add, fp32_mul, fp32_add   (index)      2x fp32_le  (saturate)
//   fp32_mul, fp32_add, fp32_add   (combine)    2x fp32_le  (clamp)
//   fp32_to_int_floor              (index)
// all COMBINATIONAL and all in series behind the ROM read. That is the exact
// "round-1 problem" the nn_pipelined_int header warned about: an fp32_add is
// four sequential barrel shifters (align, add, LZD, normalise) and measured
// 49.4 ns on its own. Six of them back to back would have owned the build.
//
// ===========================================================================
// WHAT THIS VERSION COMPUTES
//
//   final_rgb = clamp( center_u8 + dlut[z_idx], 0, 255 )
//
// One ROM read, one add, one saturate. That is the entire module. Two
// observations get it there.
//
// 1. THE INDEX IS AFFINE IN zi, SO FOLD LAMBDA3 INTO IT.
//      idx = floor((z + 8)*INV_STEP + 0.5)          INV_STEP = (LUT_N-1)/16
//          = floor(zi * (LAMBDA3*INV_STEP)) + LUT_N/2
//    LAMBDA3*INV_STEP is a compile-time real constant, so this is ONE integer
//    constant multiply and an add. It now lives in nn_pipelined_int, which
//    already had three free registered stages there and no longer needs
//    sint_to_fp32 or fp32_mul at all.  ==> this module receives `z_idx`, a
//    plain 12-bit LUT address, and does no index arithmetic whatsoever.
//
// 2. THE TABLE STORES delta IN 8-BIT PIXEL LSBs, ALREADY ROUNDED.
//    The old table held sigmoid(z) as fp32 and then spent an fp32_mul and an
//    fp32_add turning it into 4s-2 at RUNTIME, every pixel, three times over.
//    That is constant folding done in hardware. Bake it into the .hex:
//
//        dlut[i] = round( 255 * (4*sigmoid(z_i) - 2) )     range [-510, +510]
//
//    The endpoints are FORCED to +/-2 (see the generator), which reproduces
//    the sat_lo/sat_hi branches exactly: z <= -8 clamps idx to 0, z >= 8
//    clamps it to LUT_N-1, and those two entries hold the saturated values.
//    So the two fp32_le compares are deleted, not approximated.
//
// ===========================================================================
// WHY THE TABLE HAS NO FRACTIONAL BITS -- THIS IS NOT A SHORTCUT
//
// The obvious "safe" choice is to store delta with a few fractional bits and
// round at the output: dlut[i] = round(255*delta*2^DF), then
// (center<<DF) + dlut + 2^(DF-1) >>> DF. It looks strictly more accurate.
// It is strictly LESS accurate, and it costs extra logic to be so.
//
// center_u8 is an INTEGER, so
//     rint(center + 255*delta)  ==  center + rint(255*delta)      exactly.
// Rounding delta by itself at 1-LSB resolution is mathematically identical to
// rounding once at the very end. Fractional bits do not refine that -- they
// insert a SECOND rounding step (table quantisation) whose result can land
// exactly on a .5 boundary that the single-rounding reference falls just short
// of, and then the output rounding tips it the other way.
//
// Measured over the module's ENTIRE input space, 4096 indices x 256 centres:
//
//     DF=0    table 10b    0 mismatches / 1048576      <-- exact
//     DF=4    table 14b    3220
//     DF=6    table 16b    1013
//     DF=8    table 18b     466
//     DF=10   table 20b      81
//
// Every fractional bit is a regression. DF=0 also deletes the shifter, the
// rounding constant and 6 bits of adder width, and costs the SAME 8 M10K
// (4096 deep is 8 blocks at any width from 11 to 20 in 512x20 mode). It is
// smaller, faster and exact. Do not "improve" this by adding precision.
//
// The old fp32 table was 4096x32 = 16 M10K per channel, so this also halves
// the memory: 48 -> 24 M10K.
//
// ===========================================================================
// TWO RULES CARRIED OVER FROM post_mlp.sv -- STILL LOAD-BEARING
//
// 1. d_r / c_r have NO RESET. A reset on a ROM output register defeats M10K
//    inference and sends the table back into ALMs (or gets it swept, which is
//    what "assigned a value but never read" meant last time). Validity is
//    tracked by photo_stage_int_v2's p_valid, not by resetting data.
// 2. Each channel declares its OWN dlut array INSIDE the generate block.
//    Three concurrent reads of one array would need Quartus to replicate it
//    automatically, which it may simply refuse to do. Three single-port ROMs
//    is the only shape that infers reliably.
//
// DLUT_FILE is deliberately untyped: `parameter string` makes Quartus throw
// "has an aggregate value" at $readmemh.
//
// ===========================================================================
// NO CASTS IN ANY EXPRESSION IN THE GENERATE BLOCK. Same Verific front-end
// bug documented in nn_pipelined_int.sv -- a size/sign cast nested inside a
// binary operator inside a generate-for crashes quartus_map. Everything below
// is a plain assignment between pre-declared signed variables, plus one bit-
// field write to widen the unsigned centre.

module post_mlp_int #(
    parameter int PIX_W  = 8,      // uint8 pixels in and out
    parameter int IDXW   = 12,     // LUT address width, from nn_pipelined_int
    parameter int DLUT_W = 12,     // signed table width; delta is [-510, +510]
    parameter     DLUT_FILE = "data_files/delta_lut.hex"
) (
    input  logic clock,
    input  logic en,                                 // advance the pipeline reg

    input  logic [IDXW-1:0]  z_idx     [2:0],        // from nn_pipelined_int
    input  logic [PIX_W-1:0] center_u8 [2:0],        // native uint8 centre
    output logic [PIX_W-1:0] final_rgb [2:0]         // valid 1 cycle after en
);

    localparam int LUT_N = 1 << IDXW;
    localparam int PMAX  = (1 << PIX_W) - 1;         // 255

    // centre 0..255 plus delta -510..+510 spans [-510, 765]: 11 bits signed.
    // DLUT_W+2 = 14 leaves headroom and costs nothing.
    localparam int SUMW = DLUT_W + 2;

    localparam logic signed [SUMW-1:0]  S_ZERO = 0;
    localparam logic signed [SUMW-1:0]  S_PMAX = PMAX;
    localparam logic        [PIX_W-1:0] PMAX_U = PMAX;   // typed: no int part-select

    genvar k;
    generate
        for (k = 0; k < 3; k++) begin : gen_ch

            // ---- per-channel ROM: one read port each ----
            logic signed [DLUT_W-1:0] dlut [LUT_N-1:0];
            initial $readmemh(DLUT_FILE, dlut);

            // ================= PIPELINE REGISTER =======================
            // NO RESET -- see rule 1 in the header. The centre must travel
            // with the ROM read or it arrives a cycle early.
            logic signed [DLUT_W-1:0] d_r;
            logic        [PIX_W-1:0]  c_r;

            always_ff @(posedge clock) begin
                if (en) begin
                    d_r <= dlut[z_idx[k]];       // <-- the ROM read
                    c_r <= center_u8[k];
                end
            end

            // ================= STAGE 2 : add, saturate (comb) ===========
            logic signed [SUMW-1:0] c_w;    // centre, zero-extended
            logic signed [SUMW-1:0] d_w;    // delta, sign-extended
            logic signed [SUMW-1:0] sum;

            always_comb begin
                // centre is UNSIGNED, so zero-extend by bit-field placement
                // rather than by assignment (which would sign-extend once the
                // target is signed). Pure wiring, no logic.
                c_w              = '0;
                c_w[PIX_W-1 : 0] = c_r;

                d_w = d_r;                  // signed -> wider signed, auto-extends
                sum = c_w + d_w;
            end

            // clamp(.,0,1) in [0,1] becomes saturate to [0,255]
            always_comb begin
                if      (sum <= S_ZERO) final_rgb[k] = '0;
                else if (sum >= S_PMAX) final_rgb[k] = PMAX_U;
                else                    final_rgb[k] = sum[PIX_W-1:0];
            end

        end
    endgenerate

endmodule