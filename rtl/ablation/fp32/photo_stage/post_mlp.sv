// post_mlp.sv  -- PIPELINED, sigmoid LUT now infers as ROM
//
// Post-MLP pixel chain for A12, matching Javad's ApproachA12.forward:
//   delta = sigmoid(z); delta = 4*delta - 2; final = clamp(center + delta, 0, 1)
//
// WHAT CHANGED AND WHY
// --------------------
// The sigmoid table used to be read combinationally from inside do_sigmoid(),
// which was called from a continuous assign. Quartus cannot map an async read
// to an M10K -- the block has a physical output register -- so it abandoned
// memory inference, discarded the $readmemh data, and swept the whole table:
//     Warning (10036): object "sig_lut" assigned a value but never read
// The module was silently doing nothing in hardware.
//
// The read is now clocked, so it infers as ROM and the .hex data actually
// loads. Costs ONE cycle of latency; throughput is unchanged (this is a
// pipeline, not a loop -- a new pixel still enters every cycle).
//
// TWO RULES THAT MUST NOT BE BROKEN HERE
// --------------------------------------
// 1. sig_r has NO RESET. A reset on a ROM output register defeats M10K
//    inference and sends you straight back to logic. Validity is tracked
//    outside this module by photo_stage_int's p_valid, not by resetting data.
// 2. Each channel gets its OWN sig_lut array. Three concurrent reads of one
//    array would need Quartus to replicate it automatically, which it may
//    refuse to do. Declaring the array inside the generate block guarantees
//    three single-port ROMs -> clean inference. Cost ~13 M10K each.
//
// Note LUT_FILE is deliberately untyped. `parameter string` makes Quartus
// throw "LUT_FILE has an aggregate value" at $readmemh.

import fp32_pkg::*;

module post_mlp #(
    parameter int BW      = 32,
    parameter int LUT_N   = 4096,
    parameter     LUT_FILE = "data_files/sigmoid_lut.hex"
) (
    input  logic clock,
    input  logic en,                            // advance the pipeline register

    input  logic [BW-1:0] z          [2:0],
    input  logic [BW-1:0] center_rgb [2:0],
    output logic [BW-1:0] final_rgb  [2:0]      // valid 1 cycle after en
);

    localparam int IDXW = $clog2(LUT_N);

    localparam logic [31:0] FP_ZERO     = 32'h00000000;  //  0.0
    localparam logic [31:0] FP_ONE      = 32'h3F800000;  //  1.0
    localparam logic [31:0] FP_FOUR     = 32'h40800000;  //  4.0
    localparam logic [31:0] FP_NEG_TWO  = 32'hC0000000;  // -2.0
    localparam logic [31:0] FP_EIGHT    = 32'h41000000;  //  8.0
    localparam logic [31:0] FP_NEG_EIGHT= 32'hC1000000;  // -8.0
    localparam logic [31:0] FP_INV_STEP = 32'h437FF000;  // 255.9375 = (LUT_N-1)/16
    localparam logic [31:0] FP_HALF     = 32'h3F000000;  //  0.5 (round-to-nearest)

    // ---- FP32 -> int, floor, pure bits ----
    function automatic int fp32_to_int_floor(input logic [31:0] x);
        logic [7:0]  exp;
        logic [23:0] mant;
        int          e;
        int unsigned val;
        exp  = x[30:23];
        mant = {1'b1, x[22:0]};
        if (exp == 0) return 0;
        e = int'(exp) - 127;
        if (e < 0) return 0;
        if (e >= 23) val = mant << (e - 23);
        else         val = mant >> (23 - e);
        return int'(val);
    endfunction

    // ---- FP32 sign/magnitude compare: 1 if a <= b as reals ----
    function automatic logic fp32_le(input logic [31:0] a, input logic [31:0] b);
        logic [31:0] ao, bo;
        ao = a[31] ? (~a) : (a | 32'h8000_0000);
        bo = b[31] ? (~b) : (b | 32'h8000_0000);
        return (ao <= bo);
    endfunction

    genvar k;
    generate
        for (k = 0; k < 3; k++) begin : gen_ch

            // ---- per-channel ROM: one read port each ----
            logic [BW-1:0] sig_lut [LUT_N-1:0];
            initial $readmemh(LUT_FILE, sig_lut);

            // ================= STAGE 1 : index + saturation (comb) =========
            logic [31:0]     shifted, scaled, rounded;
            int              idx_i;
            logic [IDXW-1:0] idx_c;
            logic            sat_lo_c, sat_hi_c;

            always_comb begin
                // saturation decided up front; the LUT is only valid in [-8,8]
                sat_lo_c = fp32_le(z[k], FP_NEG_EIGHT);
                sat_hi_c = fp32_le(FP_EIGHT, z[k]);

                // idx = floor( (z + 8) * INV_STEP + 0.5 )
                shifted  = fp32_add(z[k], FP_EIGHT);
                scaled   = fp32_mul(shifted, FP_INV_STEP);
                rounded  = fp32_add(scaled, FP_HALF);
                idx_i    = fp32_to_int_floor(rounded);

                // clamp into a NARROW index. Handing Quartus a 32-bit int as a
                // memory address is a good way to lose ROM inference.
                if (idx_i < 0)            idx_c = '0;
                else if (idx_i > LUT_N-1) idx_c = IDXW'(LUT_N-1);
                else                      idx_c = IDXW'(idx_i);
            end

            // ================= PIPELINE REGISTER ===========================
            // NO RESET here -- see rule 1 in the header.
            logic [BW-1:0] sig_r, center_r;
            logic          sat_lo_r, sat_hi_r;

            always_ff @(posedge clock) begin
                if (en) begin
                    sig_r    <= sig_lut[idx_c];      // <-- the ROM read
                    sat_lo_r <= sat_lo_c;
                    sat_hi_r <= sat_hi_c;
                    center_r <= center_rgb[k];       // must travel with it
                end
            end

            // ================= STAGE 2 : 4s-2, add centre, clamp (comb) ====
            logic [BW-1:0] s, delta, summed;

            assign s      = sat_lo_r ? FP_ZERO
                          : sat_hi_r ? FP_ONE
                          : sig_r;
            assign delta  = fp32_add(fp32_mul(FP_FOUR, s), FP_NEG_TWO); // 4s - 2
            assign summed = fp32_add(center_r, delta);                  // centre + delta

            assign final_rgb[k] = fp32_le(summed, FP_ZERO) ? FP_ZERO
                                : fp32_le(FP_ONE, summed)  ? FP_ONE
                                : summed;
        end
    endgenerate

endmodule