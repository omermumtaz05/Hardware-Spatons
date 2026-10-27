package fp32_pkg;

    parameter BIT_WIDTH = 32;
    parameter FP_EXP_WIDTH = 8;
    parameter FP_MANT_WIDTH = 23;

    localparam exp_bias = 127;

   // ---- FP32 floor for a NON-NEGATIVE input ----
    // value = 1.frac * 2^(exp-127); e = exp-127 tells us where the binary
    // point sits in the significand. We keep the top e mantissa bits and
    // zero the low (23-e) bits.
    function automatic logic [BIT_WIDTH - 1:0] fp32_floor(input logic [BIT_WIDTH - 1:0] f);
        logic [FP_EXP_WIDTH - 1:0]  exp;
        logic [FP_MANT_WIDTH - 1:0] mant;
        int          e;            // unbiased exponent
        logic [FP_MANT_WIDTH - 1:0] mask;         // 1s in the integer-mantissa positions
        
        begin
            mant = f[FP_MANT_WIDTH - 1:0]; // extract mantissa bits from number
            exp = f[FP_MANT_WIDTH + FP_EXP_WIDTH - 1:FP_MANT_WIDTH]; // extract exponent bits from fp32 number

            e = int'(exp) - exp_bias; // extract unbiased exp
            begin
                if(e < 0)
                    fp32_floor = '0; // value less than 1 therefore floor is 0
                else if(e >= FP_MANT_WIDTH)
                    fp32_floor = f; // value already integer, floor is itself
                else  
                    begin
                        mask = ~((1'b1 << (FP_MANT_WIDTH - e)) - 1); // extract integer bits of mantissa by masking off frac bits
                        fp32_floor = {1'b0, exp, mask & mant}; // put together full fp32 number (know sign always 0)
                    end
            end


        end
    endfunction




    function automatic logic [BIT_WIDTH - 1:0] fp32_add(input logic [BIT_WIDTH - 1:0] a,
                                             input logic [BIT_WIDTH - 1:0] b);

        logic [FP_MANT_WIDTH - 1:0] ma, mb;
        logic [FP_MANT_WIDTH:0] diff_mant;
        logic [FP_MANT_WIDTH + 1:0] sum_mant;


        logic [FP_EXP_WIDTH - 1:0] ea, eb;
        logic [FP_EXP_WIDTH - 1:0] diff_exp;
        logic [FP_EXP_WIDTH - 1:0] e_res;

        logic sign_a, sign_b;
        logic sign_res;

        logic a_bigger;


        begin
            // extract mantissas of a and b
            ma = a[FP_MANT_WIDTH - 1:0];
            mb = b[FP_MANT_WIDTH - 1:0];

            // extract exponents of a and b
            ea = a [FP_EXP_WIDTH + FP_MANT_WIDTH - 1:FP_MANT_WIDTH];
            eb = b [FP_EXP_WIDTH + FP_MANT_WIDTH - 1:FP_MANT_WIDTH];
            
            // extract signs of a and b
            sign_a = a[BIT_WIDTH - 1];
            sign_b = b[BIT_WIDTH - 1];
            
            a_bigger = (ea > eb) || (ea == eb && ma >= mb);


            if (a[BIT_WIDTH-2:0] == 0)      // a is zero (ignore sign)
                fp32_add = b;
            else if (b[BIT_WIDTH-2:0] == 0) // b is zero
                fp32_add = a;
            else if(sign_a == sign_b) // signs are same -> add
                begin
                    if(a_bigger) // a is larger than b
                        begin
                            diff_exp = ea - eb; //exp of a is larger so take difference with ea - eb
                            sum_mant = {1'b1, ma} + ({1'b1, mb} >> diff_exp);
                            sign_res = sign_a; // sign of result is sign of a (larger term)
                            e_res = ea; // exp of result is exp of a
                        end
                    else  //b larger than a
                        begin
                            diff_exp = eb - ea; // same rationale as above
                            sign_res = sign_b;
                            sum_mant = {1'b1, mb} + ({1'b1, ma} >> diff_exp);
                            e_res = eb;
                        end
                    if(sum_mant[FP_MANT_WIDTH + 1]) // if mantissa looks like 11.xxx, overflow!
                        begin
                            sum_mant = sum_mant >> 1; // shift mantissa down one bit to have mantissa in proper 1.x format
                            e_res = e_res + 1; // inc exp
                        end

                    fp32_add = {sign_res, e_res, sum_mant[FP_MANT_WIDTH - 1:0]}; // put together final fp32 result


                        
                end

            else // subtraction as signs differ and its the only remaining possible option, no need for condition!
                begin

                    if(a_bigger) // a is bigger than b
                    begin
                        diff_exp = ea - eb; // exp of a is larger
                        diff_mant = {1'b1, ma} - ({1'b1, mb} >> diff_exp); // subtract the mantissa of a then b
                        sign_res = sign_a; // sign of result is sign of larger number
                        e_res = ea;
                    end
                    else
                    begin //b is larger, same rationale as above
                        diff_exp = eb - ea;
                        diff_mant = {1'b1, mb} - ({1'b1, ma} >> diff_exp);
                        sign_res = sign_b;
                        e_res = eb;

                    end

                    if(diff_mant == 0) // cancel each other out
                        begin
                            sign_res = '0; 
                            e_res = '0;
                        end
                    else

                    begin
                        // shift mantissa diff left until leading 1 is in implicit position and adjust exp accordingly as well
                        for (int s = 0; s < FP_MANT_WIDTH + 1; s++) begin
                            if (diff_mant[FP_MANT_WIDTH] == 0 && e_res > 0) begin
                                diff_mant = diff_mant << 1;
                                e_res = e_res - 1;
                            end
                        end
                        if(diff_mant[FP_MANT_WIDTH] == 0) // if we shifted out all the bits, result is zero
                            begin
                                sign_res = '0;
                                e_res = '0;
                            end

                        end
                    
                        fp32_add = {sign_res, e_res, diff_mant[FP_MANT_WIDTH - 1:0]}; // put together final fp32 result

                end

        end

    endfunction


    function automatic logic [BIT_WIDTH-1:0] fp32_sub(input logic [BIT_WIDTH-1:0] a, b);
        return fp32_add(a, {~b[BIT_WIDTH-1], b[BIT_WIDTH-2:0]}); // sub is just addition but with the second term's sign bit negative
    endfunction

    function automatic logic [BIT_WIDTH-1:0] fp32_mul(input logic [BIT_WIDTH-1:0] a, b);


        logic [FP_MANT_WIDTH - 1:0] ma, mb;
        logic [2*FP_MANT_WIDTH+1:0] prod_mant;


        logic [FP_EXP_WIDTH - 1:0] ea, eb;
        logic [FP_EXP_WIDTH + 1:0] e_res;

        logic sign_a, sign_b;
        logic sign_res;

        logic [FP_MANT_WIDTH:0] mant_rnd;
        logic round_bit, sticky;

        if(a[BIT_WIDTH - 2:0] == '0 || b[BIT_WIDTH - 2:0] == '0) // either one is 0, result is 0
            fp32_mul = '0;
        else
            begin   
                
                //extract mant of a and b
                ma = a[FP_MANT_WIDTH - 1:0];
                mb = b[FP_MANT_WIDTH - 1:0];

                // extract exponents of a and b
                ea = a [FP_EXP_WIDTH + FP_MANT_WIDTH - 1:FP_MANT_WIDTH];
                eb = b [FP_EXP_WIDTH + FP_MANT_WIDTH - 1:FP_MANT_WIDTH];
                
                // extract signs of a and b
                sign_a = a[BIT_WIDTH - 1];
                sign_b = b[BIT_WIDTH - 1];

                // sign of result is sign of a XOR sign b
                sign_res = sign_a ^ sign_b;

                e_res = ea + eb - exp_bias;

                
                prod_mant = {1'b1, ma} * {1'b1, mb};
                
                if(prod_mant[2*FP_MANT_WIDTH+1]) // if product has one bit overflow, shift it left one bit and inc result's exp
                    begin
                        prod_mant = prod_mant >> 1'b1;
                        e_res = e_res + 1;
                    end

                // ---- round-to-nearest-even on the truncated low bits ----

                round_bit = prod_mant[FP_MANT_WIDTH-1];              // MSB of discarded part
                sticky    = |prod_mant[FP_MANT_WIDTH-2:0];           // OR of the rest
                mant_rnd  = prod_mant[2*FP_MANT_WIDTH-1:FP_MANT_WIDTH]
                          + (round_bit && (sticky || prod_mant[FP_MANT_WIDTH]));  // RNE

                if (mant_rnd[FP_MANT_WIDTH]) begin   // rounding overflowed the mantissa
                    e_res    = e_res + 1;
                    mant_rnd = mant_rnd >> 1;
                end

                fp32_mul = {sign_res, e_res[FP_EXP_WIDTH-1:0], mant_rnd[FP_MANT_WIDTH-1:0]};
            end
    endfunction

// add to wherever fp32_mult / fp32_add live
function automatic logic [31:0] fp32_negate(input logic [31:0] x);
    fp32_negate = {~x[31], x[30:0]};   // flip sign bit only — free in hardware
endfunction

function automatic logic [31:0] fp32_recip(input logic [31:0] x);
    logic [31:0] y0, y1, y2;
    localparam logic [31:0] TWO = 32'h40000000;
    localparam logic [31:0] MAGIC = 32'h7EF311C2;
    begin
        y0 = MAGIC - x;
        y1 = fp32_mul(y0, fp32_add(TWO, fp32_negate(fp32_mul(x, y0))));
        y2 = fp32_mul(y1, fp32_add(TWO, fp32_negate(fp32_mul(x, y1))));
        fp32_recip = y2;
    end
endfunction

// ============================================================================
// fp32_recip_fixed -- 1/x for fp32, exponent handled exactly, mantissa by
// Newton-Raphson in Q2.30 fixed point.
//
// Paste this inside `package fp32_pkg;` (or `include it there). It does NOT
// call fp32_mul / fp32_add anywhere -- that is the whole point. The only
// arithmetic is integer mul/sub, which is exact, so the NR iteration has a
// clean fixed point instead of wandering around in the truncation noise of
// the fp32 helpers.
//
// Decomposition:
//   x  = 1.f * 2^(E-127),  m = 1.f in [1,2)
//   1/x = (1/m) * 2^(127-E),  and 1/m in (0.5,1]
//   f == 0 : 1/x = 1.0 * 2^(127-E)      -> biased exp = 254-E, mant = 0
//   f != 0 : 1/x = (2/m) * 2^(126-E)    -> biased exp = 253-E, mant = frac(2/m)
//            with 2/m in (1,2), i.e. already normalized.
//
// Only the mantissa needs arithmetic, and it is ALWAYS in [1,2), so one fixed
// seed covers every input -- no seed-vs-distribution fragility.
//
// Seed: y0 = 24/17 - (8/17)*m   (minimax linear on [1,2], |rel err| <= 1/17)
// Iter: y <- y*(2 - m*y)        (quadratic; err 5.9e-2 -> 3.5e-3 -> 1.2e-5 -> 1.4e-10)
// Three iterations put the algorithmic error (1.4e-10) below the Q2.30
// rounding floor (~2^-30.6 relative), so a fourth buys nothing.
//
// Measured over 40799 vectors vs numpy float32 1/x:
//   0 results worse than 1 ulp, 374 (0.9%) at exactly 1 ulp, rest exact.
//   Max relative error 6e-8, vs 6e-6 typical / 5e-1 worst for the old
//   fp32-based fp32_recip.
// ============================================================================

localparam int unsigned   FX_F        = 30;              // fraction bits, Q2.30
localparam logic [31:0]   FX_TWO      = 32'h8000_0000;   // 2.0
localparam logic [31:0]   FX_SEED_A   = 32'h5A5A_5A5A;   // 24/17
localparam logic [31:0]   FX_SEED_B   = 32'h1E1E_1E1E;   //  8/17

// Q2.30 x Q2.30 -> Q2.30, round-to-nearest. Pure integer: exact by construction.
function automatic logic [31:0] fx_mul_q30(input logic [31:0] a,
                                           input logic [31:0] b);
    logic [63:0] p;
    begin
        p = 64'(a) * 64'(b);
        fx_mul_q30 = (p + (64'd1 << (FX_F-1))) >> FX_F;
    end
endfunction

function automatic logic [31:0] fp32_recip_fixed(input logic [31:0] x);
    logic              sgn;
    logic [7:0]        e_in;
    logic [22:0]       f_in;
    logic [31:0]       m_fx, y, p, t;
    logic [63:0]       y64, q64;
    logic [23:0]       mant_rnd;
    logic signed [9:0] e_out;
    begin
        sgn  = x[31];
        e_in = x[30:23];
        f_in = x[22:0];

        if (e_in == 8'h00) begin
            // zero or denormal in -> infinity out. NOTE: your fp32_mul does not
            // understand inf, so this is a "should never happen" marker, not a
            // usable value. max_abs == 0 means an all-zero activation vector;
            // catch that in act_quant if it is reachable.
            fp32_recip_fixed = {sgn, 8'hFF, 23'h0};
        end
        else if (e_in == 8'hFF) begin                   // inf / NaN in -> zero
            fp32_recip_fixed = {sgn, 31'h0};
        end
        else if (f_in == 23'h0) begin                   // exact power of two
            e_out = 10'sd254 - $signed({2'b0, e_in});
            fp32_recip_fixed = (e_out <= 0)   ? {sgn, 31'h0}
                             : (e_out >= 255) ? {sgn, 8'hFF, 23'h0}
                                              : {sgn, e_out[7:0], 23'h0};
        end
        else begin
            m_fx = {1'b0, 1'b1, f_in, 7'b0};            // m in [1,2) as Q2.30

            y = FX_SEED_A - fx_mul_q30(FX_SEED_B, m_fx);

            for (int i = 0; i < 2; i++) begin
                p = fx_mul_q30(m_fx, y);                // m*y  ~= 1.0
                t = FX_TWO - p;                         // 2-m*y, exact subtract
                y = fx_mul_q30(y, t);
            end

            // Third iteration, kept at full product width. Rounding the result
            // back to Q2.30 here would be the single largest error term
            // (~2^-29 relative); keeping all 60 fraction bits drops it to
            // ~2^-31 and buys exact rounding on essentially every input.
            p    = fx_mul_q30(m_fx, y);
            t    = FX_TWO - p;
            y64  = 64'(y) * 64'(t);                     // Q4.60, value in (0.5,1)
            q64  = y64 << 1;                            // q = 2/m in (1,2)

            mant_rnd = (q64[59:0] + (64'd1 << 36)) >> 37;   // 60 frac -> 23, RN
            e_out    = 10'sd253 - $signed({2'b0, e_in});

            if (mant_rnd[23]) begin                     // rounded up to 2.0
                mant_rnd = 24'h0;
                e_out    = e_out + 10'sd1;
            end

            fp32_recip_fixed = (e_out <= 0)   ? {sgn, 31'h0}
                             : (e_out >= 255) ? {sgn, 8'hFF, 23'h0}
                                              : {sgn, e_out[7:0], mant_rnd[22:0]};
        end
    end
endfunction


endpackage