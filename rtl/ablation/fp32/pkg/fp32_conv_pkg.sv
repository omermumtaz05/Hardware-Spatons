package fp32_conv_pkg;

    parameter BIT_WIDTH_FP   = 32;
    parameter BIT_WIDTH_UINT = 8;

    parameter FP_EXP_WIDTH  = 8;
    parameter FP_MANT_WIDTH = 23;

    // Q8.8 fixed-point format
    parameter Q_RAW_WIDTH = 16;   // total bits of the stored Q8.8 word
    parameter Q_FRAC      = 8;    // fractional bits (value = raw / 2^Q_FRAC)

    localparam exp_bias = 127;

    function automatic logic [BIT_WIDTH_UINT-1:0] fp32_to_int(input logic [BIT_WIDTH_FP-1:0] f);
        logic [FP_EXP_WIDTH-1:0]  exp;
        logic [FP_MANT_WIDTH-1:0] mant;
        logic [FP_MANT_WIDTH:0]   sig;
        int e;
        begin
            exp  = f[FP_MANT_WIDTH + FP_EXP_WIDTH - 1 : FP_MANT_WIDTH];
            mant = f[FP_MANT_WIDTH-1:0];
            sig  = {1'b1, mant};
            e    = int'(exp) - exp_bias;
            if (e < 0)
                fp32_to_int = '0;                         // value < 1 -> 0
            else if (e >= FP_MANT_WIDTH)
                fp32_to_int = sig << (e - FP_MANT_WIDTH); // large integer
            else
                fp32_to_int = sig >> (FP_MANT_WIDTH - e); // drop fractional bits
        end
    endfunction

    // Q8.8 (unsigned) -> IEEE-754 FP32.  value = raw / 2^Q_FRAC
    function automatic logic [BIT_WIDTH_FP-1:0] q88_to_fp32(input logic [Q_RAW_WIDTH-1:0] raw);
        integer i;
        logic [Q_RAW_WIDTH-1:0]  m;
        logic [FP_EXP_WIDTH-1:0] exp;
        logic [FP_MANT_WIDTH-1:0] mant;
        integer msb;
        begin
            if (raw == '0) begin
                q88_to_fp32 = '0;
            end else begin
                // find MSB position of raw (0 .. Q_RAW_WIDTH-1)
                msb = 0;
                for (i = 0; i < Q_RAW_WIDTH; i = i + 1)
                    if (raw[i]) msb = i;

                // value = raw * 2^-Q_FRAC.  Normalized: 1.mant x 2^(msb-Q_FRAC)
                exp = exp_bias + msb - Q_FRAC;

                // shift raw left so leading 1 lands at bit (Q_RAW_WIDTH-1),
                // then take the bits below it and left-align into the mantissa
                m    = raw << (Q_RAW_WIDTH - 1 - msb);
                mant = {m[Q_RAW_WIDTH-2:0], {(FP_MANT_WIDTH-(Q_RAW_WIDTH-1)){1'b0}}};

                q88_to_fp32 = {1'b0, exp, mant};
            end
        end
    endfunction
// add alongside fp32_to_int, matching its style
parameter BIT_WIDTH_ACT = 12;

function automatic logic signed [BIT_WIDTH_ACT-1:0] fp32_to_sint_round(input logic [BIT_WIDTH_FP-1:0] f);
    logic sign;
    logic [FP_EXP_WIDTH-1:0]  exp;
    logic [FP_MANT_WIDTH-1:0] mant;
    logic [FP_MANT_WIDTH:0]   sig;
    int e, shift_amt;
    logic [FP_MANT_WIDTH+1:0] magnitude;
    localparam int QMAX = (1 << (BIT_WIDTH_ACT-1)) - 1;
    localparam int QMIN = -(1 << (BIT_WIDTH_ACT-1));
    begin
        sign = f[BIT_WIDTH_FP-1];
        exp  = f[FP_MANT_WIDTH +: FP_EXP_WIDTH];
        mant = f[FP_MANT_WIDTH-1:0];
        sig  = {1'b1, mant};
        e    = int'(exp) - exp_bias;

        if (e - FP_MANT_WIDTH >= 0) begin
            magnitude = sig << (e - FP_MANT_WIDTH);        // will clamp below anyway
        end else begin
            shift_amt = FP_MANT_WIDTH - e;
            if (shift_amt >= FP_MANT_WIDTH + 2)
                magnitude = '0;                             // too small to round up to 1
            else begin
                magnitude = sig >> shift_amt;
                if (sig[shift_amt-1])                       // round-half-up on dropped bit
                    magnitude = magnitude + 1'b1;
            end
        end

        if (sign)
            fp32_to_sint_round = (magnitude > -QMIN) ? QMIN : -$signed({1'b0,magnitude});
        else
            fp32_to_sint_round = (magnitude > QMAX)  ? QMAX :  $signed({1'b0,magnitude});
    end
endfunction

parameter ACC_WIDTH = 32;   // must match dot_prod_int's BW_ACC
 
function automatic logic [BIT_WIDTH_FP-1:0] sint_to_fp32(input logic signed [ACC_WIDTH-1:0] val);
    logic                       sign;
    logic [ACC_WIDTH-1:0]       mag;
    logic [ACC_WIDTH-1:0]       shifted;
    logic [FP_EXP_WIDTH-1:0]    exp;
    logic [FP_MANT_WIDTH-1:0]   mant;
    integer                     msb;
    integer                     i;
    begin
        if (val == '0) begin
            sint_to_fp32 = '0;                       // +0.0
        end else begin
            sign = val[ACC_WIDTH-1];
            mag  = sign ? (~val + 1'b1) : val;        // two's complement -> magnitude
 
            // find MSB position of mag (0 .. ACC_WIDTH-1)
            msb = 0;
            for (i = 0; i < ACC_WIDTH; i = i + 1)
                if (mag[i]) msb = i;
 
            // value = mag = 1.xxxx * 2^msb
            exp = exp_bias + msb;
 
            // shift so the leading 1 lands at bit (ACC_WIDTH-1), then take the
            // bits just below it as the mantissa (implicit 1 is dropped)
            shifted = mag << (ACC_WIDTH - 1 - msb);
 
            mant = shifted[ACC_WIDTH-2 -: FP_MANT_WIDTH];
 
            sint_to_fp32 = {sign, exp, mant};
        end
    end
endfunction

endpackage