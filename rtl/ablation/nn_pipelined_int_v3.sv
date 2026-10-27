// nn_pipelined_int_v3.sv
//
// A12 photo-stage MLP -- ALL-INTEGER, END TO END. fp32 is GONE from this
// module and from post_mlp_int. Nothing downstream of the line buffer touches
// a float any more.
//
// ===========================================================================
// WHAT CHANGED vs v2  (v2 had fp32 in exactly one place: the final combine)
//
//   port   z [31:0]  ->  z_idx [11:0]      the sigmoid LUT ADDRESS, not a value
//   param  LAMBDA3   ->  IDX_M             the fp32 scale, folded into that address
//   stage2 sint_to_fp32 -> saturate to 23b
//   stage3 fp32_mul     -> x IDX_M, >>> IDX_SH, + LUT_N/2, clamp
//   imports fp32_pkg / fp32_conv_pkg  DELETED
//
// ZLAT is still 3, so cv_sr and out_valid at the bottom are byte-identical.
//
// ===========================================================================
// THE IDEA -- post_mlp's INDEX IS AFFINE IN zi, SO LAMBDA3 NEED NOT BE A FLOAT
//
// The v2 header's round-2 note ended with "post_mlp.sv still has the round-1
// problem ... the same fold may apply there too." It does, and it applies one
// step EARLIER than expected. post_mlp computed:
//
//     z   = LAMBDA3 * zi                                (here, in fp32)
//     idx = floor((z + 8)*INV_STEP + 0.5)               (there, in fp32)
//
// Substitute and the float disappears:
//
//     idx = floor(zi * (LAMBDA3*INV_STEP)) + LUT_N/2
//         = ((zi * IDX_M) >>> IDX_SH) + LUT_N/2         IDX_M = round(C*2^IDX_SH)
//
// LAMBDA3 was only ever a scale on the way into a table lookup, and a scale on
// the way into a table lookup is just a different table stride. So it becomes
// ONE integer constant and the value `z` never has to exist.
//
// DELETED PER CHANNEL, PER PIXEL:
//   sint_to_fp32   LZD + normalise barrel shift
//   fp32_mul       24x24 mantissa multiply -> DSP blocks
// and, downstream in post_mlp:
//   fp32_add, fp32_mul, fp32_add, fp32_to_int_floor     (index)
//   fp32_mul, fp32_add, fp32_add                        (combine)
//   4x fp32_le                                          (saturate + clamp)
//
// The 24x24 mantissa multiplies are the DSP win. The barrel shifters inside
// sint_to_fp32 and fp32_add are the Fmax win -- z_flat[25..27], the fp32
// exponent field, is where every failing path used to terminate.
//
// ===========================================================================
// ACCURACY -- 500k random (zi, centre) vs the exact-arithmetic fp32 model:
//   bit-exact 99.917%,  within 1 LSB 100.000%,  mean 0.00083 LSB,  max 1 LSB
//
// Context: the design already carries mean 0.542 / max 11.5 LSB against the
// PyTorch golden, dominated by 10-bit weight quantisation. This change adds
// ~650x less error than that. It is free.
//
// The residual 0.08% are pixels whose zi sits within one ULP of a LUT bin
// boundary; IDX_M carries ~20 significant bits against a required ~12, so the
// index differs by at most 1 and the value by at most 1 LSB. Reproduce with
// gen_post_mlp_luts.py --check.
//
// ===========================================================================
// REGENERATE IDX_M AND data_files/delta_lut.hex TOGETHER.
// Both encode INV_STEP. Rebuild only one and they disagree by a scale factor:
// the image comes out washed out while every handshake assertion still passes.
//     python3 gen_post_mlp_luts.py --check
//
// ===========================================================================
// (Everything below this line is unchanged from v2 and is reproduced verbatim.)
//
// DSP BUDGET -- Analysis & Synthesis reported 146; the 5CSEMA5F31C6 has 87.
//
//   ~51  102 neuron MACs, 12x10                     KEPT -- this is the work
//   ~20  5 x requant_int, FOLDS x (32x18) reciprocal multiply   DELETED
//   ~24  4 x layer_int, 32x32 "maxacc * X"                      DELETED
//   ~4   4 x layer_int, 24x18 "gmant * K_MANT"                  DELETED
//   ~16  4 x layer_int, AP_FOLDS x (12x24) "racc * m_fx"        -> ALM logic
//   ~3   3 x final-combine fp32_mul, 24x24                      DELETED (v3)
//   ---
//   ~51  after, leaving ~36 for the geometric stage (~16 for bilinear)
//
// The enabling change is in requant_int: normalising by a POWER OF TWO
// instead of by the exact abs-max. That removes its own multipliers AND makes
// the downstream scale correction a pure shift, which collapses layer_int's
// entire M chain (see the header of layer_int.sv for the derivation).
//
// It also removes the 18-cycle restoring divide from every scan. Measured on
// the streaming testbench: initiation interval 151 -> 90 cycles per pixel,
// L2 bound.
//
// ACCURACY -- measured at the PIXEL, after post_mlp, over 9 test images
// (natural, gradient, checkerboard, white noise, flat black/grey/white, hard
// edges, very dark). Error in 8-bit LSBs vs the float model:
//
//   config                              mean    p99    max   >2LSB
//   per-tensor weights                 1.344  9.107 22.161  26.35%
//   PER-CHANNEL on layer1+layer2       0.542  3.403 11.500   6.24%
//
// The error that mattered was 10-bit WEIGHT quantisation, which was there all
// along -- layer2's max|w| is 67x its median, so the median weight resolved to
// ~2.9 of its 10 bits. Per-channel scales on L1+L2 fix that for ZERO DSP
// blocks; the per-neuron multiply is held in ALMs by multstyle in layer_int.sv.
//
// PIPELINING SCHEME (unchanged)
//   L1 and residual accept each pixel together since they share the feature
//   vector. Residual pushes into a FIFO and frees itself immediately, so the
//   input side can accept pixel N+1 while N-2..N are still in flight.
//
// CONSTANTS -- all generated by gen_shift_consts.py from the checkpoint.
// MSH is gone; it is folded into K_SHIFT. The applied shift is
//     msh = K_SHIFT - s_in - s_acc,  computed at runtime.
// Measured msh minima over the 9 test images: L1 5, RES 6, L2 9, L3 8.

// NO IMPORTS. fp32_pkg and fp32_conv_pkg are no longer referenced anywhere in
// this module -- sint_to_fp32 and fp32_mul were their only users.

module nn_pipelined_int_v3 #(
    parameter int IN_DIM    = 29,
    parameter int HIDDEN    = 48,
    parameter int OUT_DIM   = 3,
    parameter int FEAT_W    = 8,
    parameter int BW        = 32,     // internal zi width cap (was: fp32 port width)
    parameter int BW_W      = 10,
    parameter int BW_ACT    = 12,
    parameter int BW_ACC    = 28,     // was 32: 22-bit product + clog2(48) = 28
    parameter int QF        = 24,
    parameter int BW_FX     = 26,
    parameter int SHW       = 5,      // requant shift exponent width

    // post_mlp_int LUT address width. Must be a PARAMETER, not a body
    // localparam -- it appears in the port list, and a body localparam is not
    // visible there. Same reason requant_int declares SHW as a parameter.
    parameter int IDXW      = 12,     // LUT_N = 4096

    parameter int NUM_FOLDS_1      = 1,
    parameter int NUM_FOLDS_2      = 1,
    parameter int NUM_FOLDS_3      = 1,
    parameter int NUM_FOLDS_R      = 1, // 1 MAC per neuron. Raising this to 2
                                      // doubles the neuron DSP count to ~102
                                      // and blows the budget -- don't.
                                      // COMPUTE is 50 of the 90-cycle II, so
                                      // this is also the throughput floor.
    parameter int RQ_FOLDS       = 8, // requant has no DSPs now, so this is
                                      // free to raise; costs ~100 ALMs/lane
    parameter int AP_FOLDS       = 8,
    parameter int RES_FIFO_DEPTH = 4,

    // ---- per-layer scale constants, from gen_shift_consts.py ----
    // L1 and L2 read per-neuron K_MANT ROMs (KM_PER_CHANNEL=1 below), so
    // their scalars here are only the ROM maxima and are otherwise unused.
    parameter int L1_KM = 131072, parameter int L1_KS = 31,
    parameter int L2_KM = 211738, parameter int L2_KS = 27,
    parameter int L3_KM = 246485, parameter int L3_KS = 24,
    parameter int RS_KM = 131072, parameter int RS_KS = 31,

    // ---- per-channel K_MANT ROMs for L1 and L2 ----
    // L1 and L2 use one weight scale PER NEURON, which cuts final pixel error
    // 3.1x for zero DSP blocks (the multiply goes to ALMs). L3 and residual
    // gain nothing from per-channel and stay on the scalar K_MANT above.
    // Generate both files with gen_shift_consts.py --per-channel.
    parameter L1_KM_FILE = "data_files/km_layer1.hex",
    parameter L2_KM_FILE = "data_files/km_layer2.hex",

    // ---- final output scale ----
    // LAMBDA3 IS GONE. It was s_w[layer3]/2^F applied as an fp32 multiply, but
    // its only consumer was post_mlp's LUT index, which is affine in zi. So it
    // is now an integer stride:
    //
    //     IDX_M = round( LAMBDA3 * ((2^IDXW - 1)/16) * 2^IDX_SH )
    //
    // for LAMBDA3 = 32'h3668be95, IDXW = 12, IDX_SH = 24.
    //
    // MUST be regenerated together with data_files/delta_lut.hex -- both encode
    // the same INV_STEP. gen_post_mlp_luts.py emits both and --check proves
    // they agree against the float model.
    parameter int IDX_M    = 14892,

    // RES_RATIO is UNCHANGED. The residual branch is folded into the layer3
    // branch as an integer ratio:
    //     RES_RATIO = round( (s_w[residual]/s_w[layer3]) * 2^RSCALE )
    parameter int RES_RATIO = 102702,   // 0.097943817 * 2^20

    // ---- memory init ----
    parameter L1_W_Q = "data_files/a12_layer1_w_q.hex",
    parameter L1_B   = "data_files/bias_int_layer1.hex",
    parameter L2_W_Q = "data_files/a12_layer2_w_q.hex",
    parameter L2_B   = "data_files/bias_int_layer2.hex",
    parameter L3_W_Q = "data_files/a12_layer3_w_q.hex",
    parameter L3_B   = "data_files/bias_int_layer3.hex",
    parameter RS_W_Q = "data_files/a12_residual_w_q.hex",
    parameter RS_B   = "data_files/bias_int_residual.hex"
) (
    input  logic clock,
    input  logic reset,

    input  logic in_valid,
    output logic in_ready,
    input  logic [FEAT_W-1:0] f [IN_DIM-1:0],

    output logic out_valid,
    input  logic out_ready,
    // WAS: output logic [BW-1:0] z [OUT_DIM-1:0]  -- an fp32 value.
    // NOW: the delta-LUT address for post_mlp_int. 12 bits, not 32.
    output logic [IDXW-1:0] z_idx [OUT_DIM-1:0]
);

    // ================= weights / integer biases =================
    logic signed [BW_W-1:0] w1 [HIDDEN*IN_DIM-1:0];   logic signed [31:0] b1 [HIDDEN-1:0];
    logic signed [BW_W-1:0] w2 [HIDDEN*HIDDEN-1:0];   logic signed [31:0] b2 [HIDDEN-1:0];
    logic signed [BW_W-1:0] w3 [OUT_DIM*HIDDEN-1:0];  logic signed [31:0] b3 [OUT_DIM-1:0];
    logic signed [BW_W-1:0] wr [OUT_DIM*IN_DIM-1:0];  logic signed [31:0] br [OUT_DIM-1:0];

    initial begin
        $readmemh(L1_W_Q, w1); $readmemh(L1_B, b1);
        $readmemh(L2_W_Q, w2); $readmemh(L2_B, b2);
        $readmemh(L3_W_Q, w3); $readmemh(L3_B, b3);
        $readmemh(RS_W_Q, wr); $readmemh(RS_B, br);
    end

    // ================= integer feature front end =================
    logic signed [BW_FX-1:0]  f_fx  [IN_DIM-1:0];
    logic signed [31:0]       f_fx32[IN_DIM-1:0];

    genvar fi;
    generate
        for (fi = 0; fi < IN_DIM; fi++) begin : gen_f2fx
            // v/255 in Q0.24 IS {v,v,v}.  v * 0x010101 = v * 65793 = v/255 * 2^24
            assign f_fx[fi]   = $signed({{(BW_FX-3*FEAT_W){1'b0}}, f[fi], f[fi], f[fi]});
            assign f_fx32[fi] = 32'(f_fx[fi]);
        end
    endgenerate

    logic                     rq0_iv, rq0_ir, rq0_ov, rq0_or;
    logic signed [BW_ACT-1:0] q0 [IN_DIM-1:0];
    logic [IN_DIM*BW_ACT-1:0] q0_flat;
    logic [SHW-1:0]           xshift;      // was `xmax`: now the exponent

    requant_int #(
        .N(IN_DIM), .BW_IN(32), .BW_OUT(BW_ACT), .FOLDS(RQ_FOLDS), .SHW(SHW)
    ) u_rq0 (
        .clock(clock), .reset(reset),
        .in_valid(rq0_iv), .in_ready(rq0_ir), .vec_in(f_fx32),
        .out_valid(rq0_ov), .out_ready(rq0_or),
        .q_out(q0), .q_flat(q0_flat), .vshift(xshift)
    );

    // ================= inter-layer buses: INTEGER =================
    logic signed [BW_ACC-1:0] h1 [HIDDEN-1:0];
    logic signed [BW_ACC-1:0] h2 [HIDDEN-1:0];
    logic signed [BW_ACC-1:0] y  [OUT_DIM-1:0];
    logic signed [BW_ACC-1:0] yr [OUT_DIM-1:0];

    logic l1_iv, l1_ir, l1_ov, l1_or;
    logic l2_iv, l2_ir, l2_ov, l2_or;
    logic l3_iv, l3_ir, l3_ov, l3_or;
    logic res_iv, res_ir, res_ov, res_or;

    logic                    fifo_full, fifo_empty, fifo_wr, fifo_rd;
    logic [3*BW_ACC-1:0]     fifo_wdata, fifo_rdata;
    logic signed [BW_ACC-1:0] yr_head [OUT_DIM-1:0];

    // Tie-offs for the unused port on each layer. An assignment pattern
    // ('{default:'0}) is NOT legal in a port connection -- it needs a real
    // signal. These optimise away completely.
    logic signed [BW_ACC-1:0] tie_vec  [IN_DIM-1:0];
    logic signed [BW_ACT-1:0] tie_q48  [HIDDEN-1:0];
    generate
        for (fi = 0; fi < IN_DIM; fi++) begin : gen_tie29
            assign tie_vec[fi] = '0;
        end
        for (fi = 0; fi < HIDDEN; fi++) begin : gen_tie48
            assign tie_q48[fi] = '0;
        end
    endgenerate

    // ---- input side: L1 and residual accept the same pixel together ----
    logic both_ready;
    assign both_ready = l1_ir && res_ir;
    assign rq0_iv     = in_valid;
    assign in_ready   = rq0_ir;
    assign rq0_or     = rq0_ov && both_ready;
    assign l1_iv      = rq0_ov && both_ready;
    assign res_iv     = rq0_ov && both_ready;

    // ================= Layer 1: 29 -> 48, ReLU =================
    layer_int #(
        .INPUT_SIZE(IN_DIM), .NUM_NEURONS(HIDDEN),
        .NUM_FOLDS(NUM_FOLDS_1), .RQ_FOLDS(RQ_FOLDS), .AP_FOLDS(AP_FOLDS),
        .USE_RELU(1), .INPUT_IS_Q(1),
        .BW_ACT(BW_ACT), .BW_W(BW_W), .BW_ACC(BW_ACC), .BW_Y(BW_ACC),
        .K_MANT(L1_KM), .K_SHIFT(L1_KS), .SHW(SHW),
        .KM_PER_CHANNEL(1), .KM_FILE(L1_KM_FILE)
    ) u_l1 (
        .clock(clock), .reset(reset),
        .in_valid(l1_iv), .in_ready(l1_ir),
        .in_vec(tie_vec), .in_q(q0), .xin_shift(xshift),
        .W_flat(w1), .B_int(b1),
        .out_valid(l1_ov), .out_ready(l1_or), .out_vec(h1)
    );

    // ================= Layer 2: 48 -> 48, ReLU =================
    assign l2_iv = l1_ov;  assign l1_or = l2_ir;
    layer_int #(
        .INPUT_SIZE(HIDDEN), .NUM_NEURONS(HIDDEN),
        .NUM_FOLDS(NUM_FOLDS_2), .RQ_FOLDS(RQ_FOLDS), .AP_FOLDS(AP_FOLDS),
        .USE_RELU(1), .INPUT_IS_Q(0),
        .BW_ACT(BW_ACT), .BW_W(BW_W), .BW_ACC(BW_ACC), .BW_Y(BW_ACC),
        .K_MANT(L2_KM), .K_SHIFT(L2_KS), .SHW(SHW),
        .KM_PER_CHANNEL(1), .KM_FILE(L2_KM_FILE)
    ) u_l2 (
        .clock(clock), .reset(reset),
        .in_valid(l2_iv), .in_ready(l2_ir),
        .in_vec(h1), .in_q(tie_q48), .xin_shift('0),
        .W_flat(w2), .B_int(b2),
        .out_valid(l2_ov), .out_ready(l2_or), .out_vec(h2)
    );

    // ================= Layer 3: 48 -> 3, linear =================
    assign l3_iv = l2_ov;  assign l2_or = l3_ir;
    layer_int #(
        .INPUT_SIZE(HIDDEN), .NUM_NEURONS(OUT_DIM),
        .NUM_FOLDS(NUM_FOLDS_3), .RQ_FOLDS(RQ_FOLDS), .AP_FOLDS(AP_FOLDS),
        .USE_RELU(0), .INPUT_IS_Q(0),
        .BW_ACT(BW_ACT), .BW_W(BW_W), .BW_ACC(BW_ACC), .BW_Y(BW_ACC),
        .K_MANT(L3_KM), .K_SHIFT(L3_KS), .SHW(SHW)
    ) u_l3 (
        .clock(clock), .reset(reset),
        .in_valid(l3_iv), .in_ready(l3_ir),
        .in_vec(h2), .in_q(tie_q48), .xin_shift('0),
        .W_flat(w3), .B_int(b3),
        .out_valid(l3_ov), .out_ready(l3_or), .out_vec(y)
    );

    // ================= Residual: 29 -> 3, linear =================
    layer_int #(
        .INPUT_SIZE(IN_DIM), .NUM_NEURONS(OUT_DIM),
        .NUM_FOLDS(NUM_FOLDS_R), .RQ_FOLDS(RQ_FOLDS), .AP_FOLDS(AP_FOLDS),
        .USE_RELU(0), .INPUT_IS_Q(1),
        .BW_ACT(BW_ACT), .BW_W(BW_W), .BW_ACC(BW_ACC), .BW_Y(BW_ACC),
        .K_MANT(RS_KM), .K_SHIFT(RS_KS), .SHW(SHW)
    ) u_res (
        .clock(clock), .reset(reset),
        .in_valid(res_iv), .in_ready(res_ir),
        .in_vec(tie_vec), .in_q(q0), .xin_shift(xshift),
        .W_flat(wr), .B_int(br),
        .out_valid(res_ov), .out_ready(res_or), .out_vec(yr)
    );

    // ================= residual FIFO (payload narrowed to 3*BW_ACC) ==========
    assign fifo_wr    = res_ov && !fifo_full;
    assign res_or     = fifo_wr;
    assign fifo_wdata = {yr[2], yr[1], yr[0]};

    fifo_residual #(.WIDTH(3*BW_ACC), .DEPTH(RES_FIFO_DEPTH)) u_res_fifo (
        .clock(clock), .reset(reset),
        .wr_en(fifo_wr), .wr_data(fifo_wdata), .full(fifo_full),
        .rd_en(fifo_rd), .rd_data(fifo_rdata), .empty(fifo_empty)
    );

    assign yr_head[0] = $signed(fifo_rdata[1*BW_ACC-1 -: BW_ACC]);
    assign yr_head[1] = $signed(fifo_rdata[2*BW_ACC-1 -: BW_ACC]);
    assign yr_head[2] = $signed(fifo_rdata[3*BW_ACC-1 -: BW_ACC]);

    // ================= final combine =================
    //     z_idx = clamp( ((zi * IDX_M) >>> IDX_SH) + LUT_N/2 , 0, LUT_N-1 )
    //     zi    = y + ((yr * RES_RATIO) >>> RSCALE)
    //
    // in three registered stages -- SAME COUNT AS v2, so ZLAT is still 3 and
    // the cv_sr / out_valid block at the bottom is untouched:
    //
    //     1. zi -- integer combine of the layer3 and residual branches  (as v2)
    //     2. zc -- zi saturated to ZC bits              (v2: sint_to_fp32)
    //     3. zx -- index into post_mlp_int's delta LUT  (v2: fp32_mul LAMBDA3)
    //
    // =======================================================================
    // WHY zi IS SATURATED BEFORE THE MULTIPLY (stage 2)
    //
    // Only |zi| <= (LUT_N/2)/C ~= 2.31e6 can produce an in-range index --
    // everything past that clamps to 0 or LUT_N-1 anyway. Saturating to ZC=23
    // bits FIRST turns the constant multiply from 30x15 into 23x15, which with
    // multstyle="logic" is a 7-term shift-add tree in ALMs instead of a DSP
    // block. ZC=23 is not arbitrary: 2307260 needs 23 bits signed, printed by
    // gen_post_mlp_luts.py so you can re-check it if LAMBDA3 moves.
    //
    // Do NOT replace this with a truncating slice. A raw slice WRAPS, and a
    // wrapped large-magnitude zi lands at an arbitrary LUT index -- which shows
    // up as single-pixel salt-and-pepper on bright edges and is miserable to
    // trace back to here.
    //
    // =======================================================================
    // ACCURACY OF THE FOLD -- 500k random (zi, centre) vs the exact fp32 model:
    //   bit-exact 99.917%, within 1 LSB 100%, mean 0.00083 LSB, max 1 LSB.
    // IDX_M carries ~20 significant bits against a required ~12 (index error is
    // |zi| * dC, and |zi| is bounded by the saturate above), so the index is
    // identical except within one ULP of a bin boundary.
    // =======================================================================

    localparam int ZLAT   = 3;
    localparam int RSCALE = 20;                  // RES_RATIO is Q0.RSCALE
    localparam int BW_RR  = 18;                  // signed width of RES_RATIO
    localparam int BW_RP  = BW_ACC + BW_RR;      // yr * RES_RATIO
    // y + scaled yr, with headroom. BW_ACC is 28 by default; the streaming TB
    // overrides it to 32, which is what makes the clamp necessary.
    localparam int BW_ZI  = ((BW_ACC + 2) > BW) ? BW : (BW_ACC + 2);

    // ---- index-domain constants ----
    localparam int LUT_N  = 1 << IDXW;           // 4096
    localparam int IDX_SH = 24;                  // IDX_M is Q0.IDX_SH
    localparam int IDX_MW = 15;                  // signed width of IDX_M
    localparam int ZC     = 23;                  // saturated zi width
    localparam int BW_IP  = ZC + IDX_MW;         // zc * IDX_M, full width

    // Typed constants rather than inline casts. Writing
    //     yr_head[k] * signed'(BW_RR'(RES_RATIO))
    // CRASHES quartus_map with an access violation in
    // VeriTypeCast::Evaluate -> VeriBinaryOperator::Evaluate, i.e. a nested
    // cast inside a binary operator inside a generate-for. That is a Verific
    // front-end bug, but it is trivially avoidable: declare the constant with
    // the type you want and let plain signed assignment do the work.
    localparam logic signed [BW_RR-1:0]  RES_RATIO_S = RES_RATIO;
    localparam logic signed [IDX_MW-1:0] IDX_M_S     = IDX_M;
    localparam logic signed [BW_RP-1:0]  RHALF       = 1 <<< (RSCALE - 1);

    localparam logic signed [BW_ZI-1:0] ZC_MAX_W =  (1 <<< (ZC-1)) - 1;
    localparam logic signed [BW_ZI-1:0] ZC_MIN_W = -(1 <<< (ZC-1));
    localparam logic signed [ZC-1:0]    ZC_MAX   =  (1 <<< (ZC-1)) - 1;
    localparam logic signed [ZC-1:0]    ZC_MIN   = -(1 <<< (ZC-1));

    localparam logic signed [BW_IP-1:0] LUT_HALF   = LUT_N / 2;      // 2048
    localparam logic signed [BW_IP-1:0] LUT_MAX_S  = LUT_N - 1;      // 4095
    localparam logic signed [BW_IP-1:0] LUT_ZERO_S = 0;
    localparam logic        [IDXW-1:0]  LUT_MAX_U  = LUT_N - 1;

    logic signed [BW_ZI-1:0] zi_r [OUT_DIM-1:0];   // stage 1: integer combine
    logic signed [ZC-1:0]    zc_r [OUT_DIM-1:0];   // stage 2: saturated
    logic        [IDXW-1:0]  zx_r [OUT_DIM-1:0];   // stage 3: LUT index

    logic            core_valid;
    logic [ZLAT-1:0] cv_sr;

    // core_valid is a LEVEL: it stays high for as long as layer3 sits in HOLD
    // and the FIFO has a head. Both y and yr_head are stable across that whole
    // window, which is what makes the free-running pipeline below correct.
    assign core_valid = l3_ov && !fifo_empty;

    genvar k;
    generate
        for (k = 0; k < OUT_DIM; k++) begin : gen_add
            // NO CASTS ANYWHERE IN HERE -- see the RES_RATIO_S note above.
            // Every step is a plain assignment between SIGNED variables, which
            // sign-extends (or truncates) on its own. It is also immune to the
            // signed-size-cast portability trap described in layer_int.sv.

            // ---- stage 1 input: integer residual fold (UNCHANGED from v2) ----
            logic signed [BW_RP-1:0] rp;      // yr_head * RES_RATIO, full width
            logic signed [BW_ZI-1:0] yr_sc;   // that, scaled back down
            logic signed [BW_ZI-1:0] y_w;     // y widened to the sum width
            logic signed [BW_ZI-1:0] zi;

            always_comb begin
                rp    = yr_head[k] * RES_RATIO_S;   // signed x signed
                yr_sc = (rp + RHALF) >>> RSCALE;    // round half up, then scale
                y_w   = y[k];                       // signed -> wider signed
                zi    = y_w + yr_sc;
            end

            // ---- stage 2 input: saturate to ZC bits (v2: sint_to_fp32) ----
            logic signed [ZC-1:0] zc;

            always_comb begin
                if      (zi_r[k] > ZC_MAX_W) zc = ZC_MAX;
                else if (zi_r[k] < ZC_MIN_W) zc = ZC_MIN;
                else                         zc = zi_r[k];   // proven in range
            end

            // ---- stage 3 input: delta-LUT index (v2: fp32_mul LAMBDA3) ----
            // multstyle keeps the constant multiply out of the DSP blocks: it
            // is a 7-term shift-add tree, ~150 ALMs. Without this Quartus will
            // happily spend DSP blocks on a multiply by a 14-bit constant.
            (* multstyle = "logic" *) logic signed [BW_IP-1:0] ip;
            logic signed [BW_IP-1:0] ish;
            logic signed [BW_IP-1:0] isum;
            logic        [IDXW-1:0]  zx;

            always_comb begin
                ip   = zc_r[k] * IDX_M_S;      // 23 x 15 signed
                ish  = ip >>> IDX_SH;          // arithmetic == floor, incl. negatives
                isum = ish + LUT_HALF;

                if      (isum <= LUT_ZERO_S) zx = '0;
                else if (isum >= LUT_MAX_S)  zx = LUT_MAX_U;
                else                         zx = isum[IDXW-1:0];  // proven in range
            end

            // NO enable, deliberately. The inputs hold still while layer3 is in
            // HOLD, so after ZLAT cycles zx_r is correct and simply stays
            // correct until the sources change. An enable here bought nothing
            // and was one more thing to get wrong.
            always_ff @(posedge clock) begin
                zi_r[k] <= zi;      // stage 1: integer combine
                zc_r[k] <= zc;      // stage 2: saturate
                zx_r[k] <= zx;      // stage 3: -> LUT index
            end

            assign z_idx[k] = zx_r[k];
        end
    endgenerate

    always_ff @(posedge clock or posedge reset) begin
        if (reset) cv_sr <= '0;
        else       cv_sr <= {cv_sr[ZLAT-2:0], core_valid};
    end

    // out_valid = "the core is valid NOW, and has been for ZLAT cycles".
    //
    // The `core_valid &&` term is the load-bearing part. Without it out_valid
    // is a delayed level with no interlock: it stays asserted after l3_or
    // fires, and a greedy sink takes several transfers of the SAME pixel.
    // Keeping the live level in the expression makes out_valid collapse the
    // instant layer3 leaves HOLD -- exactly the self-limiting behaviour the
    // original combinational `l3_ov && !fifo_empty` had, just ZLAT later.
    assign out_valid = core_valid && cv_sr[ZLAT-1];
    assign l3_or     = out_valid && out_ready;
    assign fifo_rd   = out_valid && out_ready;
    // synthesis translate_off
    /* verilator lint_off BLKSEQ */
    int x_min = 99, x_max = 0, x_n = 0;
    always @(posedge clock)
        if (!reset && rq0_ov && rq0_or) begin
            if (int'(xshift) < x_min) x_min = int'(xshift);
            if (int'(xshift) > x_max) x_max = int'(xshift);
            x_n++;
        end
    final $display("%m  u_rq0   min=%0d max=%0d spread=%0d n=%0d",
                x_min, x_max, x_max-x_min, x_n);
    /* verilator lint_on BLKSEQ */
    // synthesis translate_on
endmodule