

module nn_pipelined_int_v4 #(
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

    parameter int NUM_FOLDS_1      = 2,
    parameter int NUM_FOLDS_2      = 2,
    parameter int NUM_FOLDS_3      = 2,
    parameter int NUM_FOLDS_R      = 2, // 1 MAC per neuron. Raising this to 2
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
        parameter int L1_KM = 131072, parameter int L1_KS = 20,   // 31 - 11          msh = 20 - xshift = 7
    parameter int L2_KM = 211738, parameter int L2_KS =  10,   // 27 -  9 - 10     msh = 8   CONSTANT
    parameter int L3_KM = 246485, parameter int L3_KS =  8,   // 24 -  6 - 10     msh = 8   CONSTANT
    parameter int RS_KM = 131072, parameter int RS_KS = 19,    // 31 - 12          msh = 19 - xshift = 6

    // ---- per-channel K_MANT ROMs for L1 and L2 ----
    // L1 and L2 use one weight scale PER NEURON, which cuts final pixel error
    // 3.1x for zero DSP blocks (the multiply goes to ALMs). L3 and residual
    // gain nothing from per-channel and stay on the scalar K_MANT above.
    // Generate both files with gen_shift_consts.py --per-channel.
    parameter L1_KM_FILE = "data_files/km_layer1.hex",
    parameter L2_KM_FILE = "data_files/km_layer2.hex",


    parameter int IDX_M    = 14892,


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

    logic signed [BW_ACT-1:0] q0 [IN_DIM-1:0];
    logic [SHW-1:0]           xshift;      // was `xmax`: now the exponent

    localparam int XSHIFT = 13;   // provable max: msb(255*65793)=23, 23-10=13

    genvar qi;
    generate
        for (qi = 0; qi < IN_DIM; qi++) begin : gen_q0
            wire [31:0] q0_rnd = f_fx32[qi] + 32'd4096;      // round half up
            wire [31:0] q0_sh  = q0_rnd >> XSHIFT;           // constant -> wires
            assign q0[qi] = (q0_sh > 32'd2047) ? 12'sd2047
                                               : $signed(q0_sh[11:0]);
        end
    endgenerate
    assign xshift = XSHIFT;
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
        assign in_ready   = both_ready;
    assign l1_iv      = in_valid && both_ready;
    assign res_iv     = in_valid && both_ready;
   
    // ================= Layer 1: 29 -> 48, ReLU =================
    layer_int_v2 #(
        .INPUT_SIZE(IN_DIM), .NUM_NEURONS(HIDDEN),
        .NUM_FOLDS(NUM_FOLDS_1), .RQ_FOLDS(RQ_FOLDS), .AP_FOLDS(AP_FOLDS),
        .USE_RELU(1), .INPUT_IS_Q(1), .S_ACC(11),
        .BW_ACT(BW_ACT), .BW_W(BW_W), .BW_ACC(BW_ACC), .BW_Y(BW_ACC),
        .K_MANT(L1_KM), .K_SHIFT(L1_KS), .SHW(SHW), .XIN_SHIFT(XSHIFT),
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
    layer_int_v2 #(
        .INPUT_SIZE(HIDDEN), .NUM_NEURONS(HIDDEN),
        .NUM_FOLDS(NUM_FOLDS_2), .RQ_FOLDS(RQ_FOLDS), .AP_FOLDS(AP_FOLDS),
        .USE_RELU(1), .INPUT_IS_Q(0),  .S_IN(10),  .S_ACC(7),
        .BW_ACT(BW_ACT), .BW_W(BW_W), .BW_ACC(BW_ACC),  .BW_Y(BW_ACC),
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
    layer_int_v2 #(
        .INPUT_SIZE(HIDDEN), .NUM_NEURONS(OUT_DIM),
        .NUM_FOLDS(NUM_FOLDS_3), .RQ_FOLDS(RQ_FOLDS), .AP_FOLDS(AP_FOLDS),
        .USE_RELU(0), .INPUT_IS_Q(0),  .S_IN(8),  .S_ACC(8),
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
    layer_int_v2 #(
        .INPUT_SIZE(IN_DIM), .NUM_NEURONS(OUT_DIM),
        .NUM_FOLDS(NUM_FOLDS_R), .RQ_FOLDS(RQ_FOLDS), .AP_FOLDS(AP_FOLDS),
        .USE_RELU(0), .INPUT_IS_Q(1), .S_ACC(12),  .DYN_ACC(0),
        .BW_ACT(BW_ACT), .BW_W(BW_W), .BW_ACC(BW_ACC), .BW_Y(BW_ACC),
        .K_MANT(RS_KM), .K_SHIFT(RS_KS), .SHW(SHW), .XIN_SHIFT(XSHIFT)
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


    assign core_valid = l3_ov && !fifo_empty;

    genvar k;
    generate
        for (k = 0; k < OUT_DIM; k++) begin : gen_add

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

    assign out_valid = core_valid && cv_sr[ZLAT-1];
    assign l3_or     = out_valid && out_ready;
    assign fifo_rd   = out_valid && out_ready;

endmodule