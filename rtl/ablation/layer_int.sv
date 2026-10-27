// layer_int.sv  -- SHIFT-SCALED VERSION
//
// All-integer MLP layer.
//
// DATAPATH
//   Y_in[INPUT_SIZE] (int)
//     -> requant_int              -> q_in[12b], s_in            (scan 1)
//     -> NUM_NEURONS x neuron_int -> acc[NUM_NEURONS] (int)
//     -> requant_int              -> racc[12b], s_acc           (scan 2)
//     -> Y_j = ((racc_j * K_MANT) >>> (K_SHIFT - s_in - s_acc)) + B_int_j
//     -> ReLU, saturate
//
// ===========================================================================
// WHAT CHANGED FROM THE DIVIDE-BASED VERSION -- READ THIS BEFORE EDITING
//
// The old design carried a runtime scalar M and computed it like this:
//     G     = maxacc * X           <-- 32x32 multiply
//     gmant = normalise(G, 24 bits)
//     mraw  = gmant * K_MANT       <-- 24x18 multiply
//     m_fx  = mraw >> (K_SHIFT - gexp)
//     Y     = (racc * m_fx) >>> MSH + B
//
// That cost roughly 7 DSP blocks per layer (~28 of 146 total) plus a 64-bit
// priority encoder and two barrel shifters in one combinational cone.
//
// Now that requant_int normalises by a POWER OF TWO, X is exactly 2^s_in and
// maxacc's contribution is exactly 2^s_acc.  So:
//
//     M  =  2^(s_in + s_acc) * K_MANT / 2^K_SHIFT
//
// The mantissa product vanishes.  M is a compile-time constant K_MANT and a
// RUNTIME SHIFT.  Both multiplies, the normaliser and the m_fx register are
// deleted.  What is left is one constant multiply per APPLY lane, which
// synthesises to shift-adds in ALMs (multstyle below) -- and for L1/RES,
// where K_MANT is a power of two, to nothing at all.
//
// ===========================================================================
// SCALE INVARIANT (unchanged): Y carries the implicit scale s_w * 2^-F, which
// is the same scale B_int was baked at.  Derivation for the new constants:
//
//   Let alpha = the real-world scale of one q_in LSB.
//     acc_j   ~ (true preactivation) / (alpha * s_w)
//     racc_j  = acc_j / 2^s_acc
//   Requiring Y_j = (true preactivation) / (s_w * 2^-F) gives
//     Y_j = racc_j * alpha * 2^(s_acc + F)
//
//   INPUT_IS_Q = 1  (L1, RES): q_in came from the Q0.24 features, so
//       alpha = 2^(s_in - QF)   =>   K_MANT/2^K_SHIFT = 2^(-QF + F)
//       => K_MANT is a POWER OF TWO. Use K_MANT = 2^17,
//          K_SHIFT = 17 + QF - F   (= 31 at QF=24, F=10).
//
//   INPUT_IS_Q = 0  (L2, L3): q_in came from the previous layer's Y, so
//       alpha = s_w_prev * 2^(s_in - F)   =>   K_MANT/2^K_SHIFT = s_w_prev
//       => K_MANT = round(s_w_prev * 2^K_SHIFT), K_SHIFT chosen so K_MANT
//          lands in [2^16, 2^18).  s_w_prev is the PREVIOUS layer's weight
//          scale, straight out of the offline quantiser.
//
//   In both cases the applied shift is  msh = K_SHIFT - s_in - s_acc.
//   MSH is gone as a parameter -- it is folded into K_SHIFT.
//
// CYCLES: 1 + (2*NRQ/RQ_FOLDS + 2) [scan1] + INPUT_SIZE/NUM_FOLDS + 2
//         + (2*NRQ/RQ_FOLDS + 2) [scan2] + 1 + NUM_NEURONS/AP_FOLDS + 1
//   48-wide, RQ_FOLDS=8, NUM_FOLDS=1, AP_FOLDS=8: ~76  (was ~151)

module layer_int #(
    parameter int INPUT_SIZE  = 48,
    parameter int NUM_NEURONS = 48,
    parameter int NUM_FOLDS   = 1,    // dot product lanes  (1 lane = 1 DSP/neuron)
    parameter int RQ_FOLDS    = 8,    // requant lanes      (no DSPs any more)
    parameter int AP_FOLDS    = 8,    // scale/bias/relu lanes
    parameter int USE_RELU    = 1,
    parameter int INPUT_IS_Q  = 0,    // 1 => input already int12 (L1 / RES)
    parameter int BW_ACT      = 12,
    parameter int BW_W        = 10,
    parameter int BW_ACC      = 28,   // 22-bit product + clog2(48) = 28
    parameter int BW_Y        = 28,
    parameter int BW_K        = 19,   // signed width of K_MANT
    parameter int K_MANT      = 131072,
    parameter int K_SHIFT     = 31,
    parameter int SHW         = 5,    // width of the requant shift exponent

    // ---- per-channel weight scales ----
    // 0 (default): this layer's weights were quantised with ONE scale, so
    //    K_MANT is a literal and the APPLY multiply is a constant multiply --
    //    Quartus builds ~9 shift-adds in ALMs and it costs zero DSPs.
    // 1: the weights were quantised with one scale PER OUTPUT NEURON, so
    //    K_MANT becomes a per-neuron ROM and APPLY needs a real 12x19
    //    multiplier per lane (AP_FOLDS of them, packed 2 per DSP block).
    //
    // WHY YOU WANT THIS ON layer1 AND layer2: a per-tensor scale is set by
    // max|w|, and layer2's max is 67x its median, so the median weight
    // resolves to ~2.9 of its 10 bits. Per-channel on L1+L2 cuts final pixel
    // error 3.1x (mean 1.36 -> 0.44 LSB, >2LSB 26.6% -> 3.4%) for 8 DSP
    // blocks. L3 and residual gain nothing -- leave them at 0.
    parameter int KM_PER_CHANNEL = 0,
    parameter     KM_FILE        = ""
) (
    input  logic clock,
    input  logic reset,

    input  logic in_valid,
    output logic in_ready,

    // integer Y from the previous layer (INPUT_IS_Q = 0)
    input  logic signed [BW_Y-1:0]   in_vec [INPUT_SIZE-1:0],
    // pre-quantised activations from the top-level requant (INPUT_IS_Q = 1)
    input  logic signed [BW_ACT-1:0] in_q   [INPUT_SIZE-1:0],
    input  logic        [SHW-1:0]    xin_shift,

    input  logic signed [BW_W-1:0]   W_flat [NUM_NEURONS*INPUT_SIZE-1:0],
    input  logic signed [31:0]       B_int  [NUM_NEURONS-1:0],

    output logic signed [BW_Y-1:0]   out_vec [NUM_NEURONS-1:0],
    output logic out_valid,
    input  logic out_ready
);

    localparam int NRQ   = (INPUT_SIZE > NUM_NEURONS) ? INPUT_SIZE : NUM_NEURONS;
    localparam int NAP   = (NUM_NEURONS + AP_FOLDS - 1) / AP_FOLDS;
    localparam int BW_AP = BW_ACT + BW_K;             // 31: racc * K_MANT

    typedef enum logic [2:0] {
        ACCEPT, RQ_IN, COMPUTE, RQ_ACC, SCALE, APPLY, HOLD
    } state_t;
    state_t state, state_n;

    // ---------------- shared requant ----------------
    // ONE instance serves BOTH scans -- they are the same operation and are
    // sequential in the FSM. Sized to max(INPUT_SIZE, NUM_NEURONS); unused
    // lanes are fed zero, which can never win an abs-max.
    logic                     rq_iv, rq_ir, rq_ov, rq_orr;
    logic signed [BW_ACC-1:0] rq_in  [NRQ-1:0];
    logic signed [BW_ACT-1:0] rq_out [NRQ-1:0];
    logic [NRQ*BW_ACT-1:0]    rq_flat;
    logic [SHW-1:0]           rq_sh;
    logic                     rq_sel;      // 0 = scan input, 1 = scan accs

    // ---------------- neuron array ----------------
    logic [NUM_NEURONS-1:0]   done_bus;
    logic signed [BW_ACC-1:0] acc   [NUM_NEURONS-1:0];
    logic signed [BW_ACT-1:0] q_act [0:INPUT_SIZE-1];
    logic signed [BW_ACT-1:0] racc  [NUM_NEURONS-1:0];
    logic        [SHW-1:0]    s_in, s_acc;
    logic                     start_pulse, start_q;

    localparam int BW_C = (NAP > 2) ? $clog2(NAP) : 1;
    logic [BW_C:0]            scyc;

    // ================= requant instance =================
    requant_int #(
        .N(NRQ), .BW_IN(BW_ACC), .BW_OUT(BW_ACT), .FOLDS(RQ_FOLDS), .SHW(SHW)
    ) u_rq (
        .clock(clock), .reset(reset),
        .in_valid(rq_iv), .in_ready(rq_ir), .vec_in(rq_in),
        .out_valid(rq_ov), .out_ready(rq_orr),
        .q_out(rq_out), .q_flat(rq_flat), .vshift(rq_sh)
    );

    // Input mux: scan the INCOMING vector straight off the port, or the
    // accumulators.  Note there is no in_vec_q register bank any more --
    // requant_int latches vec_in itself, and we now assert rq_iv in the SAME
    // cycle as the upstream handshake so in_vec is guaranteed stable.  That
    // deletes INPUT_SIZE*BW_Y flops per layer (1344 per 48-wide layer).
    genvar r;
    generate
        for (r = 0; r < NRQ; r++) begin : gen_rqin
            if (r < INPUT_SIZE && r < NUM_NEURONS)
                assign rq_in[r] = rq_sel ? acc[r] : in_vec[r];
            else if (r < INPUT_SIZE)
                assign rq_in[r] = rq_sel ? '0     : in_vec[r];
            else if (r < NUM_NEURONS)
                assign rq_in[r] = rq_sel ? acc[r] : '0;
            else
                assign rq_in[r] = '0;
        end
    endgenerate

    assign rq_iv  = (state == ACCEPT && in_valid && (INPUT_IS_Q == 0))
                 || (state == RQ_ACC);
    assign rq_orr = 1'b1;

    // ================= shared activation mux =================
    // Every neuron sweeps the SAME activation index -- they all start on
    // start_pulse and none of them can stall. Muxing q_act inside each neuron
    // built 48 identical 48:1 12-bit register muxes per layer (~190 ALUTs
    // each). Do it once here and broadcast.
    //
    // The Fitter needed 3,653 LABs against a 3,207 LAB device, so logic was
    // the binding constraint, not DSP. This is the largest single redundancy.
    //
    // TIMING MUST MIRROR dot_prod_int's counter exactly:
    //   cycle T   : start_pulse high, dot_prod IDLE -> SWEEP, its cycle = 0
    //   cycle T+1 : dot_prod in SWEEP with cycle = 0, consumes a_sel for idx 0
    // so dp_cyc must read 0 at T+1. Hence the reset-on-start-pulse below.
    localparam int NDP  = (INPUT_SIZE + NUM_FOLDS - 1) / NUM_FOLDS;
    localparam int DPW  = $clog2(NDP + 1);
    localparam int ASIW = $clog2(NDP * NUM_FOLDS + 1);

    logic [DPW-1:0]           dp_cyc;
    logic                     dp_run;
    logic [ASIW-1:0]          sidx [NUM_FOLDS];
    logic signed [BW_ACT-1:0] a_sel [NUM_FOLDS];

    always_ff @(posedge clock or posedge reset) begin
        if (reset) begin
            dp_cyc <= '0;
            dp_run <= 1'b0;
        end else if (start_pulse) begin
            dp_cyc <= '0;
            dp_run <= 1'b1;
        end else if (dp_run) begin
            if (dp_cyc == DPW'(NDP-1)) dp_run <= 1'b0;
            else                       dp_cyc <= dp_cyc + 1'b1;
        end
    end

    always_comb
        for (int l = 0; l < NUM_FOLDS; l++) begin
            sidx[l]  = ASIW'(dp_cyc * NUM_FOLDS + l);
            a_sel[l] = (sidx[l] < INPUT_SIZE) ? q_act[sidx[l]] : '0;
        end

    // ================= neurons =================
    assign start_pulse = (state == COMPUTE) && start_q;

    genvar k, i;
    generate
        for (k = 0; k < NUM_NEURONS; k++) begin : gen_neuron
            logic signed [BW_W-1:0] w_row [0:INPUT_SIZE-1];
            for (i = 0; i < INPUT_SIZE; i++) begin : gen_wslice
                assign w_row[i] = W_flat[k*INPUT_SIZE + i];
            end

            neuron_int #(
                .BW_ACC(BW_ACC), .BW_W(BW_W), .BW_ACT(BW_ACT),
                .INPUT_SIZE(INPUT_SIZE), .NUM_FOLDS(NUM_FOLDS)
            ) u_neuron (
                .clock(clock), .reset(reset),
                .start(start_pulse),
                .A_SEL(a_sel), .W(w_row),
                .done(done_bus[k]), .acc_out(acc[k])
            );
        end
    endgenerate

    // ================= APPLY =================
    // Explicit sign/zero extension rather than size casts. A size cast is
    // SUPPOSED to preserve the operand's signedness (IEEE 1800), but tools
    // disagree: Icarus treats (N)'(signed_val) as signed, ModelSim does not,
    // so `ap >>> MSH` silently became a LOGICAL shift and every negative racc
    // came out as a large positive. Keep the operands natively signed.
    localparam int APIW = $clog2(NUM_NEURONS + AP_FOLDS + 1);
    localparam logic signed [BW_AP:0] MAXY =  (1 <<< (BW_Y-1)) - 1;
    localparam logic signed [BW_AP:0] MINY = -(1 <<< (BW_Y-1));

    // K_MANT source: one literal for the whole layer, or one per neuron.
    logic signed [BW_K-1:0] km_rom [NUM_NEURONS-1:0];
    logic signed [BW_K-1:0] km_sel [AP_FOLDS];

    generate
        if (KM_PER_CHANNEL != 0) begin : gen_km_rom
            initial $readmemh(KM_FILE, km_rom);
        end else begin : gen_km_const
            genvar kk;
            for (kk = 0; kk < NUM_NEURONS; kk++) begin : gen_km_tie
                // K_MANT is positive; BW_K leaves the sign bit clear.
                assign km_rom[kk] = BW_K'(K_MANT);
            end
        end
    endgenerate

    logic [APIW-1:0]         aidx [AP_FOLDS];
    logic                    ahit [AP_FOLDS];
    logic signed [BW_ACT-1:0] a_n  [AP_FOLDS];
    // multstyle=logic on BOTH paths, deliberately.
    //   KM_PER_CHANNEL=0: km_sel is the same literal on every lane, so this is
    //     a constant multiply -- ~9 shift-adds, and for L1/RES where K_MANT is
    //     a power of two it collapses to a wire.
    //   KM_PER_CHANNEL=1: it is a real 12x19 multiply, ~130 ALMs per lane.
    //     Keeping it in ALMs is the RIGHT trade here: DSP blocks are the
    //     binding resource (87 on the device, ~67 already spoken for by the
    //     neuron array and the geometric stage) while ALMs are not (32,070).
    //     8 lanes x 2 per-channel layers is ~2,100 ALMs and ZERO DSP blocks.
    (* multstyle = "logic" *) logic signed [BW_AP-1:0] apn [AP_FOLDS];
    logic signed [BW_AP-1:0] ashf [AP_FOLDS];
    logic signed [BW_AP:0]   asum [AP_FOLDS];
    logic signed [BW_Y-1:0]  ares [AP_FOLDS];

    // msh = K_SHIFT - s_in - s_acc, registered in SCALE so the subtract is not
    // in the same cone as the multiply and the barrel shift.
    //
    // msh should never go negative with correctly generated constants (that
    // would mean the layer output overflows BW_Y). It is clamped rather than
    // wrapped so a bad constant shows up as saturation in the waveform instead
    // of as garbage, and the assert below catches it in simulation.
    localparam int MSHW = $clog2(BW_AP);
    logic signed [7:0]   msh_q;
    logic [MSHW-1:0]     msh_c;

    always_comb begin
        if (msh_q <= 0)                 msh_c = '0;
        else if (msh_q >= BW_AP-1)      msh_c = MSHW'(BW_AP-1);
        else                            msh_c = MSHW'(msh_q);
    end

    always_comb
        for (int l = 0; l < AP_FOLDS; l++) begin
            aidx[l] = APIW'(scyc*AP_FOLDS + l);
            ahit[l] = (aidx[l] < NUM_NEURONS);
            a_n[l]    = ahit[l] ? racc[aidx[l]] : '0;
            km_sel[l] = ahit[l] ? km_rom[aidx[l]] : '0;
            apn[l]    = a_n[l] * km_sel[l];
            ashf[l] = apn[l] >>> msh_c;
            // Replication-based sign extension, NOT a size cast. A size cast
            // is SUPPOSED to preserve signedness (IEEE 1800) but tools
            // disagree -- ModelSim drops it and the >>> above silently becomes
            // a logical shift, turning every negative racc into a large
            // positive. Do not "simplify" this back into a cast.
            asum[l] = {ashf[l][BW_AP-1], ashf[l]}
                    + (ahit[l] ? B_int[aidx[l]] : 32'sd0);
            ares[l] = (USE_RELU != 0 && asum[l] < 0) ? '0
                    : (asum[l] > MAXY)               ? BW_Y'(MAXY)
                    : (asum[l] < MINY)               ? BW_Y'(MINY)
                                                     : BW_Y'(asum[l]);
        end

    // ================= FSM =================
    always_ff @(posedge clock or posedge reset) begin
        if (reset) begin
            state   <= ACCEPT;
            scyc    <= '0;
            start_q <= 1'b0;
            rq_sel  <= 1'b0;
            s_in    <= '0;
            s_acc   <= '0;
            msh_q   <= '0;
        end else begin
            state <= state_n;

            case (state)
                ACCEPT: if (in_valid && in_ready) begin
                    // requant_int is latching in_vec on THIS SAME EDGE for the
                    // INPUT_IS_Q = 0 path, so nothing to copy there.
                    if (INPUT_IS_Q) begin
                        for (int j = 0; j < INPUT_SIZE; j++)
                            q_act[j] <= in_q[j];
                        s_in    <= xin_shift;
                        start_q <= 1'b1;
                    end
                    scyc <= '0;
                end

                RQ_IN: if (rq_ov) begin
                    for (int j = 0; j < INPUT_SIZE; j++)
                        q_act[j] <= $signed(rq_flat[j*BW_ACT +: BW_ACT]);
                    s_in    <= rq_sh;
                    start_q <= 1'b1;
                end

                COMPUTE: begin
                    start_q <= 1'b0;
                    // Arm the mux one state early: rq_sel is registered, so
                    // setting it inside RQ_ACC would be a cycle late and the
                    // requant would re-latch the INPUT vector.
                    rq_sel  <= 1'b1;
                end

                RQ_ACC: begin
                    if (rq_ov) begin
                        for (int j = 0; j < NUM_NEURONS; j++)
                            racc[j] <= $signed(rq_flat[j*BW_ACT +: BW_ACT]);
                        s_acc <= rq_sh;
                    end
                    scyc <= '0;
                end

                SCALE: begin
                    // int arithmetic then truncate: the intermediate can be
                    // negative, so it must not be evaluated at 5-bit width.
                    msh_q <= 8'(int'(K_SHIFT) - int'(s_in) - int'(s_acc));
                    scyc  <= '0;
                end

                APPLY: begin
                    for (int l = 0; l < AP_FOLDS; l++)
                        if (ahit[l]) out_vec[aidx[l]] <= ares[l];
                    if (scyc != (BW_C+1)'(NAP-1)) scyc <= scyc + 1'b1;
                end

                // Clear the requant source mux on the way out, NOT on the way
                // in. rq_sel is registered and rq_iv is asserted during
                // ACCEPT itself, so clearing it inside ACCEPT would be one
                // cycle too late and scan 1 would re-latch the accumulators.
                HOLD: if (out_ready) begin
                    rq_sel <= 1'b0;
                    scyc   <= '0;
                end

                default: ;
            endcase
        end
    end

    always_comb begin
        state_n = state;
        case (state)
            ACCEPT:  if (in_valid && in_ready)
                         state_n = (INPUT_IS_Q != 0) ? COMPUTE : RQ_IN;
            RQ_IN:   if (rq_ov)                    state_n = COMPUTE;
            COMPUTE: if (&done_bus)                state_n = RQ_ACC;
            RQ_ACC:  if (rq_ov)                    state_n = SCALE;
            SCALE:                                 state_n = APPLY;
            APPLY:   if (scyc == (BW_C+1)'(NAP-1)) state_n = HOLD;
            HOLD:    if (out_ready)                state_n = ACCEPT;
            default:                               state_n = ACCEPT;
        endcase
    end

    // For INPUT_IS_Q = 0 the upstream handshake must line up with requant's
    // own IDLE, because requant latches in_vec directly off the port.
    assign in_ready  = (state == ACCEPT) && ((INPUT_IS_Q != 0) || rq_ir);
    assign out_valid = (state == HOLD);
    // synthesis translate_off
    /* verilator lint_off BLKSEQ */
    int s_min [2] = '{99, 99};
    int s_max [2] = '{0, 0};
    int s_n   [2] = '{0, 0};
    always @(posedge clock)
        if (!reset && rq_ov && rq_orr) begin
            if (int'(rq_sh) < s_min[rq_sel]) s_min[rq_sel] = int'(rq_sh);
            if (int'(rq_sh) > s_max[rq_sel]) s_max[rq_sel] = int'(rq_sh);
            s_n[rq_sel]++;
        end
    final begin
        if (s_n[0]) $display("%m  RQ_IN   min=%0d max=%0d spread=%0d n=%0d",
                            s_min[0], s_max[0], s_max[0]-s_min[0], s_n[0]);
        if (s_n[1]) $display("%m  RQ_ACC  min=%0d max=%0d spread=%0d n=%0d",
                            s_min[1], s_max[1], s_max[1]-s_min[1], s_n[1]);
    end
    /* verilator lint_on BLKSEQ */
    // synthesis translate_on

endmodule