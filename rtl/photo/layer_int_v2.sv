
module layer_int_v2 #(
    parameter int INPUT_SIZE  = 48,
    parameter int NUM_NEURONS = 48,
    parameter int NUM_FOLDS   = 1,    // dot product lanes  (1 lane = 1 DSP/neuron)
    parameter int RQ_FOLDS    = 8,    // UNUSED -- kept so existing instantiations
                                      // that pass .RQ_FOLDS still elaborate.
    parameter int DYN_ACC     = 0,    // UNUSED AND IGNORED -- kept so existing
                                      // instantiations that pass .DYN_ACC(0)
                                      // still elaborate. The dynamic
                                      // accumulator scan has been deleted.
                                      // Passing 1 does NOT re-enable it.
    parameter int AP_FOLDS    = 8,    // scale/bias/relu lanes
    parameter int USE_RELU    = 1,
    parameter int INPUT_IS_Q  = 0,    // 1 => input already int12 (L1 / RES)
    parameter int BW_ACT      = 12,
    parameter int BW_W        = 10,
    parameter int BW_ACC      = 28,   // 22-bit product + clog2(48) = 28
    parameter int BW_Y        = 28,
    parameter int BW_BINT     = 32,   // width of the B_int port
    parameter int BW_K        = 19,   // signed width of K_MANT
    parameter int K_MANT      = 131072,
    parameter int K_SHIFT     = 31,
    parameter int SHW         = 5,    // width of the (now unread) xin_shift port

    // ---- static requant shifts ----
    // S_IN  : fixed shift on the incoming vector. IGNORED when INPUT_IS_Q=1.
    // S_ACC : fixed shift on the accumulators.
    // Both are already folded out of K_SHIFT by the caller. Err HIGH if you
    // are unsure: too high loses precision but msh compensates and the scale
    // stays correct; too low CLIPS.
    parameter int S_IN        = 11,
    parameter int S_ACC       = 11,

    // ---- static value of what used to arrive on the xin_shift port ----
    // See note 2 in the header. MUST be set to 13 (== XSHIFT) on u_l1 and
    // u_res. Leave at 0 for INPUT_IS_Q=0 layers.
    parameter int XIN_SHIFT   = 0,

    // ---- per-channel weight scales ----
    // 0 (default): one scale for the layer, K_MANT is a literal.
    // 1: one scale PER OUTPUT NEURON, K_MANT becomes a per-neuron ROM.
    //    Cuts final pixel error 3.1x on L1+L2. L3 and residual gain nothing.
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
    // DEAD PORT. Kept so callers do not have to change their connections.
    // The value is taken from the XIN_SHIFT PARAMETER instead -- see header.
    /* verilator lint_off UNUSED */
    input  logic        [SHW-1:0]    xin_shift,
    /* verilator lint_on UNUSED */

    input  logic signed [BW_W-1:0]     W_flat [NUM_NEURONS*INPUT_SIZE-1:0],
    input  logic signed [BW_BINT-1:0]  B_int  [NUM_NEURONS-1:0],

    output logic signed [BW_Y-1:0]   out_vec [NUM_NEURONS-1:0],
    output logic out_valid,
    input  logic out_ready
);

    localparam int NAP   = (NUM_NEURONS + AP_FOLDS - 1) / AP_FOLDS;
    localparam int BW_AP = BW_ACT + BW_K;             // 31: racc * K_MANT
    localparam int QMAX  = (1 << (BW_ACT-1)) - 1;     // 2047

    // RQ_ACC is gone with DYN_ACC.
    typedef enum logic [2:0] { ACCEPT, COMPUTE, APPLY, AP_DRAIN, HOLD } state_t;
    state_t state, state_n;

    // ---------------- neuron array / pipeline registers ----------------
    logic [NUM_NEURONS-1:0]   done_bus;
    logic signed [BW_ACC-1:0] acc   [NUM_NEURONS-1:0];
    logic signed [BW_ACT-1:0] q_act [0:INPUT_SIZE-1];
    logic signed [BW_ACT-1:0] racc  [NUM_NEURONS-1:0];
    logic                     start_pulse, start_q;

    localparam int BW_C = (NAP > 2) ? $clog2(NAP) : 1;
    logic [BW_C:0]            scyc;

    // ================= static requant =================
    // Fixed-shift requant. `s` is always a parameter, so `>> s` is a bit slice,
    // `+ half` is a constant add, and the saturate is one compare. No barrel
    // shifter, no FSM, no cycles, no DSPs.
    //
    // Rounding is kept (half away from zero on the magnitude) so the transfer
    // stays symmetric about zero -- an arithmetic >>> on the signed value
    // biases every negative element one LSB toward -inf and shows up as a DC
    // shift in the layer output.
    function automatic logic signed [BW_ACT-1:0]
             rq_s (input logic signed [BW_ACC-1:0] v, input int s);
        logic [BW_ACC-1:0] mag, res;
        logic [BW_ACT-1:0] sat;
        mag = v[BW_ACC-1] ? (~v + 1'b1) : v;
        res = (mag + ((s == 0) ? '0 : (BW_ACC'(1) << (s-1)))) >> s;
        sat = (res > BW_ACC'(QMAX)) ? BW_ACT'(QMAX) : BW_ACT'(res);
        return v[BW_ACC-1] ? -$signed(sat) : $signed(sat);
    endfunction

    // ---- calibration probe, simulation only ----
    function automatic int rq_need (input logic signed [BW_ACC-1:0] v);
        logic [BW_ACC-1:0] mag;
        int s;
        mag = v[BW_ACC-1] ? (~v + 1'b1) : v;
        s   = 0;
        for (int b = BW_ACT-1; b < BW_ACC; b++)
            if (mag[b]) s = b - (BW_ACT-2);   // last-wins = highest set bit
        return s;
    endfunction

    // Sign-extend in_vec (BW_Y) to the function's BW_ACC width. Replication,
    // not a size cast -- see the APPLY comment below for why casts are unsafe.
    logic signed [BW_ACC-1:0] in_ext [INPUT_SIZE-1:0];
    genvar e;
    generate
        for (e = 0; e < INPUT_SIZE; e++) begin : gen_inext
            if (BW_ACC > BW_Y)
                assign in_ext[e] = {{(BW_ACC-BW_Y){in_vec[e][BW_Y-1]}}, in_vec[e]};
            else
                assign in_ext[e] = in_vec[e][BW_ACC-1:0];
        end
    endgenerate

    // ================= shared activation mux =================
    // Every neuron sweeps the SAME activation index -- they all start on
    // start_pulse and none of them can stall. Muxing q_act inside each neuron
    // built 48 identical 48:1 12-bit register muxes per layer. Do it once here
    // and broadcast.
    //
    // TIMING MUST MIRROR dot_prod_int's counter exactly:
    //   cycle T   : start_pulse high, dot_prod IDLE -> SWEEP, its cycle = 0
    //   cycle T+1 : dot_prod in SWEEP with cycle = 0, consumes a_sel for idx 0
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

    // NOTE: the requant_int instance, the ra_iv/ra_ir/ra_ov/ra_q/ra_flat/ra_sh
    // signals and the whole DYN_ACC generate block used to live here. Deleted.

    // ================= APPLY =================
    // Explicit sign/zero extension rather than size casts. A size cast is
    // SUPPOSED to preserve the operand's signedness (IEEE 1800), but tools
    // disagree: Icarus treats (N)'(signed_val) as signed, ModelSim does not,
    // so `ap >>> MSH` silently became a LOGICAL shift and every negative racc
    // came out as a large positive. Keep the operands natively signed and use
    // plain signed-to-wider-signed assignment, which sign-extends everywhere.
    localparam int APIW = $clog2(NUM_NEURONS + AP_FOLDS + 1);
    localparam logic signed [BW_AP:0] MAXY =  (1 <<< (BW_Y-1)) - 1;
    localparam logic signed [BW_AP:0] MINY = -(1 <<< (BW_Y-1));

    // ---- THE SHIFT IS A CONSTANT. There is no barrel shifter. ----
    // MSH_I is resolved at elaboration, so `apn >>> MSH_C` is a fixed bit
    // slice and costs zero logic. Do NOT turn MSH_C back into a register or
    // derive it from the xin_shift port -- that is what cost ~2.5 ns and a
    // 5-level 31-bit shifter on every one of the AP_FOLDS lanes.
    localparam int MSHW  = $clog2(BW_AP);
    localparam int MSH_I = K_SHIFT - XIN_SHIFT;
    localparam logic [MSHW-1:0] MSH_C = (MSH_I < 0)        ? '0
                                      : (MSH_I > BW_AP-1)  ? MSHW'(BW_AP-1)
                                                           : MSHW'(MSH_I);

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

    logic [APIW-1:0]          aidx [AP_FOLDS];
    logic                     ahit [AP_FOLDS];
    logic signed [BW_ACT-1:0] a_n  [AP_FOLDS];

    // NO multstyle attribute, deliberately -- see note 3 in the header. This
    // multiply lands in a DSP block, which is what you want while APPLY is a
    // single combinational cycle.
    logic signed [BW_AP-1:0] apn  [AP_FOLDS];
    logic signed [BW_AP-1:0] ashf [AP_FOLDS];
    logic signed [BW_AP:0]   ashf_x [AP_FOLDS];   // ashf widened by one
    logic signed [BW_AP:0]   b_sel  [AP_FOLDS];   // B_int widened to match
    logic signed [BW_AP:0]   asum [AP_FOLDS];
    logic signed [BW_Y-1:0]  ares [AP_FOLDS];

    always_comb
        for (int l = 0; l < AP_FOLDS; l++) begin
            aidx[l]   = APIW'(scyc*AP_FOLDS + l);
            ahit[l]   = (aidx[l] < NUM_NEURONS);
            a_n[l]    = ahit[l] ? racc[aidx[l]]   : '0;
            km_sel[l] = ahit[l] ? km_rom[aidx[l]] : '0;

            apn[l]    = a_n[l] * km_sel[l];       // 12 x 19 signed -> DSP
            ashf[l]   = apn[l] >>> MSH_C;         // CONSTANT shift -> wires

            // Plain signed-to-wider-signed assignments. These sign-extend on
            // every tool. Do NOT "simplify" them into size casts or a
            // {sign, value} concatenation -- a concat is UNSIGNED, which makes
            // the whole addition unsigned and zero-extends a negative B_int
            // the moment BW_BINT < BW_AP+1.
            ashf_x[l] = ashf[l];
            b_sel[l]  = ahit[l] ? B_int[aidx[l]] : '0;
            asum[l]   = ashf_x[l] + b_sel[l];

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
        end else begin
            state <= state_n;

            case (state)
                ACCEPT: if (in_valid && in_ready) begin
                    // The RQ_IN pass is now this one line. in_vec is stable
                    // during the handshake cycle, so it is sliced straight off
                    // the port into q_act -- no scan, no extra state.
                    for (int j = 0; j < INPUT_SIZE; j++)
                        q_act[j] <= (INPUT_IS_Q != 0) ? in_q[j]
                                                      : rq_s(in_ext[j], S_IN);
                    // msh_q used to be loaded here. It is now the MSH_C
                    // localparam. Nothing to load.
                    start_q <= 1'b1;
                    scyc    <= '0;
                end

                COMPUTE: begin
                    start_q <= 1'b0;
                    if (&done_bus)
                        for (int j = 0; j < NUM_NEURONS; j++)
                            racc[j] <= rq_s(acc[j], S_ACC);
                    scyc <= '0;
                end

                APPLY: begin
                    for (int l = 0; l < AP_FOLDS; l++)
                        if (ahit[l]) out_vec[aidx[l]] <= ares[l];
                    if (scyc != (BW_C+1)'(NAP-1)) scyc <= scyc + 1'b1;
                end

                HOLD: if (out_ready) scyc <= '0;

                default: ;
            endcase
        end
    end

    always_comb begin
        state_n = state;
        case (state)
            ACCEPT:  if (in_valid && in_ready)     state_n = COMPUTE;
            COMPUTE: if (&done_bus)                state_n = APPLY;
            APPLY:   if (scyc == (BW_C+1)'(NAP-1)) state_n = HOLD;
            HOLD:    if (out_ready)                state_n = ACCEPT;
            default:                               state_n = ACCEPT;
        endcase
    end

    // No requant handshake to line up with any more -- ACCEPT is unconditional.
    assign in_ready  = (state == ACCEPT);
    assign out_valid = (state == HOLD);

    // ================= sanity =================
    // msh is a constant now, so this is a ONE-TIME check at time 0 rather than
    // a per-cycle assert. msh < 0 means K_SHIFT - XIN_SHIFT went negative: the
    // layer output overflows BW_Y, MSH_C clamps to 0, and the image comes out
    // systematically too bright with every handshake assertion still passing.
    //
    // The $display is not optional noise -- it is how you verify you passed
    // .XIN_SHIFT correctly. Expect, in the sim log:
    //     u_l1  MSH=7    u_l2  MSH=10    u_l3  MSH=8    u_res MSH=6
    // synthesis translate_off
    initial begin
        $display("%m  MSH=%0d  (K_SHIFT=%0d XIN_SHIFT=%0d S_IN=%0d S_ACC=%0d)",
                 MSH_I, K_SHIFT, XIN_SHIFT, S_IN, S_ACC);
        if (MSH_I < 0)
            $error("%m: msh = %0d < 0 -- K_SHIFT/XIN_SHIFT mismatch", MSH_I);
    end
    // synthesis translate_on

// synthesis translate_off
    /* verilator lint_off BLKSEQ */
    int clamp_in = 0, clamp_acc = 0;
    int hi_in = 0, lo_in = 99, hi_acc = 0, lo_acc = 99;
    int n_in = 0, n_acc = 0;
    int v_need;

    always @(posedge clock) begin
        if (!reset && state == ACCEPT && in_valid && in_ready && INPUT_IS_Q == 0) begin
            v_need = 0;
            for (int j = 0; j < INPUT_SIZE; j++)
                if (rq_need(in_ext[j]) > v_need) v_need = rq_need(in_ext[j]);
            if (v_need > hi_in) hi_in = v_need;
            if (v_need < lo_in) lo_in = v_need;
            if (v_need > S_IN)  clamp_in++;
            n_in++;
        end
        if (!reset && state == COMPUTE && (&done_bus)) begin
            v_need = 0;
            for (int j = 0; j < NUM_NEURONS; j++)
                if (rq_need(acc[j]) > v_need) v_need = rq_need(acc[j]);
            if (v_need > hi_acc) hi_acc = v_need;
            if (v_need < lo_acc) lo_acc = v_need;
            if (v_need > S_ACC)  clamp_acc++;
            n_acc++;
        end
    end

    final begin
        if (INPUT_IS_Q == 0)
            $display("%m  RQ_IN   S_IN=%0d  need %0d..%0d (spread %0d)  clamps=%0d",
                     S_IN, lo_in, hi_in, hi_in-lo_in, clamp_in);
        $display("%m  RQ_ACC  S_ACC=%0d need %0d..%0d (spread %0d)  clamps=%0d",
                 S_ACC, lo_acc, hi_acc, hi_acc-lo_acc, clamp_acc);
    end
    /* verilator lint_on BLKSEQ */
    // synthesis translate_on

endmodule