// requant_int.sv  -- SHIFT-NORMALISING VERSION
//
// q[i] = sat( round( v[i] / 2^s ) ),   s = max(0, msb(max|v|) - (BW_OUT-1))
//
// WHAT CHANGED AND WHY
//   The old version computed q[i] = round(v[i] * QMAX / max|v|), which needed
//   a reciprocal (18-cycle restoring divide) and then FOLDS parallel 32x18
//   multiplies.  On Cyclone V that cost ~4 DSP multipliers per instance, and
//   there are FIVE instances -> ~20 of the 87 DSP blocks on the DE1-SoC, plus
//   18 dead cycles per scan and two scans per layer.
//
//   Normalising by a POWER OF TWO instead of by the exact max costs at most
//   1 bit of the 12-bit output range (the result lands in [2^10, 2^11) rather
//   than filling [0, 2047] exactly) and buys:
//     - zero DSP multipliers
//     - no divider: NCYC + 1 + NCYC + 1 cycles instead of 2*NCYC + 20
//     - the residual scale factor is 2^s, so the DOWNSTREAM scale correction
//       becomes a shift too.  That is what lets layer_int delete its entire
//       M chain (a 32x32 and a 24x18 multiply per layer).  See layer_int.sv.
//
//   Net accuracy cost: up to 1 bit of activation SNR (~6 dB).  Net saving
//   across the design: ~48 of 146 DSP blocks and ~60 cycles per pixel.
//   If you need the bit back, raise BW_OUT to 13 -- the multipliers stay gone.
//
// PORT CHANGE
//   `vmax` (the integer abs-max) is replaced by `vshift` (the exponent s).
//   Consumers want the scale, and the scale is now exactly 2^-s.
//
// CYCLE COST: ceil(N/FOLDS) scan + 1 norm + ceil(N/FOLDS) shift + 1 done
//   N=48, FOLDS=8 -> 6 + 1 + 6 + 1 = 14  (was 43)

module requant_int #(
    parameter int N      = 48,   // vector length
    parameter int BW_IN  = 32,   // input width (signed)
    parameter int BW_OUT = 12,   // output width (signed)
    parameter int FOLDS  = 8,    // elements processed per cycle in scan/shift
    // Derived -- do not override. Declared here rather than as a localparam
    // because it appears in the port list, and a body localparam is not
    // visible there.
    parameter int SHW    = $clog2(BW_IN)
) (
    input  logic clock,
    input  logic reset,

    // ---- input handshake ----
    input  logic in_valid,
    output logic in_ready,
    input  logic signed [BW_IN-1:0] vec_in [N-1:0],

    // ---- output handshake ----
    output logic out_valid,
    input  logic out_ready,
    output logic signed [BW_OUT-1:0] q_out [N-1:0],
    // Packed mirror of q_out. Prefer this for module-to-module connections:
    // unpacked-array OUTPUT ports are a portability minefield (Icarus drops
    // them entirely; some synthesis front ends mis-handle them). Packed buses
    // always work and synthesise identically.
    output logic [N*BW_OUT-1:0]      q_flat,

    // Normalisation exponent: the vector this block just emitted represents
    // the input vector divided by 2^vshift.  Zero input -> vshift = 0.
    output logic [SHW-1:0]           vshift
);

    localparam int QMAX = (1 << (BW_OUT-1)) - 1;          // 2047
    localparam int NCYC = (N + FOLDS - 1) / FOLDS;

    typedef enum logic [2:0] {IDLE, SCAN, NORM, SHIFT, DONE} state_t;
    state_t state, state_n;

    logic signed [BW_IN-1:0] vec_reg [N-1:0];
    logic        [BW_IN-1:0] max_reg;
    logic        [SHW-1:0]   s_reg;
    logic        [$clog2(NCYC+1)-1:0] fcyc;

    // ---------------- absolute value ----------------
    function automatic logic [BW_IN-1:0] absv(input logic signed [BW_IN-1:0] v);
        return v[BW_IN-1] ? (~v + 1'b1) : v;
    endfunction

    localparam int IDXW = $clog2(NCYC*FOLDS + 1);

    logic [IDXW-1:0] lidx [FOLDS];
    logic            lhit [FOLDS];

    always_comb
        for (int l = 0; l < FOLDS; l++) begin
            lidx[l] = IDXW'(fcyc*FOLDS + l);
            lhit[l] = (lidx[l] < N);          // guards N not divisible by FOLDS
        end

    // ---------------- folded abs-max ----------------
    logic [BW_IN-1:0] scan_max;
    logic [BW_IN-1:0] s_abs [FOLDS];

    always_comb begin
        for (int l = 0; l < FOLDS; l++)
            s_abs[l] = lhit[l] ? absv(vec_reg[lidx[l]]) : '0;
        scan_max = max_reg;
        for (int l = 0; l < FOLDS; l++)
            if (s_abs[l] > scan_max) scan_max = s_abs[l];
    end

    // ---------------- SHIFT stage ----------------
    // Round half away from zero on the magnitude, then re-apply the sign.
    // Doing it on the magnitude keeps the transfer symmetric about zero; an
    // arithmetic >>> on the signed value would bias every negative element
    // one LSB toward -inf and show up as a DC shift in the layer output.
    //
    // NO MULTIPLIER HERE. This is a variable right shift by s_reg (0..20),
    // which Quartus builds as a ~5-stage barrel out of ALMs.  FOLDS trades
    // ALMs for cycles: each lane is one barrel shifter, roughly 100 ALMs.
    logic signed [BW_IN-1:0] sh_v    [FOLDS];
    logic        [BW_IN-1:0] sh_mag  [FOLDS];
    logic        [BW_IN-1:0] sh_add  [FOLDS];
    logic        [BW_IN-1:0] sh_res  [FOLDS];
    logic        [BW_OUT-1:0] sh_sat [FOLDS];
    logic signed [BW_OUT-1:0] sh_out [FOLDS];

    logic [BW_IN-1:0] half;
    assign half = (s_reg == '0) ? '0 : (BW_IN'(1) << (s_reg - 1'b1));

    always_comb
        for (int l = 0; l < FOLDS; l++) begin
            sh_v[l]   = lhit[l] ? vec_reg[lidx[l]] : '0;
            sh_mag[l] = absv(sh_v[l]);
            sh_add[l] = sh_mag[l] + half;
            sh_res[l] = sh_add[l] >> s_reg;
            // s is chosen so this lands in [0, 2^(BW_OUT-1)]; the == 2^(BW_OUT-1)
            // case (and the rounding carry above it) is the only one that
            // saturates, hence the compare against QMAX rather than a mask.
            sh_sat[l] = (sh_res[l] > BW_IN'(QMAX)) ? BW_OUT'(QMAX)
                                                   : BW_OUT'(sh_res[l]);
            sh_out[l] = sh_v[l][BW_IN-1] ? -$signed(sh_sat[l])
                                         :  $signed(sh_sat[l]);
        end

    // ---------------- FSM ----------------
    always_ff @(posedge clock or posedge reset) begin
        if (reset) begin
            state   <= IDLE;
            max_reg <= '0;
            fcyc    <= '0;
            s_reg   <= '0;
        end else begin
            state <= state_n;

            case (state)
                IDLE: if (in_valid && in_ready) begin
                    for (int i = 0; i < N; i++) vec_reg[i] <= vec_in[i];
                    max_reg <= '0;
                    fcyc    <= '0;
                end

                SCAN: begin
                    max_reg <= scan_max;
                    if (fcyc != NCYC-1) fcyc <= fcyc + 1'b1;
                    else                fcyc <= '0;
                end

                NORM: begin
                    // s = max(0, msb(max_reg) - (BW_OUT-2)).
                    //
                    // BW_OUT-2, NOT BW_OUT-1. The result must fit in BW_OUT-1
                    // magnitude bits (2047 for BW_OUT=12). max_reg < 2^(msb+1),
                    // so we need msb+1-s <= BW_OUT-1, i.e. s >= msb-(BW_OUT-2).
                    // Getting this off by one lets the shifted value reach 4095
                    // and every large element saturates at 2047 -- which shows
                    // up as a ~2x amplitude error on the layer output, not as
                    // an obvious failure. Verified in check_scale_scheme.py.
                    //
                    // Last-assignment-wins in the loop gives a priority encode
                    // on the HIGHEST set bit, which is what we want.
                    s_reg <= '0;
                    for (int b = BW_OUT-1; b < BW_IN; b++)
                        if (max_reg[b]) s_reg <= SHW'(b - (BW_OUT-2));
                    fcyc <= '0;
                end

                SHIFT: begin
                    for (int l = 0; l < FOLDS; l++)
                        if (lhit[l]) q_out[lidx[l]] <= sh_out[l];
                    if (fcyc != NCYC-1) fcyc <= fcyc + 1'b1;
                end

                default: ;
            endcase
        end
    end

    always_comb begin
        state_n = state;
        case (state)
            IDLE:  if (in_valid && in_ready)   state_n = SCAN;
            SCAN:  if (fcyc == NCYC-1)         state_n = NORM;
            NORM:                              state_n = SHIFT;
            SHIFT: if (fcyc == NCYC-1)         state_n = DONE;
            DONE:  if (out_valid && out_ready) state_n = IDLE;
            default: state_n = IDLE;
        endcase
    end

    assign in_ready  = (state == IDLE);
    assign out_valid = (state == DONE);
    assign vshift    = s_reg;

    genvar gf;
    generate
        for (gf = 0; gf < N; gf++) begin : gen_qflat
            assign q_flat[gf*BW_OUT +: BW_OUT] = q_out[gf];
        end
    endgenerate

    // q_out is driven PROCEDURALLY from the SHIFT stage rather than through a
    // continuous assign from an internal q_reg. Driving an unpacked array
    // port with `assign` is a portability trap -- some tools silently fail to
    // propagate it. Registering the port directly is always safe.

endmodule