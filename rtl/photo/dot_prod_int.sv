// dot_prod_int.sv
//
// Folded integer dot product for the quantized MLP.
//   acc = sum_i ( ACT_SEL[i] * W_Q[i] )   -- both SIGNED
//
// ===========================================================================
// WHY ACT_SEL IS A SCALAR PER LANE AND NOT THE WHOLE VECTOR
//
// This used to take the full ACT_Q[INPUT_SIZE] array and do
//     prod = ACT_Q[idx] * W_Q[idx]
// with idx from this neuron's own counter. That built a 48:1 mux over 12-bit
// REGISTERS inside every neuron -- roughly 190 ALUTs each, times 48 neurons
// per layer, for a value that is IDENTICAL in all of them.
//
// Every neuron in a layer is started by the same start_pulse and has no stall
// path, so all their counters hold the same value on every cycle. The mux is
// therefore computed ONCE in layer_int and broadcast here.
//
// Motivation, measured: the Fitter needed 3,653 LABs against a 3,207 LAB
// device -- LOGIC, not DSP, was the binding constraint. This is the single
// largest redundancy in the neuron array.
//
// W_Q stays a full array on purpose. Its mux is over CONSTANTS with a 6-bit
// select, which Quartus builds as one 6-input LUT per output bit -- about 10
// ALUTs per neuron, versus ~190 for the register mux. Not worth hoisting, and
// it could not be shared anyway since every neuron has a different row.
//
// ---------------------------------------------------------------------------
// LOCKSTEP REQUIREMENT -- READ THIS BEFORE ADDING ANY STALL
//
// The caller must present ACT_SEL for the SAME cycle index this module is
// using internally. That holds today because layer_int runs one counter for
// the whole array and every neuron starts on the same pulse. If you add
// per-neuron stalling, back-pressure, or a variable-latency weight fetch,
// this breaks SILENTLY -- products pair the wrong activation with the wrong
// weight, and the layer output stays plausible while being wrong. If that day
// comes, drive the index from layer_int explicitly instead.
// ---------------------------------------------------------------------------
//
// ACCUMULATOR SIZING: worst-case product is BW_ACT + BW_W bits (22 for 12x10).
// Summing INPUT_SIZE of them needs + clog2(INPUT_SIZE) more bits.
//   48 inputs: 22 + 6 = 28 bits minimum.
// BW_ACC is 28: exactly that.

module dot_prod_int #(
    parameter int INPUT_SIZE = 29,
    parameter int NUM_FOLDS  = 1,
    parameter int BW_W       = 10,
    parameter int BW_ACT     = 12,
    parameter int BW_ACC     = 28
)(
    input  logic clock,
    input  logic reset,
    input  logic start,                                   // pulse to begin sweep
    // Pre-muxed activations for the current cycle, one per lane, supplied by
    // layer_int and shared across every neuron in the layer.
    input  logic signed [BW_ACT-1:0] ACT_SEL [NUM_FOLDS],
    input  logic signed [BW_W-1:0]   W_Q   [INPUT_SIZE],  // SIGNED weights
    output logic signed [BW_ACC-1:0] dot_prod,            // SIGNED accumulator
    output logic done
);

           localparam int NUM_CYCLES = (INPUT_SIZE + NUM_FOLDS - 1) / NUM_FOLDS;
    localparam int BW_PROD    = BW_ACT + BW_W;
    localparam int NLANE      = ((NUM_FOLDS + 1) / 2) * 2;   // round up to even
    localparam int NPAIR      = NLANE / 2;                   // always >= 1
    localparam int NPAD       = NUM_CYCLES * NUM_FOLDS + NLANE;

    typedef enum logic [1:0] {IDLE, SWEEP, FINISH} state_t;
    state_t state;
    logic [$clog2(NUM_CYCLES+2)-1:0] cycle;
    logic signed [BW_ACC-1:0] acc, partial;

    // Zero-pad the WEIGHTS past INPUT_SIZE so the sweep never needs an
    // `idx < INPUT_SIZE` guard -- that guard is what blocks sum-of-two-18x18
    // inference. Sized for the extra drain cycle and the odd pad lane.
    logic signed [BW_W-1:0]   w_pad   [NPAD];
    // Zero-pad the LANES to an even count so NPAIR is never 0 when NUM_FOLDS
    // is odd. act_pad[NUM_FOLDS] is a hard zero, so its product folds away.
    logic signed [BW_ACT-1:0] act_pad [NLANE];

    genvar g;
    generate
        for (g = 0; g < NPAD; g++) begin : gen_wpad
            if (g < INPUT_SIZE) assign w_pad[g] = W_Q[g];
            else                assign w_pad[g] = '0;
        end
        for (g = 0; g < NLANE; g++) begin : gen_apad
            if (g < NUM_FOLDS) assign act_pad[g] = ACT_SEL[g];
            else               assign act_pad[g] = '0;
        end
    endgenerate

    // ONE Cyclone V DSP block per pair: Sum of two 18x18 + output register.
    // Do not add a conditional or a width cast inside this expression.
    (* multstyle = "dsp" *) logic signed [BW_PROD:0] pair_r [NPAIR];

    always_ff @(posedge clock)
        for (int p = 0; p < NPAIR; p++)
            pair_r[p] <= (act_pad[2*p]   * w_pad[cycle*NUM_FOLDS + 2*p])
                       + (act_pad[2*p+1] * w_pad[cycle*NUM_FOLDS + 2*p+1]);

    // Widen with a plain signed-to-wider-signed assignment, NOT `BW_ACC'(...)`.
    // A size cast on a signed value is the exact trap layer_int_v2's header
    // warns about -- ModelSim drops the signedness and zero-extends, so every
    // negative partial product becomes a large positive and the layer output
    // stays plausible while being wrong.
    logic signed [BW_ACC-1:0] pair_x [NPAIR];
    generate
        for (g = 0; g < NPAIR; g++) begin : gen_pairx
            assign pair_x[g] = pair_r[g];
        end
    endgenerate

    always_comb begin
        partial = '0;
        for (int p = 0; p < NPAIR; p++) partial = partial + pair_x[p];
    end

    // ---- FSM ----
    always_ff @(posedge clock or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            acc   <= '0;
            cycle <= '0;
            done  <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        acc   <= '0;
                        cycle <= '0;
                        state <= SWEEP;
                    end
                end

                SWEEP: begin
                    if (cycle != '0) acc <= acc + partial;   // pair_r valid from cycle 1
                    if (cycle == NUM_CYCLES) state <= FINISH;
                    else                     cycle <= cycle + 1'b1;
                end

                FINISH: begin
                    done  <= 1'b1;    // one-cycle pulse; consumer must latch on it
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

    assign dot_prod = acc;

endmodule