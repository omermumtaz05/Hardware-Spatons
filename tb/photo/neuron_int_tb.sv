// tb_neuron_int.sv
//
// Self-checking testbench for neuron_int.
//
// Because the datapath is now pure integer, the golden model is an EXACT
// integer sum -- no float tolerance, no epsilon. Any mismatch is a real bug.
//
// Coverage:
//   1. all zeros
//   2. all +1 x +1                      -> acc == INPUT_SIZE
//   3. saturation corners (most-negative activation x most-negative weight)
//   4. alternating signs                -> catches sign-extension bugs in the
//                                          fold adder tree
//   5. one-hot                          -> catches lane-indexing bugs
//   6. randomised sweep                 -> N_RAND vectors
//   7. back-to-back starts              -> no state leaks between pixels
//   8. done is a single-cycle pulse
//
// The INPUT_SIZE=29 / NUM_FOLDS=4 case is the important one: 29 is NOT
// divisible by 4, so the final fold cycle is partial and the disabled lanes
// must contribute zero. That is exactly where a fold-indexing bug hides.
//
// Run (Icarus):
//   iverilog -g2012 -o tb tb_neuron_int.sv rtl/neuron_int.sv rtl/dot_prod_int.sv
//   ./tb
//
// Run (ModelSim ASE):
//   vlog -sv rtl/dot_prod_int.sv rtl/neuron_int.sv tb_neuron_int.sv
//   vsim -c tb_neuron_int -do "run -all; quit"

`timescale 1ns/1ps

module neuron_int_tb();

    // ---- geometry under test; override with -P or edit ----
    parameter int INPUT_SIZE = 29;
    parameter int NUM_FOLDS  = 4;
    parameter int BW_W       = 10;
    parameter int BW_ACT     = 12;
    parameter int BW_ACC     = 32;
    parameter int N_RAND     = 500;

    localparam int ACT_MAX = (1 << (BW_ACT-1)) - 1;   //  2047
    localparam int ACT_MIN = -(1 << (BW_ACT-1));      // -2048
    localparam int W_MAX   = (1 << (BW_W-1)) - 1;     //   511
    localparam int W_MIN    = -(1 << (BW_W-1));       //  -512

    logic clock = 1'b0;
    logic reset;
    logic start;

    logic signed [BW_ACT-1:0] A [INPUT_SIZE-1:0];
    logic signed [BW_W-1:0]   W [INPUT_SIZE-1:0];

    logic                     done;
    logic signed [BW_ACC-1:0] acc_out;

    int errors     = 0;
    int checks     = 0;
    int done_pulse = 0;

    always #5 clock = ~clock;

    neuron_int #(
        .BW_ACC     (BW_ACC),
        .BW_W       (BW_W),
        .BW_ACT     (BW_ACT),
        .INPUT_SIZE (INPUT_SIZE),
        .NUM_FOLDS  (NUM_FOLDS)
    ) dut (
        .clock   (clock),
        .reset   (reset),
        .start   (start),
        .A       (A),
        .W       (W),
        .done    (done),
        .acc_out (acc_out)
    );

    // ---------------- golden model ----------------
    // Exact 64-bit integer dot product. Deliberately independent of the RTL's
    // fold structure: a flat loop, no NUM_FOLDS anywhere.
    function automatic longint golden();
        longint s;
        s = 0;
        for (int i = 0; i < INPUT_SIZE; i++)
            s = s + longint'(A[i]) * longint'(W[i]);
        return s;
    endfunction

    // ---------------- one transaction ----------------
    task automatic run_vec(input string name);
        longint exp;
        int     guard;
        begin
            exp = golden();

            @(negedge clock);
            start = 1'b1;
            @(negedge clock);
            start = 1'b0;

            // wait for done, with a timeout so a hang reports instead of spinning
            guard = 0;
            while (!done && guard < 1000) begin
                @(negedge clock);
                guard++;
            end

            checks++;
            if (guard >= 1000) begin
                $display("[FAIL] %-24s TIMEOUT waiting for done", name);
                errors++;
            end else if (acc_out !== BW_ACC'(exp)) begin
                $display("[FAIL] %-24s acc=%0d expected=%0d", name, acc_out, exp);
                errors++;
            end else begin
                $display("[ ok ] %-24s acc=%0d", name, acc_out);
            end

            // let done deassert before the next transaction
            @(negedge clock);
        end
    endtask

    task automatic fill(input int av, input int wv);
        for (int i = 0; i < INPUT_SIZE; i++) begin
            A[i] = BW_ACT'(av);
            W[i] = BW_W'(wv);
        end
    endtask

    // ---------------- done must be a one-cycle pulse ----------------
    // A multi-cycle done would let layer_quant latch the same acc twice.
    always @(posedge clock) begin
        if (!reset && done) begin
            done_pulse++;
            if (done_pulse > 1) begin
                $display("[FAIL] done asserted for >1 consecutive cycle");
                errors++;
            end
        end else if (!done) begin
            done_pulse = 0;
        end
    end

    // ---------------- stimulus ----------------
    initial begin
        reset = 1'b1;
        start = 1'b0;
        fill(0, 0);
        repeat (4) @(negedge clock);
        reset = 1'b0;
        @(negedge clock);

        $display("== neuron_int: INPUT_SIZE=%0d NUM_FOLDS=%0d (%0d fold cycles, %0d lanes idle on last) ==",
                 INPUT_SIZE, NUM_FOLDS,
                 (INPUT_SIZE + NUM_FOLDS - 1) / NUM_FOLDS,
                 ((INPUT_SIZE + NUM_FOLDS - 1) / NUM_FOLDS) * NUM_FOLDS - INPUT_SIZE);

        // 1. zeros
        fill(0, 0);
        run_vec("all zeros");

        // 2. unit
        fill(1, 1);
        run_vec("all +1 x +1");

        // 3. corners -- these are the magnitudes BW_ACC has to survive
        fill(ACT_MAX, W_MAX);
        run_vec("max x max");
        fill(ACT_MIN, W_MIN);
        run_vec("min x min (positive)");
        fill(ACT_MIN, W_MAX);
        run_vec("min x max (negative)");
        fill(ACT_MAX, W_MIN);
        run_vec("max x min (negative)");

        // 4. alternating signs -- a truncated (not sign-extended) negative
        //    product shows up here and nowhere else
        for (int i = 0; i < INPUT_SIZE; i++) begin
            A[i] = BW_ACT'((i % 2) ? ACT_MIN : ACT_MAX);
            W[i] = BW_W'((i % 2) ? W_MAX : W_MIN);
        end
        run_vec("alternating signs");

        // 5. one-hot -- walks a single 1 through every position. If the fold
        //    indexing drops or double-counts a lane, exactly one of these fails.
        for (int k = 0; k < INPUT_SIZE; k++) begin
            fill(0, 0);
            A[k] = BW_ACT'(1);
            W[k] = BW_W'(k + 1);
            run_vec($sformatf("one-hot idx=%0d", k));
        end

        // 6. randomised
        for (int t = 0; t < N_RAND; t++) begin
            for (int i = 0; i < INPUT_SIZE; i++) begin
                A[i] = BW_ACT'($urandom_range(0, 2**BW_ACT - 1));
                W[i] = BW_W'($urandom_range(0, 2**BW_W - 1));
            end
            run_vec($sformatf("random %0d", t));
        end

        // 7. back-to-back with no idle gap -- catches acc not clearing on start
        for (int t = 0; t < 8; t++) begin
            for (int i = 0; i < INPUT_SIZE; i++) begin
                A[i] = BW_ACT'($urandom_range(0, 2**BW_ACT - 1));
                W[i] = BW_W'($urandom_range(0, 2**BW_W - 1));
            end
            begin
                longint exp;
                int     guard;
                exp   = golden();
                guard = 0;
                @(negedge clock);
                start = 1'b1;
                @(negedge clock);
                start = 1'b0;
                while (!done && guard < 1000) begin
                    @(negedge clock);
                    guard++;
                end
                checks++;
                if (acc_out !== BW_ACC'(exp)) begin
                    $display("[FAIL] back-to-back %0d acc=%0d expected=%0d",
                             t, acc_out, exp);
                    errors++;
                end
            end
        end
        $display("[ ok ] back-to-back x8");

        $display("");
        $display("================================================");
        if (errors == 0)
            $display("PASS  %0d checks, 0 mismatches", checks);
        else
            $display("FAIL  %0d checks, %0d mismatches", checks, errors);
        $display("================================================");

        $finish;
    end

endmodule