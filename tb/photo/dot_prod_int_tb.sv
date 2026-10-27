// dot_prod_int_tb.sv
//
// Golden reference is an EXACT integer sum -- no float, no tolerance. Any
// mismatch is a real bug, not a rounding artifact. That exactness is the whole
// reason dequantization was kept out of this module.
//
// Tests: directed edge cases (all zero, all max positive, all max negative,
// alternating signs) plus randomized vectors.

module dot_prod_int_tb;

    localparam int INPUT_SIZE = 29;
    localparam int NUM_FOLDS  = 4;
    localparam int BW_W       = 10;
    localparam int BW_ACT     = 12;
    localparam int BW_ACC     = 32;

    localparam int W_MAX   =  (1 << (BW_W-1)) - 1;      //  511
    localparam int W_MIN   = -(1 << (BW_W-1));          // -512
    localparam int ACT_MAX =  (1 << (BW_ACT-1)) - 1;    //  2047
    localparam int ACT_MIN = -(1 << (BW_ACT-1));        // -2048

    logic clock = 0;
    logic reset;
    logic start;
    logic signed [BW_ACT-1:0] ACT_Q [INPUT_SIZE];
    logic signed [BW_W-1:0]   W_Q   [INPUT_SIZE];
    logic signed [BW_ACC-1:0] dot_prod;
    logic done;

    int errors = 0;
    int tests  = 0;

    dot_prod_int #(
        .INPUT_SIZE(INPUT_SIZE), .NUM_FOLDS(NUM_FOLDS),
        .BW_W(BW_W), .BW_ACT(BW_ACT), .BW_ACC(BW_ACC)
    ) dut (
        .clock(clock), .reset(reset), .start(start),
        .ACT_Q(ACT_Q), .W_Q(W_Q),
        .dot_prod(dot_prod), .done(done)
    );

    always #5 clock = ~clock;

    // ---- exact integer golden reference ----
    function automatic longint golden_dot();
        longint sum;
        sum = 0;
        for (int i = 0; i < INPUT_SIZE; i++)
            sum += longint'(ACT_Q[i]) * longint'(W_Q[i]);
        return sum;
    endfunction

    task automatic run_case(string name);
        longint expected;
        int timeout;

        expected = golden_dot();
        tests++;

        @(posedge clock);
        start = 1'b1;
        @(posedge clock);
        start = 1'b0;

        timeout = 0;
        while (!done && timeout < 200) begin
            @(posedge clock);
            timeout++;
        end

        if (timeout >= 200) begin
            $display("[%s] TIMEOUT -- done never asserted", name);
            errors++;
        end else if (longint'(dot_prod) !== expected) begin
            $display("[%s] MISMATCH: got=%0d expected=%0d", name, longint'(dot_prod), expected);
            errors++;
        end else begin
            $display("[%s] PASS  (dot=%0d)", name, longint'(dot_prod));
        end

        @(posedge clock);
    endtask

    task automatic set_all(int act_val, int w_val);
        for (int i = 0; i < INPUT_SIZE; i++) begin
            ACT_Q[i] = act_val;
            W_Q[i]   = w_val;
        end
    endtask

    initial begin
        reset = 1'b1;
        start = 1'b0;
        set_all(0, 0);
        repeat (3) @(posedge clock);
        reset = 1'b0;
        @(posedge clock);

        // --- directed edge cases ---
        set_all(0, 0);
        run_case("all-zero");

        set_all(1, 1);
        run_case("all-ones");

        set_all(ACT_MAX, W_MAX);
        run_case("max-positive");          // largest positive accumulation

        set_all(ACT_MIN, W_MIN);
        run_case("max-negative-inputs");   // (-2048)*(-512) -> large POSITIVE sum

        set_all(ACT_MAX, W_MIN);
        run_case("max-magnitude-negative");// largest negative accumulation

        // alternating signs: checks sign handling per-element, not just uniformly
        for (int i = 0; i < INPUT_SIZE; i++) begin
            ACT_Q[i] = (i % 2 == 0) ?  ACT_MAX : ACT_MIN;
            W_Q[i]   = (i % 2 == 0) ?  W_MIN   : W_MAX;
        end
        run_case("alternating-signs");

        // single non-zero element at the LAST index -- catches fold-boundary
        // bugs where the final partial cycle is dropped or mis-indexed
        set_all(0, 0);
        ACT_Q[INPUT_SIZE-1] = 1234;
        W_Q[INPUT_SIZE-1]   = -321;
        run_case("last-element-only");

        // single non-zero at index 0
        set_all(0, 0);
        ACT_Q[0] = -2000;
        W_Q[0]   = 500;
        run_case("first-element-only");

        // --- randomized ---
        for (int t = 0; t < 20; t++) begin
            for (int i = 0; i < INPUT_SIZE; i++) begin
                ACT_Q[i] = $urandom_range(ACT_MAX - ACT_MIN, 0) + ACT_MIN;
                W_Q[i]   = $urandom_range(W_MAX   - W_MIN,   0) + W_MIN;
            end
            run_case($sformatf("random-%0d", t));
        end

        // --- back-to-back starts: confirm acc actually clears between runs ---
        set_all(100, 2);
        run_case("b2b-first");
        set_all(50, 3);
        run_case("b2b-second");   // fails if acc wasn't reset on start

        $display("=== %0d tests, %0d errors ===", tests, errors);
        if (errors == 0) $display("=== ALL PASS ===");
        else             $display("=== FAILURES PRESENT ===");
        $finish;
    end

endmodule