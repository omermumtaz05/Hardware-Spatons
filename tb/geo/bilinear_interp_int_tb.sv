`timescale 1ns/1ps

// bilinear_interp_int_tb.sv -- integer, 3-stage version
//
// ===========================================================================
// FOUR INDEPENDENT CHECKS, EACH CATCHING A DIFFERENT CLASS OF BUG
//
//  1. FLAT PATCH. If all four corners hold the same value V, the output must
//     be EXACTLY V for every (dx,dy). This is an indirect but total test of
//     the weight-sum invariant: it passes if and only if
//     w00+w01+w10+w11 == 2^16 exactly. Any width truncation on mdx/mdy, or a
//     wrong shift amount, breaks it immediately. Run exhaustively over all
//     65,536 (dx,dy) pairs.
//
//  2. REAL-ARITHMETIC MODEL. Checks the output is genuinely bilinear
//     interpolation with the CORRECT CORNER PAIRING -- P_01 weighted by
//     (1-dx)*dy and P_10 by dx*(1-dy). Swapping those two gives a transposed
//     interpolation that an integer reference would happily agree with if the
//     reference were written with the same mistake. This is the only check
//     that can catch a conceptual error rather than an arithmetic one.
//
//  3. BIT-EXACT INTEGER MODEL. Same arithmetic in 64-bit, so overflow and
//     rounding are checked exactly rather than within a tolerance.
//
//  4. STREAMING.  <-- THE IMPORTANT ONE
//     Every check above holds the inputs still for 3 cycles. A pipeline with
//     the pixel-forwarding registers MISSING passes all of them, because with
//     static inputs the "previous" pixel and the current one are identical.
//     Only pushing a different vector every cycle exposes it. If you delete
//     one test from this file, do not let it be this one.
//
// ===========================================================================
// NOTE ON dx = 1.0
//   Q0.8 cannot represent 1.0 -- dx maxes out at 255/256 = 0.99609375. So
//   "dx=1 should give exactly P_10" is not a valid expectation; the directed
//   cases use the real model, which knows this.
//
// ===========================================================================
// ICARUS COMPATIBILITY NOTE
//   iverilog cannot pass unpacked arrays as task/function arguments, so the
//   corner vectors live in module-scope arrays a00/a01/a10/a11 and the tasks
//   read them directly. Callers set the a-arrays, then call drive/do_check/
//   verify with only the scalar (dx,dy,tag) arguments.
//
// ===========================================================================
// RUN
//   iverilog -g2012 -o /tmp/bi_tb rtl/geo_stage/bilinear_interp_int.sv \
//            tb/geo_stage/bilinear_interp_int_tb.sv && /tmp/bi_tb

module bilinear_interp_int_tb();

    localparam int BIT_WIDTH_INT = 8;
    localparam int BIT_WIDTH_D   = 8;
    localparam int LAT           = 3;      // pipeline depth
    localparam int MAX_REPORT    = 15;
    localparam int NSTREAM       = 600;

    logic clock, reset, stall;

    logic [BIT_WIDTH_INT-1:0] P_00 [2:0], P_01 [2:0], P_10 [2:0], P_11 [2:0];
    logic [BIT_WIDTH_D-1:0]   dx, dy;
    logic [BIT_WIDTH_INT-1:0] P_out [2:0];

    int errors = 0, checks = 0, reported = 0;

    // ---- scratch corner vectors: the implicit arguments to drive/verify ----
    int a00 [0:2], a01 [0:2], a10 [0:2], a11 [0:2];

    // ---- streaming history ----
    int s00 [0:NSTREAM-1][0:2], s01 [0:NSTREAM-1][0:2];
    int s10 [0:NSTREAM-1][0:2], s11 [0:NSTREAM-1][0:2];
    int sdx [0:NSTREAM-1], sdy [0:NSTREAM-1];

    bilinear_interp_int #(
        .BIT_WIDTH_INT(BIT_WIDTH_INT), .BIT_WIDTH_D(BIT_WIDTH_D)
    ) uut (
        .clock(clock), .reset(reset), .stall(stall),
        .P_00(P_00), .P_01(P_01), .P_10(P_10), .P_11(P_11),
        .dx(dx), .dy(dy), .P_out(P_out)
    );

    initial clock = '0;
    always #5 clock = ~clock;

    task automatic fail(input string what, input string detail);
        errors++;
        if (reported < MAX_REPORT) begin
            reported++;
            $display("  FAIL [%s] %s", what, detail);
            if (reported == MAX_REPORT) $display("  ... further failures suppressed");
        end
    endtask

    // ---- bit-exact integer reference ----
    function automatic int unsigned ref_int(input int p00, input int p01,
                                            input int p10, input int p11,
                                            input int fdx, input int fdy);
        longint unsigned w00, w01, w10, w11, acc;
        begin
            w00 = (256 - fdx) * (256 - fdy);
            w10 =        fdx  * (256 - fdy);
            w01 = (256 - fdx) *        fdy;
            w11 =        fdx  *        fdy;
            acc = w00*p00 + w10*p10 + w01*p01 + w11*p11;
            ref_int = int'((acc + 32768) >> 16);
        end
    endfunction

    // ---- real-arithmetic reference: what bilinear interpolation MEANS ----
    function automatic real ref_real(input int p00, input int p01,
                                     input int p10, input int p11,
                                     input int fdx, input int fdy);
        real ux, uy;
        begin
            ux = real'(fdx) / 256.0;
            uy = real'(fdy) / 256.0;
            ref_real = (1.0-ux)*(1.0-uy)*real'(p00)     // P_00
                     +      ux *(1.0-uy)*real'(p10)     // P_10 : dx, not dy
                     + (1.0-ux)*     uy *real'(p01)     // P_01 : dy, not dx
                     +      ux *     uy *real'(p11);
        end
    endfunction

    // ---- set all three channels of the scratch vectors to one corner set ----
    task automatic set_corners(input int v00, input int v01,
                               input int v10, input int v11);
        for (int c = 0; c < 3; c++) begin
            a00[c] = v00; a01[c] = v01; a10[c] = v10; a11[c] = v11;
        end
    endtask

    // ---- drive the scratch vectors onto the DUT (does not wait) ----
    task automatic drive(input int fdx, input int fdy);
        for (int c = 0; c < 3; c++) begin
            P_00[c] = BIT_WIDTH_INT'(a00[c]);
            P_01[c] = BIT_WIDTH_INT'(a01[c]);
            P_10[c] = BIT_WIDTH_INT'(a10[c]);
            P_11[c] = BIT_WIDTH_INT'(a11[c]);
        end
        dx = BIT_WIDTH_D'(fdx);
        dy = BIT_WIDTH_D'(fdy);
    endtask

    // ---- compare P_out against both references ----
    task automatic verify(input string tag, input int fdx, input int fdy);
        int  exp_i;
        real exp_r, diff;
        begin
            checks++;
            for (int c = 0; c < 3; c++) begin
                exp_i = ref_int (a00[c], a01[c], a10[c], a11[c], fdx, fdy);
                exp_r = ref_real(a00[c], a01[c], a10[c], a11[c], fdx, fdy);

                if (int'(P_out[c]) !== exp_i)
                    fail($sformatf("%s-int", tag),
                         $sformatf("ch%0d dx=%0d dy=%0d P=(%0d,%0d,%0d,%0d) got=%0d exp=%0d",
                                   c, fdx, fdy, a00[c], a01[c], a10[c], a11[c],
                                   P_out[c], exp_i));

                // round-to-nearest means the result is within half an LSB of
                // the true value. A wider gap is a semantic error, not rounding.
                diff = real'(P_out[c]) - exp_r;
                if (diff > 0.5000001 || diff < -0.5000001)
                    fail($sformatf("%s-real", tag),
                         $sformatf("ch%0d dx=%0d dy=%0d got=%0d true=%.4f diff=%.4f",
                                   c, fdx, fdy, P_out[c], exp_r, diff));
            end
        end
    endtask

    // ---- drive + settle + verify, for static tests ----
    task automatic do_check(input int fdx, input int fdy, input string tag);
        begin
            drive(fdx, fdy);
            repeat (LAT) @(posedge clock);
            #1;
            verify(tag, fdx, fdy);
        end
    endtask

    initial begin
        reset = 1; stall = 0;
        for (int c = 0; c < 3; c++) begin
            P_00[c]='0; P_01[c]='0; P_10[c]='0; P_11[c]='0;
        end
        dx='0; dy='0;
        repeat (3) @(posedge clock);
        reset = 0;
        #1;

        $display("");
        $display("=== bilinear_interp : integer, 3-stage ===");
        $display("");

        // ==================================================================
        // 1. FLAT PATCH -- proves the weights sum to exactly 2^16
        // ==================================================================
        $display("-- flat patch, exhaustive over all 65,536 (dx,dy) --");
        begin : flat_test
            int err_before;
            err_before = errors;
            set_corners(137, 137, 137, 137);
            for (int vy = 0; vy < 256; vy++)
                for (int vx = 0; vx < 256; vx++) begin
                    drive(vx, vy);
                    repeat (LAT) @(posedge clock);
                    #1;
                    checks++;
                    for (int c = 0; c < 3; c++)
                        if (int'(P_out[c]) !== 137)
                            fail("flat", $sformatf("dx=%0d dy=%0d ch%0d got=%0d want 137",
                                                   vx, vy, c, P_out[c]));
                end
            if (errors == err_before)
                $display("   weight sum is exactly 2^16 on all 65,536 pairs");
        end

        // spot-check a few other flat values including both rails
        set_corners(0, 0, 0, 0);
        do_check(123,  77, "flat0");
        set_corners(255, 255, 255, 255);
        do_check(200, 199, "flat255");

        // ==================================================================
        // 2. DIRECTED -- corner pairing, distinct values per corner
        // ==================================================================
        $display("-- directed --");
        set_corners(0, 60, 180, 240);
        do_check(  0,   0, "corner00");  // all weight on P_00
        do_check(255,   0, "corner10");  // nearly all on P_10
        do_check(  0, 255, "corner01");  // nearly all on P_01
        do_check(255, 255, "corner11");  // nearly all on P_11
        do_check(128, 128, "centre");    // even mix
        do_check( 64,   0, "quarter-x");
        do_check(  0,  64, "quarter-y");

        // the hand-worked example: 25% from 100 to 200 is 125
        set_corners(100, 100, 200, 200);
        do_check(64, 0, "lerp-100-200");
        if (int'(P_out[0]) !== 125)
            fail("lerp", $sformatf("expected 125, got %0d", P_out[0]));

        // ==================================================================
        // 3. EXHAUSTIVE (dx,dy) with distinct corners
        // ==================================================================
        $display("-- exhaustive (dx,dy), distinct corners --");
        for (int c = 0; c < 3; c++) begin
            a00[c] = 10 + c*40; a01[c] = 200 - c*30;
            a10[c] = 90 + c*15; a11[c] = 255 - c*50;
        end
        for (int vy = 0; vy < 256; vy++)
            for (int vx = 0; vx < 256; vx++) begin
                drive(vx, vy);
                repeat (LAT) @(posedge clock);
                #1;
                verify("sweep", vx, vy);
            end

        // ==================================================================
        // 4. STREAMING -- the pipeline-alignment test
        // ==================================================================
        // Everything above holds the inputs still, so a build with the pixel
        // forwarding registers missing would pass all of it. Here a fresh
        // vector goes in every cycle, so stage 2 pairing the current pixel
        // with the previous weight shows up immediately.
        $display("-- streaming, one new vector per cycle --");
        begin : stream_test
            int err_before;
            err_before = errors;

            for (int t = 0; t < NSTREAM; t++) begin
                for (int c = 0; c < 3; c++) begin
                    s00[t][c] = $urandom_range(0,255);
                    s01[t][c] = $urandom_range(0,255);
                    s10[t][c] = $urandom_range(0,255);
                    s11[t][c] = $urandom_range(0,255);
                end
                sdx[t] = $urandom_range(0,255);
                sdy[t] = $urandom_range(0,255);
            end

            // drain: last vector goes in at t=NSTREAM-1 and comes out after
            // edge t=NSTREAM-1+(LAT-1), so the final t is NSTREAM+LAT-2.
            for (int t = 0; t < NSTREAM + LAT - 1; t++) begin
                if (t < NSTREAM) begin
                    for (int c = 0; c < 3; c++) begin
                        P_00[c] = BIT_WIDTH_INT'(s00[t][c]);
                        P_01[c] = BIT_WIDTH_INT'(s01[t][c]);
                        P_10[c] = BIT_WIDTH_INT'(s10[t][c]);
                        P_11[c] = BIT_WIDTH_INT'(s11[t][c]);
                    end
                    dx = BIT_WIDTH_D'(sdx[t]);
                    dy = BIT_WIDTH_D'(sdy[t]);
                end

                @(posedge clock);
                #1;

                // after edge t, P_out carries the vector driven at t-(LAT-1).
                // copy that historical vector into the scratch arrays so
                // verify() can see it.
                if (t >= LAT-1) begin : stream_chk
                    int k;
                    k = t - (LAT-1);
                    for (int c = 0; c < 3; c++) begin
                        a00[c] = s00[k][c]; a01[c] = s01[k][c];
                        a10[c] = s10[k][c]; a11[c] = s11[k][c];
                    end
                    verify("stream", sdx[k], sdy[k]);
                end
            end

            if (errors == err_before)
                $display("   %0d back-to-back vectors, pipeline stays aligned", NSTREAM);
        end

        // ==================================================================
        // 5. STALL -- output must freeze, then resume correctly
        // ==================================================================
        $display("-- stall hold --");
        begin : stall_test
            logic [BIT_WIDTH_INT-1:0] held [0:2];

            set_corners(11, 99, 170, 210);
            do_check(77, 33, "pre-stall");
            for (int c = 0; c < 3; c++) held[c] = P_out[c];

            stall = 1;
            drive(200, 200);                       // change inputs underneath
            repeat (5) @(posedge clock);
            #1;
            checks++;
            for (int c = 0; c < 3; c++)
                if (P_out[c] !== held[c])
                    fail("stall", $sformatf("ch%0d moved during stall: %0d -> %0d",
                                            c, held[c], P_out[c]));

            stall = 0;
            repeat (LAT) @(posedge clock);
            #1;
            verify("post-stall", 200, 200);
        end

        // ==================================================================
        $display("");
        if (errors == 0) $display("=== PASS === %0d checks", checks);
        else             $display("=== FAIL === %0d errors in %0d checks", errors, checks);
        $display("");

        if (errors != 0) $fatal(1, "bilinear_interp_int_tb failed");
        $finish;
    end

endmodule