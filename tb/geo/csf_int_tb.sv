// csf_int_tb.sv -- exhaustive testbench for the Q8.8 coordinate split
//
// ===========================================================================
// WHAT THIS PROVES
//
// csf_int replaced fp32_floor / fp32_sub with two bit slices. That is a claim
// about MEANING, not just about wiring, so this checks three separate things:
//
//   1. SEMANTICS. Against a real-arithmetic golden model: x_int really is
//      floor(x_in / 256) and dx really is the fractional part scaled by 256.
//      This is the check that catches a conceptual error -- e.g. slicing the
//      wrong half, or an off-by-one in the field boundary.
//
//   2. RECONSTRUCTION. x_int * 256 + dx == x_in, exactly, for every input.
//      No bits invented, no bits lost.
//
//   3. EQUIVALENCE vs the old FP32 path (optional, see CHECK_VS_FP32 below).
//      q88_to_fp32 -> fp32_floor -> fp32_sub must produce the same values.
//      This is the one that answers "did my refactor change behaviour".
//
// ===========================================================================
// WHY EXHAUSTIVE IS ENOUGH -- AND WHY IT IS CHEAP
//
// The module is purely combinational and x and y share no logic, so full
// coverage of one 16-bit port plus a check that the other port cannot
// influence it is a COMPLETE proof, not a sample. 65,536 vectors is about a
// second of simulation.
//
// y_in is driven with the bitwise complement of x_in rather than the same
// value, so any accidental x/y crossover shows up immediately instead of
// hiding behind identical operands.
//
// ===========================================================================
// THE REAL MODEL IS EXACT HERE
//
// Normally comparing hardware against `real` invites tolerance arguments.
// Not in this case: 256 is a power of two and x_in < 2^16, so x_in/256.0 is
// exactly representable in a double, floor() of it is exact, and the
// fractional part times 256 is exactly an integer. No epsilon needed, and if
// you find yourself wanting one, something is actually wrong.
//
// ===========================================================================
// RUN
//   iverilog -g2012 -o csf_tb csf_int_tb.sv csf_int.sv && ./csf_tb
//
//   with the FP32 equivalence check (needs both packages on the command line):
//   iverilog -g2012 -DCHECK_VS_FP32 -o csf_tb \
//            csf_int_tb.sv csf_int.sv fp32_pkg.sv fp32_conv_pkg.sv && ./csf_tb

`timescale 1ns/1ps

`ifdef CHECK_VS_FP32
import fp32_pkg::*;
import fp32_conv_pkg::*;
`endif

module csf_int_tb;

    localparam int Q_WIDTH       = 16;
    localparam int BIT_WIDTH_INT = 8;
    localparam int BIT_WIDTH_D   = 8;

    localparam int MAX_REPORT = 10;   // stop spamming after this many failures

    logic [Q_WIDTH-1:0]       x_in, y_in;
    logic [BIT_WIDTH_D-1:0]   dx, dy;
    logic [BIT_WIDTH_INT-1:0] x_int, y_int;

    int errors   = 0;
    int reported = 0;
    int checks   = 0;

    // =======================================================================
    // DUT
    // =======================================================================
    csf_int #(
        .Q_WIDTH(Q_WIDTH),
        .BIT_WIDTH_INT(BIT_WIDTH_INT),
        .BIT_WIDTH_D(BIT_WIDTH_D)
    ) dut (
        .x_in(x_in), .y_in(y_in),
        .dx(dx), .dy(dy),
        .x_int(x_int), .y_int(y_int)
    );

    // =======================================================================
    // helpers
    // =======================================================================
    task automatic fail(input string what, input string detail);
        errors++;
        if (reported < MAX_REPORT) begin
            reported++;
            $display("  FAIL [%s] %s", what, detail);
            if (reported == MAX_REPORT)
                $display("  ... further failures suppressed");
        end
    endtask

`ifdef CHECK_VS_FP32
    // FP32 bit pattern -> real, for readable comparison against the old path
    function automatic real fp32_to_real(input logic [31:0] f);
        int  e;
        real m;
        begin
            if (f[30:23] == 8'd0) return 0.0;
            e = int'(f[30:23]) - 127;
            m = 1.0;
            for (int i = 0; i < 23; i++)
                if (f[22-i]) m = m + 2.0 ** (-(i+1));
            fp32_to_real = (f[31] ? -1.0 : 1.0) * m * (2.0 ** e);
        end
    endfunction
`endif

    // =======================================================================
    // the check, applied to one (x, y) pair
    // =======================================================================
    task automatic check_pair(input logic [Q_WIDTH-1:0] xv,
                              input logic [Q_WIDTH-1:0] yv);
        real val;
        int  exp_int, exp_frac;
        int  recon;
`ifdef CHECK_VS_FP32
        logic [31:0] fp, old_int_fp, old_dx_fp;
        real         old_int_r, old_dx_r;
`endif
        begin
            x_in = xv;
            y_in = yv;
            #1;
            checks++;

            // ---- 1. semantics: real-arithmetic golden model, X ----
            val      = real'(xv) / 256.0;
            exp_int  = int'($floor(val));
            exp_frac = int'($rtoi((val - $floor(val)) * 256.0));

            if (int'(x_int) !== exp_int)
                fail("x_int", $sformatf("x_in=%04h (%0.6f)  got %0d  want %0d",
                                        xv, val, x_int, exp_int));
            if (int'(dx) !== exp_frac)
                fail("dx", $sformatf("x_in=%04h (%0.6f)  got %0d  want %0d",
                                     xv, val, dx, exp_frac));

            // ---- semantics, Y ----
            val      = real'(yv) / 256.0;
            exp_int  = int'($floor(val));
            exp_frac = int'($rtoi((val - $floor(val)) * 256.0));

            if (int'(y_int) !== exp_int)
                fail("y_int", $sformatf("y_in=%04h (%0.6f)  got %0d  want %0d",
                                        yv, val, y_int, exp_int));
            if (int'(dy) !== exp_frac)
                fail("dy", $sformatf("y_in=%04h (%0.6f)  got %0d  want %0d",
                                     yv, val, dy, exp_frac));

            // ---- 2. reconstruction: nothing invented, nothing lost ----
            recon = int'(x_int) * 256 + int'(dx);
            if (recon !== int'(xv))
                fail("recon-x", $sformatf("x_in=%04h  x_int*256+dx=%0d", xv, recon));

            recon = int'(y_int) * 256 + int'(dy);
            if (recon !== int'(yv))
                fail("recon-y", $sformatf("y_in=%04h  y_int*256+dy=%0d", yv, recon));

`ifdef CHECK_VS_FP32
            // ---- 3. equivalence with the FP32 path this replaced ----
            fp         = q88_to_fp32(xv);
            old_int_fp = fp32_floor(fp);
            old_dx_fp  = fp32_sub(fp, old_int_fp);

            old_int_r = fp32_to_real(old_int_fp);
            old_dx_r  = fp32_to_real(old_dx_fp);

            if (real'(x_int) != old_int_r)
                fail("fp32-int", $sformatf("x_in=%04h  new=%0d  old(fp32)=%0.6f",
                                           xv, x_int, old_int_r));

            // new dx is Q0.8, old was a float in [0,1)
            if (real'(dx) / 256.0 != old_dx_r)
                fail("fp32-dx", $sformatf("x_in=%04h  new=%0.6f  old(fp32)=%0.6f",
                                          xv, real'(dx) / 256.0, old_dx_r));
`endif
        end
    endtask

    // =======================================================================
    // stimulus
    // =======================================================================
    initial begin
        $display("");
        $display("=== csf_int : Q8.8 coordinate split ===");
`ifdef CHECK_VS_FP32
        $display("    FP32 equivalence check ENABLED");
`else
        $display("    FP32 equivalence check off (-DCHECK_VS_FP32 to enable)");
`endif
        $display("");

        // ---- directed cases, including the hand-worked example ----
        $display("-- directed --");
        check_pair(16'h89A0, 16'h0100);  // 137.6015625 -> int 137, dx 160
        check_pair(16'h0000, 16'hFFFF);  // 0.0         -> int 0,   dx 0
        check_pair(16'h00FF, 16'h0000);  // 0.99609375  -> int 0,   dx 255
        check_pair(16'h0100, 16'h00FF);  // 1.0         -> int 1,   dx 0
        check_pair(16'hFF00, 16'h0100);  // 255.0       -> int 255, dx 0
        check_pair(16'hFFFF, 16'h8000);  // 255.99609375-> int 255, dx 255
        check_pair(16'h8000, 16'h7FFF);  // 128.0       -> int 128, dx 0
        check_pair(16'h0080, 16'h0080);  // 0.5         -> int 0,   dx 128
        $display("   %0d checks, %0d errors so far", checks, errors);

        // ---- exhaustive over every 16-bit input ----
        // y_in gets ~x_in so an x/y crossover cannot hide behind equal operands.
        $display("-- exhaustive (65,536 vectors) --");
        for (int v = 0; v < 65536; v++)
            check_pair(Q_WIDTH'(v), Q_WIDTH'(~v));

        // ---- independence: x outputs must not move when only y moves ----
        $display("-- x/y independence --");
        begin
            logic [BIT_WIDTH_INT-1:0] xi_ref;
            logic [BIT_WIDTH_D-1:0]   dx_ref;

            x_in = 16'h89A0; y_in = 16'h0000; #1;
            xi_ref = x_int; dx_ref = dx;

            for (int v = 0; v < 65536; v += 257) begin   // 257 is coprime with 256
                y_in = Q_WIDTH'(v); #1;
                if (x_int !== xi_ref || dx !== dx_ref)
                    fail("independence",
                         $sformatf("y_in=%04h perturbed x outputs (%0d,%0d) -> (%0d,%0d)",
                                   v, xi_ref, dx_ref, x_int, dx));
            end
        end

        // ---- summary ----
        $display("");
        if (errors == 0) begin
            $display("=== PASS === %0d checks, exhaustive over all 2^16 inputs", checks);
        end else begin
            $display("=== FAIL === %0d errors in %0d checks", errors, checks);
        end
        $display("");

        if (errors != 0) $fatal(1, "csf_int_tb failed");
        $finish;
    end

endmodule