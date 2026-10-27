
`timescale 1ns/1ps
 
`ifdef CHECK_VS_FP32
import fp32_pkg::*;
`endif
 
module ffn_int_tb;
 
    localparam int IND_WIDTH  = 8;
    localparam int IMG_WIDTH  = 256;
    localparam int IMG_HEIGHT = 256;
 
    localparam int XMAX = IMG_WIDTH  - 1;
    localparam int YMAX = IMG_HEIGHT - 1;
 
    localparam int MAX_REPORT = 10;
 
    logic [IND_WIDTH-1:0]   x_int, y_int;
    logic [2*IND_WIDTH-1:0] A_00, A_01, A_10, A_11;
 
    int errors   = 0;
    int reported = 0;
    int checks   = 0;
 
    // =======================================================================
    // DUT
    // =======================================================================
    ffn_int #(
        .IND_WIDTH(IND_WIDTH), .IMG_WIDTH(IMG_WIDTH), .IMG_HEIGHT(IMG_HEIGHT)
    ) dut (
        .x_int(x_int), .y_int(y_int),
        .A_00(A_00), .A_01(A_01), .A_10(A_10), .A_11(A_11)
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
 
    // unpack a packed {y, x} corner
    function automatic int px(input logic [2*IND_WIDTH-1:0] a);
        return int'(a[IND_WIDTH-1:0]);
    endfunction
    function automatic int py(input logic [2*IND_WIDTH-1:0] a);
        return int'(a[2*IND_WIDTH-1:IND_WIDTH]);
    endfunction
 
    // =======================================================================
    // the check, applied to one (x, y)
    // =======================================================================
    task automatic check_pair(input int xv, input int yv);
        int ex1, ey1;
        begin
            x_int = IND_WIDTH'(xv);
            y_int = IND_WIDTH'(yv);
            #1;
            checks++;
 
            ex1 = (xv >= XMAX) ? XMAX : xv + 1;
            ey1 = (yv >= YMAX) ? YMAX : yv + 1;
 
            // ---- 1 + 3. corner mapping and packing, all four at once ----
            // A_01 must be the Y neighbour, A_10 the X neighbour.
            if (px(A_00) !== xv  || py(A_00) !== yv)
                fail("A_00", $sformatf("(%0d,%0d) -> got (%0d,%0d) want (%0d,%0d)",
                                       xv, yv, px(A_00), py(A_00), xv, yv));
 
            if (px(A_01) !== xv  || py(A_01) !== ey1)
                fail("A_01", $sformatf("(%0d,%0d) -> got (%0d,%0d) want (%0d,%0d)  [must be the Y neighbour]",
                                       xv, yv, px(A_01), py(A_01), xv, ey1));
 
            if (px(A_10) !== ex1 || py(A_10) !== yv)
                fail("A_10", $sformatf("(%0d,%0d) -> got (%0d,%0d) want (%0d,%0d)  [must be the X neighbour]",
                                       xv, yv, px(A_10), py(A_10), ex1, yv));
 
            if (px(A_11) !== ex1 || py(A_11) !== ey1)
                fail("A_11", $sformatf("(%0d,%0d) -> got (%0d,%0d) want (%0d,%0d)",
                                       xv, yv, px(A_11), py(A_11), ex1, ey1));
 
            // ---- 2. wrap detection, called out by name ----
            // If the clamp is missing, 255+1 wraps to 0. Report that as a
            // wrap rather than a generic value mismatch -- it is a distinct
            // and much more dangerous bug.
            if (xv == XMAX && px(A_10) == 0)
                fail("WRAP-X", $sformatf("x=%0d: x+1 wrapped to 0, clamp is missing", xv));
            if (yv == YMAX && py(A_01) == 0)
                fail("WRAP-Y", $sformatf("y=%0d: y+1 wrapped to 0, clamp is missing", yv));
 
`ifdef CHECK_VS_FP32
            begin
                logic [31:0] fx, fy, fx1, fy1;
                localparam logic [31:0] FP_ONE   = 32'h3F80_0000;
                localparam logic [31:0] FP_MAX_X = 32'h437F_0000;   // 255.0
 
                // reproduce the old module's arithmetic on the same coords
                fx  = int_to_fp32_small(xv);
                fy  = int_to_fp32_small(yv);
                fx1 = (fx >= FP_MAX_X) ? FP_MAX_X : fp32_add(fx, FP_ONE);
                fy1 = (fy >= FP_MAX_X) ? FP_MAX_X : fp32_add(fy, FP_ONE);
 
                if (fp32_small_to_int(fx1) !== ex1)
                    fail("fp32-x1", $sformatf("x=%0d: new=%0d old(fp32)=%0d",
                                              xv, ex1, fp32_small_to_int(fx1)));
                if (fp32_small_to_int(fy1) !== ey1)
                    fail("fp32-y1", $sformatf("y=%0d: new=%0d old(fp32)=%0d",
                                              yv, ey1, fp32_small_to_int(fy1)));
            end
`endif
        end
    endtask
 
`ifdef CHECK_VS_FP32
    // Tiny exact converters for 0..255. Not general -- deliberately so, the
    // range is known and this keeps the reference model obviously correct.
    function automatic logic [31:0] int_to_fp32_small(input int v);
        int e, m;
        begin
            if (v == 0) return 32'h0000_0000;
            e = 0; m = v;
            while (m > 1) begin m = m >> 1; e++; end
            // v = 1.frac * 2^e ; mantissa = (v << (23-e)) with the implicit 1 dropped
            int_to_fp32_small = {1'b0, 8'(127 + e),
                                 23'((v << (23 - e)) & 23'h7F_FFFF)};
        end
    endfunction
 
    function automatic int fp32_small_to_int(input logic [31:0] f);
        int e;
        begin
            if (f[30:23] == 0) return 0;
            e = int'(f[30:23]) - 127;
            fp32_small_to_int = int'({1'b1, f[22:0]} >> (23 - e));
        end
    endfunction
`endif
 
    // =======================================================================
    // stimulus
    // =======================================================================
    initial begin
        $display("");
        $display("=== ffn_int : integer four-neighbour fetch ===");
`ifdef CHECK_VS_FP32
        $display("    FP32 equivalence check ENABLED");
`else
        $display("    FP32 equivalence check off (-DCHECK_VS_FP32 to enable)");
`endif
        $display("");
 
        // ---- directed: interior, both edges, the corner ----
        $display("-- directed --");
        check_pair(0,    0);       // origin
        check_pair(137,  92);      // ordinary interior
        check_pair(XMAX, 0);       // right edge   -> x+1 must clamp
        check_pair(0,    YMAX);    // bottom edge  -> y+1 must clamp
        check_pair(XMAX, YMAX);    // far corner   -> both clamp
        check_pair(XMAX-1, YMAX-1);// last pair that still increments
        $display("   %0d checks, %0d errors so far", checks, errors);
 
        // ---- exhaustive over every coordinate in the image ----
        $display("-- exhaustive (256 x 256 = 65,536 pairs) --");
        for (int yv = 0; yv < IMG_HEIGHT; yv++)
            for (int xv = 0; xv < IMG_WIDTH; xv++)
                check_pair(xv, yv);
 
        // ---- edge sweeps, reported separately so a clamp bug is unmistakable ----
        $display("-- edge sweep: right column and bottom row --");
        begin
            int edge_errs = errors;
            for (int yv = 0; yv < IMG_HEIGHT; yv++) check_pair(XMAX, yv);
            for (int xv = 0; xv < IMG_WIDTH;  xv++) check_pair(xv, YMAX);
            if (errors == edge_errs)
                $display("   clamp holds on all %0d edge coordinates",
                         IMG_WIDTH + IMG_HEIGHT);
        end
 
        // ---- summary ----
        $display("");
        if (errors == 0)
            $display("=== PASS === %0d checks, exhaustive over all %0dx%0d coordinates",
                     checks, IMG_WIDTH, IMG_HEIGHT);
        else
            $display("=== FAIL === %0d errors in %0d checks", errors, checks);
        $display("");
 
        if (errors != 0) $fatal(1, "ffn_int_tb failed");
        $finish;
    end
 
endmodule
 