module coord_split_format_tb();

    parameter BIT_WIDTH = 32;

    logic [BIT_WIDTH - 1:0] x_in;
    logic [BIT_WIDTH - 1:0] y_in;

    logic [BIT_WIDTH - 1:0] x_int;
    logic [BIT_WIDTH - 1:0] y_int;

    logic [BIT_WIDTH - 1:0] dx;
    logic [BIT_WIDTH - 1:0] dy;

    int errors = 0;
    int checks = 0;

    localparam precision = 1e-6;
    coordsplit_format  #(.BIT_WIDTH(32)) uut
        (.x_in(x_in), .y_in(y_in), .x_int(x_int), .y_int(y_int),
        .dx(dx), .dy(dy));

    function automatic shortreal fabs(input shortreal x);
        return (x < 0.0) ? -x : x;
    endfunction

    task automatic do_check(input shortreal val);
        shortreal gold_floor, gold_frac;
        shortreal got_int, got_dx;

        // drive the same value on both axes
        x_in = $shortrealtobits(val);
        y_in = $shortrealtobits(val);
        #1;   // combinational settle

        // golden reference computed in host float arithmetic
        gold_floor = $floor(val);
        gold_frac  = val - gold_floor;

        // reinterpret DUT outputs (FP32 bits) as floats
        got_int = $bitstoshortreal(x_int);
        got_dx  = $bitstoshortreal(dx);

        checks++;

        assert(got_int == gold_floor)
            else begin errors++;
                $error("floor mismatch: in=%f  got=%f  exp=%f", val, got_int, gold_floor); end

        assert(fabs(got_dx - gold_frac) < precision)
            else begin errors++;
                $error("frac  mismatch: in=%f  got=%f  exp=%f", val, got_dx, gold_frac); end
    endtask

    initial begin
        // ---- directed edge cases (where the branches & subtractor live) ----
        do_check(0.0);        // exact zero
        do_check(1.0);        // smallest exact integer >0, e=0 boundary
        do_check(255.0);      // top of range, exact integer -> dx must be 0
        do_check(0.5);        // pure fraction, e<0 path, floor=0
        do_check(0.25);       // another sub-1.0
        do_check(0.999969);   // just below 1.0 (near boundary)
        do_check(1.5);        // smallest non-integer >1
        do_check(128.5);      // e=7
        do_check(137.1);    // your worked example
        do_check(63.25);      // mid-range
        do_check(254.898);    // near top, multiple frac bits
        do_check(116.6645);

        // ---- sweep across the valid 0..255 range ----
        for (int i = 0; i < 256; i++) begin
            do_check(shortreal'(i));            // exact integer (dx should be 0)
            do_check(shortreal'(i) + 0.5);      // half
            do_check(shortreal'(i) + 0.25);     // quarter
            do_check(shortreal'(i) + 0.125);    // eighth
        end

        // ---- report ----
        if (errors == 0) $display("PASS, 0 errors, checks = %0d", checks);
        else             $display("FAIL, %0d errors, checks = %0d", errors, checks);

        $finish;
    end

endmodule