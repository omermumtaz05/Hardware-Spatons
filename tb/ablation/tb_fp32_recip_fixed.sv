// tb_fp32_recip_fixed.sv
//
//   iverilog -g2012 -o tb tb_fp32_recip_fixed.sv && ./tb
//   (or: verilator --binary --timing tb_fp32_recip_fixed.sv)
//
// No shortreal, no $bitstoreal. The comparison metric is the unsigned integer
// distance between bit patterns, which for positive floats IS the ulp distance
// -- stricter and more portable than a float compare. 1 ulp ~= 6e-8 relative,
// well inside the 1e-6 target.

`timescale 1ns/1ps
    import fp32_pkg::*;


module tb_fp32_recip_fixed;

    int          fd, code, n, fails, ulp1;
    logic [31:0] x, ref_r, dut_r;
    int unsigned d, worst;
    logic [31:0] worst_x, worst_ref, worst_dut;

    initial begin
        fd = $fopen("recip_vec.txt", "r");
        if (fd == 0) begin
            $display("FATAL: cannot open recip_vec.txt (run gen_recip_vectors.py)");
            $fatal(1);
        end

        n = 0; fails = 0; ulp1 = 0; worst = 0;

        code = $fscanf(fd, "%h %h\n", x, ref_r);
        while (code == 2) begin
            dut_r = fp32_recip_fixed(x);

            d = (dut_r > ref_r) ? (dut_r - ref_r) : (ref_r - dut_r);

            if (d > worst) begin
                worst = d; worst_x = x; worst_ref = ref_r; worst_dut = dut_r;
            end
            if (d == 1) ulp1++;
            if (d > 1) begin
                fails++;
                if (fails <= 10)
                    $display("MISMATCH x=%h ref=%h dut=%h ulp=%0d", x, ref_r, dut_r, d);
            end
            n++;
            code = $fscanf(fd, "%h %h\n", x, ref_r);
        end
        $fclose(fd);

        // --- directed edge cases ---
        check_eq("1.0    ", fp32_recip_fixed(32'h3F800000), 32'h3F800000);
        check_eq("2.0    ", fp32_recip_fixed(32'h40000000), 32'h3F000000);
        check_eq("0.5    ", fp32_recip_fixed(32'h3F000000), 32'h40000000);
        check_eq("-2.0   ", fp32_recip_fixed(32'hC0000000), 32'hBF000000);
        check_eq("+0     ", fp32_recip_fixed(32'h00000000), 32'h7F800000);
        check_eq("+inf   ", fp32_recip_fixed(32'h7F800000), 32'h00000000);
        // denormal results are flushed to zero by design (never reachable from
        // a real abs_max; documenting the behaviour, not endorsing it)
        check_eq("FTZ    ", fp32_recip_fixed(32'h7F7FFFFF), 32'h00000000);

        $display("--------------------------------------------------");
        $display("vectors        : %0d", n);
        $display("exact (0 ulp)  : %0d", n - ulp1 - fails);
        $display("1 ulp off      : %0d", ulp1);
        $display("FAILS (>1 ulp) : %0d", fails);
        $display("worst: x=%h ref=%h dut=%h  ulp=%0d", worst_x, worst_ref, worst_dut, worst);
        $display("--------------------------------------------------");
        if (fails == 0) $display("PASS"); else $display("FAIL");
    end

    task automatic check_eq(input string nm, input logic [31:0] got, exp);
        if (got !== exp) begin
            $display("EDGE FAIL %s got=%h exp=%h", nm, got, exp);
            fails++;
        end
    endtask
endmodule