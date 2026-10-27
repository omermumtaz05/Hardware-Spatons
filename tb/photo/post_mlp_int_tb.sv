// post_mlp_int_tb.sv -- self-checking TB for the all-integer post-MLP.
//
// RUN:
//     python3 gen_post_mlp_luts.py --tb --check     # writes both .hex files
//     vlog post_mlp_int.sv post_mlp_int_tb.sv && vsim -c -do "run -all" post_mlp_int_tb
//     # or: iverilog -g2012 -o tb post_mlp_int.sv post_mlp_int_tb.sv && ./tb
//
// ===========================================================================
// WHAT IT CHECKS, AND WHY EACH PART EXISTS
//
// A. GOLDEN SWEEP. Every one of the 4096 LUT indices x 8 boundary centres
//    (0, 1, 63, 64, 127, 128, 254, 255), 32768 vectors, driven back to back at
//    full rate with en held high.
//
//    The expected values come from the FLOAT model in gen_post_mlp_luts.py,
//    not from a re-implementation of the RTL here. A TB that recomputes
//    `center + dlut[idx]` in behavioural code and compares it to
//    the RTL doing the same thing proves only that you typed it twice.
//
//    The three channels are driven with DECORRELATED (idx, centre) pairs --
//    ch1 is offset by 1365 indices, ch2 by 2731. If ch0 and ch1 shared a
//    stimulus, a swapped-channel wiring bug (or Quartus failing to replicate
//    the ROM and all three channels reading one array) would pass silently.
//    This is exactly the failure mode the "one dlut per generate block" rule
//    in post_mlp_int.sv exists to prevent, so the TB must be able to see it.
//
// B. THROUGHPUT. The sweep drives a new vector EVERY cycle and checks the
//    result one cycle later. If the ROM read ever stops being a single
//    registered stage -- e.g. someone "fixes" the missing reset and Quartus
//    drops M10K inference, or an extra pipeline stage sneaks in -- the whole
//    sweep shifts by a cycle and fails immediately rather than subtly.
//
// C. en=0 HOLD. post_mlp_int has NO reset on its pipeline register by design
//    (a reset on a ROM output register defeats M10K inference). Validity is
//    tracked by photo_stage_int's p_valid instead. So the ONE thing the
//    register must guarantee is that it holds its value while en is low --
//    that is what makes the enclosing handshake correct. Checked explicitly.
//
// D. SATURATION CORNERS. Index 0 and index LUT_N-1 hold the forced +/-2
//    endpoints that replace the fp32_le sat_lo/sat_hi compares. delta = -2
//    must drive ANY centre to 0, and delta = +2 must drive ANY centre to 255,
//    because centre is in [0,1] and the clamp is to [0,1]. If delta_lut.hex
//    was generated without the endpoint forcing, these are the only vectors
//    that catch it (the error is 0.34 LSB everywhere else -- invisible).
//
// E. SCALE-MISMATCH TRIPWIRE. The most likely real-world failure is
//    regenerating delta_lut.hex without regenerating IDX_M, or vice versa.
//    That leaves post_mlp_int itself perfectly correct, so the sweep still
//    passes -- the damage is upstream in nn_pipelined_int_v3. The monotonicity
//    and endpoint checks below at least catch a table that is scaled wrong,
//    truncated, or read at the wrong width.

`timescale 1ns/1ps

module post_mlp_int_tb;

    // ---- must match post_mlp_int.sv and gen_post_mlp_luts.py ----
    localparam int PIX_W  = 8;
    localparam int IDXW   = 12;
    localparam int DLUT_W = 12;
    localparam int LUT_N  = 1 << IDXW;
    localparam int PMAX   = (1 << PIX_W) - 1;

    // Printed by gen_post_mlp_luts.py --tb. 4096 indices x 8 centres.
    localparam int TB_N   = 32768;

    localparam     DLUT_FILE   = "data_files/delta_lut.hex";
    localparam     GOLDEN_FILE = "data_files/post_mlp_tb_golden.hex";

    // =======================================================================
    logic clock = 1'b0;
    logic en;
    logic [IDXW-1:0]  z_idx     [2:0];
    logic [PIX_W-1:0] center_u8 [2:0];
    logic [PIX_W-1:0] final_rgb [2:0];

    always #5 clock = ~clock;          // 100 MHz

    post_mlp_int #(
        .PIX_W(PIX_W), .IDXW(IDXW), .DLUT_W(DLUT_W),
        .DLUT_FILE(DLUT_FILE)
    ) dut (
        .clock(clock), .en(en),
        .z_idx(z_idx), .center_u8(center_u8), .final_rgb(final_rgb)
    );

    // =======================================================================
    // golden vectors, packed 84b -- see build_tb_vectors() in the generator
    //   [83:72] idx2  [71:60] idx1  [59:48] idx0
    //   [47:40] c2    [39:32] c1    [31:24] c0
    //   [23:16] e2    [15:8]  e1    [7:0]   e0
    logic [83:0] gv [TB_N-1:0];

    function automatic logic [IDXW-1:0]  g_idx(input int n, input int c);
        return gv[n][48 + 12*c +: 12];
    endfunction
    function automatic logic [PIX_W-1:0] g_cen(input int n, input int c);
        return gv[n][24 + 8*c +: 8];
    endfunction
    function automatic logic [PIX_W-1:0] g_exp(input int n, input int c);
        return gv[n][8*c +: 8];
    endfunction

    // sanity: an all-X array means $readmemh silently found nothing, which
    // otherwise shows up as 32768 passing comparisons of X against X.
    task automatic check_loaded(input string name, input logic [83:0] probe);
        if (^probe === 1'bx) begin
            $display("FATAL: %s did not load. Run gen_post_mlp_luts.py --tb first.", name);
            $fatal(1);
        end
    endtask

    int errors  = 0;
    int checked = 0;

    task automatic expect_rgb(input int n);
        for (int c = 0; c < 3; c++) begin
            checked++;
            if (final_rgb[c] !== g_exp(n, c)) begin
                errors++;
                if (errors <= 20)
                    $display("MISMATCH vec %0d ch%0d: idx=%0d centre=%0d  got=%0d exp=%0d",
                             n, c, g_idx(n,c), g_cen(n,c), final_rgb[c], g_exp(n,c));
            end
        end
    endtask

    task automatic drive(input int n);
        for (int c = 0; c < 3; c++) begin
            z_idx[c]     = g_idx(n, c);
            center_u8[c] = g_cen(n, c);
        end
    endtask

    // =======================================================================
    // D/E: read the table directly to verify its shape independent of the DUT
    logic signed [DLUT_W-1:0] ref_lut [LUT_N-1:0];

    task automatic check_table;
        int bad_mono;
        $readmemh(DLUT_FILE, ref_lut);

        if (^ref_lut[0] === 1'bx) begin
            $display("FATAL: %s did not load.", DLUT_FILE);
            $fatal(1);
        end

        // endpoints must be the FORCED saturated values, 255*(-2) and 255*(+2).
        // These replace the fp32_le compares; if the generator omitted the
        // forcing they land at -510+1/+510-1 instead.
        if (ref_lut[0] !== -510) begin
            errors++;
            $display({"TABLE: dlut[0] = %0d, expected %0d ",
                      "(endpoint not forced -- delta_lut.hex is stale)"},
                     ref_lut[0], -510);
        end
        if (ref_lut[LUT_N-1] !== 510) begin
            errors++;
            $display("TABLE: dlut[%0d] = %0d, expected %0d",
                     LUT_N-1, ref_lut[LUT_N-1], 510);
        end

        // 4*sigmoid(z)-2 is strictly increasing, so the quantised table must be
        // non-decreasing. A table read at the wrong width, byte-swapped, or
        // generated with a different INV_STEP will not be.
        bad_mono = 0;
        for (int i = 1; i < LUT_N; i++)
            if (ref_lut[i] < ref_lut[i-1]) bad_mono++;
        if (bad_mono != 0) begin
            errors++;
            $display("TABLE: %0d non-monotonic steps -- wrong width or wrong INV_STEP",
                     bad_mono);
        end

        // must straddle zero, i.e. sigmoid(0) = 0.5 -> delta = 0 near mid-table
        if (!(ref_lut[LUT_N/2 - 1] <= 0 && ref_lut[LUT_N/2 + 1] >= 0)) begin
            errors++;
            $display({"TABLE: zero crossing is not at mid-table (%0d, %0d) -- ",
                      "IDX_M and delta_lut.hex disagree on INV_STEP"},
                     ref_lut[LUT_N/2 - 1], ref_lut[LUT_N/2 + 1]);
        end
        $display("[table] endpoints, monotonicity, zero-crossing checked");
    endtask

    // =======================================================================
    int hold_err;

    initial begin
        $readmemh(GOLDEN_FILE, gv);
        check_loaded(GOLDEN_FILE, gv[0]);
        check_table();

        en = 1'b0;
        drive(0);
        @(posedge clock);

        // ---- A/B: full-rate golden sweep -------------------------------
        // en high, one new vector per cycle, result checked one cycle later.
        en = 1'b1;
        drive(0);
        @(posedge clock);

        for (int n = 1; n < TB_N; n++) begin
            drive(n);
            #1 expect_rgb(n-1);      // settle the comb tail after the clock edge
            @(posedge clock);
        end
        #1 expect_rgb(TB_N-1);
        $display("[sweep] %0d vectors, %0d channel comparisons, %0d errors",
                 TB_N, checked, errors);

        // ---- C: en=0 must HOLD (there is no reset on that register) -----
        @(posedge clock);
        en = 1'b0;
        hold_err = 0;
        begin
            logic [PIX_W-1:0] held [2:0];
            #1 for (int c = 0; c < 3; c++) held[c] = final_rgb[c];

            for (int t = 0; t < 8; t++) begin
                // change the inputs underneath it -- the output must not move
                drive((t * 4099) % TB_N);
                @(posedge clock);
                #1 for (int c = 0; c < 3; c++)
                    if (final_rgb[c] !== held[c]) hold_err++;
            end
        end
        if (hold_err != 0) begin
            errors += hold_err;
            $display("HOLD: output moved %0d times while en was low", hold_err);
        end else
            $display("[hold] en=0 holds for 8 cycles, inputs changing");

        // ---- D: saturation corners, directed ---------------------------
        // idx 0 -> delta = -2 -> every centre in [0,1] clamps to 0
        // idx LUT_N-1 -> delta = +2 -> every centre clamps to 255
        en = 1'b1;
        for (int cv = 0; cv < 256; cv += 17) begin
            for (int c = 0; c < 3; c++) begin
                z_idx[c]     = '0;
                center_u8[c] = cv[PIX_W-1:0];
            end
            @(posedge clock); #1;
            for (int c = 0; c < 3; c++)
                if (final_rgb[c] !== 8'd0) begin
                    errors++;
                    $display("SAT-LO: idx=0 centre=%0d ch%0d -> %0d, expected 0",
                             cv, c, final_rgb[c]);
                end

            for (int c = 0; c < 3; c++) begin
                z_idx[c]     = IDXW'(LUT_N-1);
                center_u8[c] = cv[PIX_W-1:0];
            end
            @(posedge clock); #1;
            for (int c = 0; c < 3; c++)
                if (final_rgb[c] !== 8'd255) begin
                    errors++;
                    $display("SAT-HI: idx=%0d centre=%0d ch%0d -> %0d, expected 255",
                             LUT_N-1, cv, c, final_rgb[c]);
                end
        end
        $display("[sat] index 0 and %0d corners checked", LUT_N-1);

        // ---- E: mid-table identity -------------------------------------
        // delta ~= 0 at the zero crossing, so final should track centre. Not
        // exact for every centre (the crossing is between two bins), so allow
        // 1 LSB -- this is a scale tripwire, not a precision test.
        en = 1'b1;
        for (int cv = 0; cv < 256; cv += 5) begin
            int got;
            for (int c = 0; c < 3; c++) begin
                z_idx[c]     = IDXW'(LUT_N/2);
                center_u8[c] = cv[PIX_W-1:0];
            end
            @(posedge clock); #1;
            for (int c = 0; c < 3; c++) begin
                got = int'(final_rgb[c]);
                if ((got > cv + 1) || (got + 1 < cv)) begin
                    errors++;
                    $display({"MID: idx=%0d centre=%0d ch%0d -> %0d, expected ~%0d ",
                              "(delta_lut.hex scale is wrong)"},
                             LUT_N/2, cv, c, final_rgb[c], cv);
                end
            end
        end
        $display("[mid] zero-crossing identity checked");

        // =================================================================
        $display("\n=====================================================");
        if (errors == 0)
            $display("PASS  --  %0d comparisons, 0 errors", checked);
        else
            $display("FAIL  --  %0d errors out of %0d comparisons", errors, checked);
        $display("=====================================================");
        $finish;
    end

    initial begin
        #20_000_000;
        $display("FATAL: timeout");
        $fatal(1);
    end

endmodule