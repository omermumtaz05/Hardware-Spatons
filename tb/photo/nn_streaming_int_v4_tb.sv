

module nn_streaming_int_v4_tb();

    // ---------------- geometry ----------------
    localparam int IN_DIM  = 29;
    localparam int HIDDEN  = 48;
    localparam int OUT_DIM = 3;
    localparam int BW      = 32;   // internal zi width cap in v4, NOT a port width
    localparam int FEAT_W  = 8;    // uint8 feature port
    localparam int MAX_VEC = 64;

    // ---------------- v4 index domain ----------------
    localparam int  IDXW     = 12;                    // must match the DUT
    localparam int  LUT_N    = 1 << IDXW;             // 4096
    localparam real INV_STEP = (LUT_N - 1) / 16.0;    // 255.9375
    localparam int  IDX_TOL  = 1;                     // allowed |index error|, LSB

    // ---------------- run modes ----------------
    localparam string GOLDEN   = "data_files/a12_golden_mlp_int.txt";
    localparam bit    CALIB    = 0;   // 1 = stimulus sweep, no golden check
    localparam string STIM     = "stim_worst.hex";
    localparam int    SINK_GAP = 0;   // 0 = greedy sink -> measures intrinsic II
                                      // N > 0 = hold out_ready low N cycles after
                                      //         every transfer -> backpressure test

    logic clock, reset;
    logic in_valid, in_ready;
    logic out_valid, out_ready;
    logic [FEAT_W-1:0] f     [IN_DIM-1:0];
    logic [IDXW-1:0]   z_idx [OUT_DIM-1:0];   // WAS [BW-1:0] z

    nn_pipelined_int_v4 #(
        .IN_DIM(IN_DIM), .HIDDEN(HIDDEN), .OUT_DIM(OUT_DIM),
        .FEAT_W(FEAT_W), .IDXW(IDXW),
        .BW(BW), .BW_W(10), .BW_ACT(12), .BW_ACC(28)
    ) uut (
        .clock(clock), .reset(reset),
        .in_valid(in_valid), .in_ready(in_ready), .f(f),
        .out_valid(out_valid), .out_ready(out_ready), .z_idx(z_idx)
    );

    initial clock = 0;
    always #5 clock = ~clock;

    // ---------------- global cycle counter ----------------
    int unsigned cycle_now = 0;
    always @(posedge clock) cycle_now++;

    // ---------------- golden storage ----------------
    logic [BW-1:0]     vec_f    [MAX_VEC][IN_DIM];   // as loaded, fp32
    logic [FEAT_W-1:0] vec_f_u8 [MAX_VEC][IN_DIM];   // quantised, what we drive
    logic [BW-1:0]     vec_z    [MAX_VEC][OUT_DIM];  // golden fp32 z
    int                vec_idx  [MAX_VEC][OUT_DIM];  // golden z folded to an index
    int n_vec = 0;
    logic [FEAT_W-1:0] stim_flat [MAX_VEC*IN_DIM-1:0];
    int checks = 0, errors = 0;
    int idx_err_max = 0;

    // quantisation round-trip diagnostics
    shortreal q_err_max = 0.0;
    int       q_err_vec = -1, q_err_idx = -1;
    int       q_clamped = 0;

    function automatic shortreal b2f(input logic [31:0] b); return $bitstoshortreal(b); endfunction

    // fp32 feature in [0,1] -> uint8, round-half-up, clamped
    function automatic logic [FEAT_W-1:0] quant_u8(input logic [31:0] b);
        shortreal v;
        int       q;
        v = b2f(b);
        q = $rtoi(v * 255.0 + 0.5);
        if (q < 0)   begin q = 0;   q_clamped++; end
        if (q > 255) begin q = 255; q_clamped++; end
        return FEAT_W'(q);
    endfunction

    // golden fp32 z -> the delta-LUT address v4 is expected to emit.
    // This is post_mlp's old index equation, which is exactly what IDX_M folds.
    function automatic int exp_idx(input logic [31:0] zb);
        shortreal zv;
        int       i;
        zv = b2f(zb);
        i  = $rtoi($floor(real'(zv) * INV_STEP)) + LUT_N/2;
        if (i < 0)         i = 0;
        if (i > LUT_N - 1) i = LUT_N - 1;
        return i;
    endfunction

    // ---------------- load golden file up front, then quantise ----------------
    initial begin : load_golden
        integer   fd, code;
        string    sep;
        shortreal orig, back, err;

        if (CALIB) begin
            // $readmemh skips // comments natively, so gen_stim's labels are fine
            foreach (stim_flat[i]) stim_flat[i] = 'x;
            $readmemh(STIM, stim_flat);
            n_vec = 0;
            while (n_vec < MAX_VEC && stim_flat[n_vec*IN_DIM] !== 8'hxx) begin
                for (int i = 0; i < IN_DIM; i++)
                    vec_f_u8[n_vec][i] = stim_flat[n_vec*IN_DIM + i];
                n_vec++;
            end
            $display("[CALIB] loaded %0d stimulus vectors from %s", n_vec, STIM);
        end
        else begin
            fd = $fopen(GOLDEN, "r");
            if (fd == 0) begin $display("[FATAL] cannot open %s", GOLDEN); $finish; end
            while (!$feof(fd) && n_vec < MAX_VEC) begin
                for (int i = 0; i < IN_DIM; i++)  code = $fscanf(fd, "%h", vec_f[n_vec][i]);
                code = $fscanf(fd, " %s", sep);
                for (int k = 0; k < OUT_DIM; k++) code = $fscanf(fd, "%h", vec_z[n_vec][k]);
                if (code != 1) break;
                n_vec++;
            end
            $fclose(fd);

            // quantise every feature, measure the round-trip error,
            // and fold every golden z into its expected LUT address
            for (int v = 0; v < n_vec; v++) begin
                for (int i = 0; i < IN_DIM; i++) begin
                    vec_f_u8[v][i] = quant_u8(vec_f[v][i]);
                    orig = b2f(vec_f[v][i]);
                    back = shortreal'(vec_f_u8[v][i]) / 255.0;
                    err  = (orig > back) ? (orig - back) : (back - orig);
                    if (err > q_err_max) begin
                        q_err_max = err;  q_err_vec = v;  q_err_idx = i;
                    end
                end
                for (int k = 0; k < OUT_DIM; k++)
                    vec_idx[v][k] = exp_idx(vec_z[v][k]);
            end
        end
    end

    // ---------------- timestamps ----------------
    int unsigned t_in_fire  [MAX_VEC];   // cycle each pixel was accepted
    int unsigned t_out_fire [MAX_VEC];   // cycle each pixel's z_idx was drained
    int in_idx  = 0;
    int out_idx = 0;

    // ---------------- SOURCE: stream pixels back-to-back ----------------
    initial begin : source
        in_valid = 0;
        wait (reset === 0);
        @(negedge clock);
        // preload first vector
        for (int i = 0; i < IN_DIM; i++) f[i] = vec_f_u8[0][i];
        in_valid = 1'b1;
        while (in_idx < n_vec) begin
            @(posedge clock);
            if (in_valid && in_ready) begin      // fire: pixel in_idx accepted
                t_in_fire[in_idx] = cycle_now;
                in_idx++;
                @(negedge clock);                // swap f safely off-edge
                if (in_idx < n_vec)
                    for (int i = 0; i < IN_DIM; i++) f[i] = vec_f_u8[in_idx][i];
                else
                    in_valid = 1'b0;             // stream exhausted
            end
        end
    end

    // ---------------- SINK: drain and check every output in order ----------------
    // SINK_GAP == 0 is the greedy sink and measures the pipeline's own II.
    // SINK_GAP > 0 injects a fixed backpressure bubble after every transfer,
    // which is what you want to prove the handshake never drops or duplicates
    // a pixel when the downstream (post_mlp_int / output_buffer) is slow.
    initial begin : sink_drive
        out_ready = 1'b1;
        if (SINK_GAP > 0) forever begin
            @(posedge clock);
            if (out_valid && out_ready) begin
                @(negedge clock);
                out_ready = 1'b0;
                repeat (SINK_GAP) @(negedge clock);
                out_ready = 1'b1;
            end
        end
    end

    always @(posedge clock) begin : sink
        if (!reset && out_valid && out_ready) begin
            t_out_fire[out_idx] = cycle_now;
            checks++;
            if (!CALIB) begin
                for (int k = 0; k < OUT_DIM; k++) begin
                    int gotv, expv, d;
                    gotv = int'(z_idx[k]);
                    expv = vec_idx[out_idx][k];
                    d    = (gotv > expv) ? (gotv - expv) : (expv - gotv);
                    if (d > idx_err_max) idx_err_max = d;
                    if (d > IDX_TOL) begin
                        errors++;
                        $display("[FAIL] vec %0d z_idx[%0d] exp=%0d got=%0d  d=%0d",
                                 out_idx, k, expv, gotv, d);
                    end
                end
            end
            out_idx++;
        end
    end

    // ---------------- stall / occupancy counters ----------------
    int unsigned in_stall = 0, out_stall = 0, busy_cyc = 0;
    always @(posedge clock) if (!reset) begin
        if (in_valid  && !in_ready)  in_stall++;    // source blocked by the core
        if (out_valid && !out_ready) out_stall++;   // core blocked by the sink
        if (in_idx > 0 && out_idx < n_vec) busy_cyc++;
    end

    // ---------------- PER-LAYER LATENCY PROBES (hierarchical) ----------------
    // input-fire -> out_valid-rise for the FIRST pixel through each stage.
    // These signal names are unchanged in v4.
    int unsigned l1_lat = 0, l2_lat = 0, l3_lat = 0, res_lat = 0;
    int unsigned l1_t0 = 0, l2_t0 = 0, l3_t0 = 0, res_t0 = 0;

    always @(posedge clock) begin
        if (uut.l1_iv  && uut.l1_ir)             l1_t0   <= cycle_now;
        if (uut.l1_ov  && l1_lat  == 0)          l1_lat  <= cycle_now - l1_t0;
        if (uut.l2_iv  && uut.l2_ir)             l2_t0   <= cycle_now;
        if (uut.l2_ov  && l2_lat  == 0)          l2_lat  <= cycle_now - l2_t0;
        if (uut.l3_iv  && uut.l3_ir)             l3_t0   <= cycle_now;
        if (uut.l3_ov  && l3_lat  == 0)          l3_lat  <= cycle_now - l3_t0;
        if (uut.res_iv && uut.res_ir)            res_t0  <= cycle_now;
        if (uut.res_ov && res_lat == 0)          res_lat <= cycle_now - res_t0;
    end

    // ---------------- reset + final report ----------------
    initial begin : main
        reset = 1;
        repeat (3) @(posedge clock);
        reset = 0;

        wait (out_idx == n_vec || errors > 20);
        repeat (5) @(posedge clock);

        report();
        $finish;
    end

    task automatic report();
        int unsigned ii, ii_min, ii_max, ii_sum;
        int unsigned span, seq_est;
        int unsigned den;

        $display("=====================================================");
        $display(" nn_pipelined_int_v4  --  streaming report");
        $display(" mode: %s   SINK_GAP=%0d", CALIB ? "CALIB" : "GOLDEN", SINK_GAP);
        $display("-----------------------------------------------------");

        // ---- quantisation round trip: the /255 identity, checked on real data ----
        $display(" UINT8 round trip: max |fp32 - u8/255| = %e", q_err_max);
        if (CALIB) begin
            $display("        -> CALIB mode, golden checks skipped");
        end else if (q_err_max > 1e-6) begin
            errors++;
            $display(" [FAIL] golden features are NOT k/255 (worst: vec %0d feat %0d).",
                     q_err_vec, q_err_idx);
            $display("        The LUT-free design assumes /255 for both pixels and");
            $display("        coords. Check the resolution the golden was generated at.");
        end else begin
            $display("        -> exact. /255 identity holds, no golden regeneration needed.");
        end
        if (q_clamped != 0)
            $display(" [WARN] %0d features fell outside [0,1] and were clamped", q_clamped);
        $display("-----------------------------------------------------");

        if (!CALIB)
            $display(" LUT index: max |error| = %0d LSB  (tolerance %0d)",
                     idx_err_max, IDX_TOL);
        if (errors == 0) $display(" FUNCTIONAL: PASS  (%0d vectors, %0d checks)", n_vec, checks);
        else             $display(" FUNCTIONAL: FAIL  (%0d errors)", errors);
        $display("-----------------------------------------------------");

        if (out_idx == 0) begin
            $display(" [FATAL] no outputs drained -- nothing to time.");
            $display("=====================================================");
            return;
        end

        $display(" Fill latency (pixel 0 in -> out): %0d cycles",
                 t_out_fire[0] - t_in_fire[0]);

        ii_min = 32'hFFFFFFFF; ii_max = 0; ii_sum = 0;
        if (n_vec > 1) begin
            for (int i = 1; i < n_vec; i++) begin
                ii = t_out_fire[i] - t_out_fire[i-1];
                ii_sum += ii;
                if (ii < ii_min) ii_min = ii;
                if (ii > ii_max) ii_max = ii;
            end
            $display(" Initiation interval: avg %0d.%02d  min %0d  max %0d cycles",
                     ii_sum/(n_vec-1), (100*ii_sum/(n_vec-1))%100, ii_min, ii_max);
        end

        span = t_out_fire[n_vec-1] - t_in_fire[0];
        den  = (span == 0) ? 1 : span;
        $display(" Total: %0d pixels in %0d cycles", n_vec, span);

        seq_est = n_vec * (t_out_fire[0] - t_in_fire[0]);
        $display(" Sequential estimate (no overlap): ~%0d cycles  -> speedup ~%0d.%02dx",
                 seq_est, seq_est/den, (100*seq_est/den)%100);
        $display("-----------------------------------------------------");
        $display(" Source stall (in_valid  & !in_ready):  %0d cycles", in_stall);
        $display(" Sink   stall (out_valid & !out_ready): %0d cycles", out_stall);
        if (busy_cyc > 0)
            $display(" Throughput: %0d.%03d pixels per 1000 cycles",
                     (1000*n_vec)/busy_cyc, ((1000000*n_vec)/busy_cyc)%1000);
        $display("-----------------------------------------------------");
        $display(" Per-layer compute latency (first pixel):");
        $display("   L1  (29->48): %0d cycles", l1_lat);
        $display("   L2  (48->48): %0d cycles", l2_lat);
        $display("   L3  (48->3):  %0d cycles", l3_lat);
        $display("   RES (29->3):  %0d cycles", res_lat);
        if (l2_lat >= l3_lat && l2_lat >= l1_lat)
             $display(" BOTTLENECK: L2 -> II floor is ~%0d cycles", l2_lat);
        else if (l3_lat >= l1_lat)
             $display(" BOTTLENECK: L3 -> II floor is ~%0d cycles", l3_lat);
        else $display(" BOTTLENECK: L1 -> II floor is ~%0d cycles", l1_lat);
        $display("=====================================================");
    endtask

    initial begin
        #5000000;
        $display("[FAIL] TIMEOUT (out_idx=%0d of %0d)", out_idx, n_vec);
        $finish;
    end

endmodule