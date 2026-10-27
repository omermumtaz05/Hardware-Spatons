// tb_inter_line_buffer.sv
// Self-checking testbench for the circular line-buffer FIFO.
//
// Covers:
//   T1  fill phase: rd_lines_valid stays low until WINDOW lines committed
//   T2  full/backpressure: wr_line_ready drops at NUM_LINES, geo would stall
//   T3  release unblocks the writer (count 4->3 -> ready high again)
//   T4  data integrity: pixels written to a line read back correctly
//   T5  window sliding: after a release, window {L0,L1,L2} -> {L1,L2,L3}
//   T6  simultaneous commit+release: count holds, both pointers advance
//   T7   pointer wraparound past NUM_LINES-1
//   T8  underflow guard: consume while empty does not decrement below 0
//   T9  steady-state soak: many rows, count stays within [WINDOW-? .. NUM_LINES]

`timescale 1ns/1ps
module tb_inter_line_buffer;

    localparam int BIT_WIDTH_FP = 32;
    localparam int IMG_WIDTH    = 8;    // small so the TB is fast/readable
    localparam int NUM_LINES    = 4;
    localparam int WINDOW       = 3;
    localparam int COLW         = $clog2(IMG_WIDTH);
    localparam int LPTRW        = $clog2(NUM_LINES);

    logic clock, reset;

    logic                    wr_line_valid, wr_line_ready, wr_pixel_en;
    logic [COLW-1:0]         wr_col;
    logic [BIT_WIDTH_FP-1:0] wr_pixel [3];

    logic                    rd_lines_valid, rd_line_consume;
    logic [COLW-1:0]         rd_col;
    logic [LPTRW-1:0]        rd_line_sel;
    logic [BIT_WIDTH_FP-1:0] rd_pixel [3];

    logic [$clog2(NUM_LINES+1)-1:0] line_count;

    integer errors = 0;
    integer checks = 0;

    // ---- DUT ----
    inter_line_buffer #(
        .BIT_WIDTH_FP(BIT_WIDTH_FP), .IMG_WIDTH(IMG_WIDTH),
        .NUM_LINES(NUM_LINES), .WINDOW(WINDOW)
    ) dut (
        .clock(clock), .reset(reset),
        .wr_line_valid(wr_line_valid), .wr_line_ready(wr_line_ready),
        .wr_pixel_en(wr_pixel_en), .wr_col(wr_col), .wr_pixel(wr_pixel),
        .rd_lines_valid(rd_lines_valid), .rd_line_consume(rd_line_consume),
        .rd_col(rd_col), .rd_line_sel(rd_line_sel), .rd_pixel(rd_pixel),
        .line_count(line_count)
    );

    // ---- clock ----
    always #5 clock = ~clock;

    // ---- check helpers ----
    task automatic chk(input logic cond, input string msg);
        begin
            checks++;
            if (!cond) begin
                errors++;
                $display("  [FAIL] %s   (t=%0t)", msg, $time);
            end else begin
                $display("  [ ok ] %s", msg);
            end
        end
    endtask

    // write one full row of pixels into the current write line.
    // channel c of column j is encoded as row*1000 + j*10 + c so we can verify.
    task automatic write_row(input integer row);
        integer j;
        begin
            for (j = 0; j < IMG_WIDTH; j++) begin
                @(negedge clock);
                wr_pixel_en = 1'b1;
                wr_col      = j[COLW-1:0];
                wr_pixel[0] = row*1000 + j*10 + 0;
                wr_pixel[1] = row*1000 + j*10 + 1;
                wr_pixel[2] = row*1000 + j*10 + 2;
            end
            @(negedge clock);
            wr_pixel_en = 1'b0;
        end
    endtask

    // pulse a line commit (assumes ready; caller checks ready separately)
    task automatic commit_line;
        begin
            @(negedge clock);
            wr_line_valid = 1'b1;
            @(negedge clock);
            wr_line_valid = 1'b0;
        end
    endtask

    // pulse a line release
    task automatic release_line;
        begin
            @(negedge clock);
            rd_line_consume = 1'b1;
            @(negedge clock);
            rd_line_consume = 1'b0;
        end
    endtask

    // read back a pixel from window-line sel, column j (2-cycle: addr then data)
    task automatic read_pixel(input integer sel, input integer j,
                              output logic [BIT_WIDTH_FP-1:0] p0,
                              output logic [BIT_WIDTH_FP-1:0] p1,
                              output logic [BIT_WIDTH_FP-1:0] p2);
        begin
            @(negedge clock);
            rd_line_sel = sel[LPTRW-1:0];
            rd_col      = j[COLW-1:0];
            @(posedge clock);   // registered read latches here
            #1;
            p0 = rd_pixel[0]; p1 = rd_pixel[1]; p2 = rd_pixel[2];
        end
    endtask

    // ---- main ----
    initial begin
        $dumpfile("tb_inter_line_buffer.vcd");
        $dumpvars(0, tb_inter_line_buffer);

        clock = 0; reset = 1;
        wr_line_valid = 0; wr_pixel_en = 0; wr_col = 0;
        wr_pixel[0]=0; wr_pixel[1]=0; wr_pixel[2]=0;
        rd_line_consume = 0; rd_col = 0; rd_line_sel = 0;

        repeat (3) @(negedge clock);
        reset = 0;
        @(negedge clock);

        $display("\n==== T1: fill phase, rd_lines_valid low until WINDOW lines ====");
        chk(line_count == 0,        "count==0 after reset");
        chk(wr_line_ready == 1,     "writer ready when empty");
        chk(rd_lines_valid == 0,    "reader not valid when empty");

        write_row(0); commit_line();
        chk(line_count == 1,        "count==1 after 1 commit");
        chk(rd_lines_valid == 0,    "reader still invalid at 1 line");

        write_row(1); commit_line();
        chk(line_count == 2,        "count==2 after 2 commits");
        chk(rd_lines_valid == 0,    "reader still invalid at 2 lines");

        write_row(2); commit_line();
        chk(line_count == 3,        "count==3 after 3 commits");
        chk(rd_lines_valid == 1,    "reader VALID at WINDOW=3 lines");

        $display("\n==== T2: full / backpressure at NUM_LINES ====");
        write_row(3); commit_line();
        chk(line_count == 4,        "count==4 after 4th commit");
        chk(wr_line_ready == 0,     "writer NOT ready when full (geo would stall)");
        chk(rd_lines_valid == 1,    "reader still valid when full");

        // attempt an illegal commit while full: count must not exceed 4
        @(negedge clock); wr_line_valid = 1'b1;
        @(negedge clock); wr_line_valid = 1'b0;
        chk(line_count == 4,        "commit ignored while full (no overflow)");

        $display("\n==== T4: data integrity, read back line 0 (window sel=0) ====");
        begin
            logic [BIT_WIDTH_FP-1:0] a0,a1,a2;
            read_pixel(0, 5, a0, a1, a2);   // line 0, col 5
            chk(a0 == 0*1000 + 5*10 + 0, "L0 col5 ch0 correct");
            chk(a1 == 0*1000 + 5*10 + 1, "L0 col5 ch1 correct");
            chk(a2 == 0*1000 + 5*10 + 2, "L0 col5 ch2 correct");
            read_pixel(2, 3, a0, a1, a2);   // window line 2 -> physical L2 (row2)
            chk(a0 == 2*1000 + 3*10 + 0, "window sel2 col3 ch0 correct (row2)");
        end

        $display("\n==== T3 + T5: release unblocks writer, window slides ====");
        release_line();                    // release oldest (L0)
        chk(line_count == 3,        "count==3 after release");
        chk(wr_line_ready == 1,     "writer ready again after release");
        // after release, window sel0 should now be old row1 (physical L1)
        begin
            logic [BIT_WIDTH_FP-1:0] a0,a1,a2;
            read_pixel(0, 4, a0, a1, a2);
            chk(a0 == 1*1000 + 4*10 + 0, "after slide, sel0 == row1 (window moved)");
        end

        $display("\n==== T6: simultaneous commit + release, count holds ====");
        // currently count=3, ready=1. drive commit and release same cycle.
        write_row(4);                      // fill the freed slot's data
        @(negedge clock);
        wr_line_valid   = 1'b1;
        rd_line_consume = 1'b1;
        @(negedge clock);
        wr_line_valid   = 1'b0;
        rd_line_consume = 1'b0;
        chk(line_count == 3,        "count holds at 3 on simultaneous commit+release");

        $display("\n==== T7: pointer wraparound (drain then refill across the ring) ====");
        // drain to empty
        release_line(); release_line(); release_line();
        chk(line_count == 0,        "count==0 after draining");
        chk(rd_lines_valid == 0,    "reader invalid when drained");
        // refill 4 -> forces wr_line_ptr to wrap past NUM_LINES-1
        write_row(10); commit_line();
        write_row(11); commit_line();
        write_row(12); commit_line();
        write_row(13); commit_line();
        chk(line_count == 4,        "count==4 after wrap refill");
        begin
            logic [BIT_WIDTH_FP-1:0] a0,a1,a2;
            read_pixel(0, 1, a0, a1, a2);
            chk(a0 == 10*1000 + 1*10 + 0, "post-wrap sel0 == row10 (data coherent across wrap)");
        end

        $display("\n==== T8: underflow guard ====");
        // drain fully, then over-consume
        release_line(); release_line(); release_line(); release_line();
        chk(line_count == 0,        "count==0 fully drained");
        release_line();                    // illegal extra consume
        chk(line_count == 0,        "count stays 0 on consume-while-empty (no underflow)");

        $display("\n==== T9: steady-state soak (20 rows, invariant check) ====");
        begin
            integer r; integer bad_inv;
            bad_inv = 0;
            // prime with WINDOW lines
            write_row(20); commit_line();
            write_row(21); commit_line();
            write_row(22); commit_line();
            for (r = 23; r < 43; r++) begin
                if (wr_line_ready) begin
                    write_row(r); commit_line();
                end
                release_line();  // photo consumes one per iteration
                if (line_count > NUM_LINES) bad_inv++;
            end
            chk(bad_inv == 0,       "occupancy never exceeded NUM_LINES across soak");
        end

        $display("\n=====================================================");
        $display(" RESULT: %0d checks, %0d errors", checks, errors);
        if (errors == 0) $display(" ALL TESTS PASSED");
        else             $display(" *** %0d FAILURES ***", errors);
        $display("=====================================================\n");
        $finish;
    end

    // global watchdog
    initial begin
        #200000;
        $display("WATCHDOG TIMEOUT");
        $finish;
    end

endmodule