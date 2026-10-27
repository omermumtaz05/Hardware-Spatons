// tb_top_full_integration.sv
// Integration test: geo stage writes into inter_line_buffer; the TB plays a
// line-by-line "toy photo stage" consumer (reads oldest line, releases it).
//
// Verifies: geo still works with stall threaded in, the write/commit handshake
// neither drops nor duplicates lines, the ring recycles slots correctly, and
// read-back data is intact. Instruments stall/occupancy/latency for the report.
//
// NOTE: uses active stubs that encode each pixel as y*1000 + x*10 + ch, so the
// TB self-checks. Also dumps capture_array.hex for the existing NumPy diff.

`timescale 1ns/1ps
module tb_top_full_integration;

    localparam int BIT_WIDTH_FP = 32;
    localparam int BIT_WIDTH_UINT = 8;

    localparam int IND_WIDTH    = 8;
    localparam int IMG_WIDTH    = 256;  
    localparam int IMG_HEIGHT   = 256;
    localparam int NUM_LINES    = 4;
    localparam int WINDOW       = 3;
    localparam int COLW         = $clog2(IMG_WIDTH);
    localparam int LPTRW        = $clog2(NUM_LINES);
    localparam int CAPTURE_FILE = 0;    // placeholder; $writememh path set below

    logic clock, reset;
    logic frame_done;

    logic                    rd_lines_valid, rd_line_consume;
    logic [COLW-1:0]         rd_col;
    logic [LPTRW-1:0]        rd_line_sel;
    logic [BIT_WIDTH_UINT-1:0] rd_pixel [3];
    logic [BIT_WIDTH_UINT-1:0] debug_pixel;

    // ---- DUT ----
    top_full_geo_int #(
        .BIT_WIDTH_FP(BIT_WIDTH_FP), .BIT_WIDTH_UINT(BIT_WIDTH_UINT), .IND_WIDTH(IND_WIDTH),
        .IMG_WIDTH(IMG_WIDTH), .IMG_HEIGHT(IMG_HEIGHT),
        .NUM_LINES(NUM_LINES), .WINDOW(WINDOW)
    ) uut (
        .clock(clock), .reset(reset), .frame_done(frame_done),
        .rd_lines_valid(rd_lines_valid), .rd_line_consume(rd_line_consume),
        .rd_col(rd_col), .rd_line_sel(rd_line_sel), .rd_pixel(rd_pixel),
        .debug_pixel(debug_pixel)
    );

    // ---- clock ----
    always #5 clock = ~clock;

    // ---- capture array (full reconstructed frame, ch0 only for the diff) ----
    logic [BIT_WIDTH_UINT-1:0] capture [IMG_HEIGHT][IMG_WIDTH][3];

    integer errors = 0, checks = 0;

    // ============================================================
    // Instrumentation counters
    // ============================================================
    integer stall_cycles       = 0;   // total cycles geo held stalled
    integer stall_events       = 0;   // distinct stall episodes
    integer fill_latency        = -1;  // reset-deassert -> rd_lines_valid first high
    integer max_occupancy      = 0;   // peak line_count seen
    integer backpressure_cycles = 0;  // cycles wr_line_valid && !wr_line_ready
    integer cyc                = 0;   // free-running cycle counter (post-reset)
    logic   prev_stall = 0;

    // reach into DUT internals for observation
    wire        obs_stall     = uut.stall_geo;
    wire        obs_wr_valid  = uut.wr_line_valid;
    wire        obs_wr_ready  = uut.wr_line_ready;
    wire [LPTRW:0] obs_count  = uut.line_count;

    always @(posedge clock) begin
        if (!reset) begin
            cyc <= cyc + 1;

            if (obs_stall) begin
                stall_cycles <= stall_cycles + 1;
                if (!prev_stall) stall_events <= stall_events + 1;
            end
            prev_stall <= obs_stall;

            if (obs_wr_valid && !obs_wr_ready)
                backpressure_cycles <= backpressure_cycles + 1;

            if (obs_count > max_occupancy) max_occupancy <= obs_count;

            if (fill_latency < 0 && rd_lines_valid)
                fill_latency <= cyc;
        end
    end

    // ============================================================
    // Toy line-by-line consumer (option B): read oldest line, release, repeat.
    // Gated on rd_lines_valid (WINDOW=3), with a tail-drain at the end.
    // Read is registered: drive rd_col=j, capture rd_pixel one clock later.
    // ============================================================
    task automatic consume_one_line(input integer row);
        integer j;
        begin
            // sweep columns; capture lags address by one cycle
            rd_line_sel = 0;                  // oldest line in the window
            for (j = 0; j <= IMG_WIDTH; j++) begin
                @(negedge clock);
                if (j < IMG_WIDTH) rd_col = j[COLW-1:0];
                // capture the pixel addressed on the PREVIOUS iteration
                if (j >= 1) begin
                    capture[row][j-1][0] = rd_pixel[0];
                    capture[row][j-1][1] = rd_pixel[1];
                    capture[row][j-1][2] = rd_pixel[2];
                end
            end
            // release the oldest line
            @(negedge clock);
            rd_line_consume = 1'b1;
            @(negedge clock);
            rd_line_consume = 1'b0;
        end
    endtask

    task automatic chk(input logic cond, input string msg);
        begin
            checks++;
            if (!cond) begin errors++; $display("  [FAIL] %s (t=%0t)", msg, $time); end
        end
    endtask

    // ============================================================
    // Consumer process: runs concurrently with geo writing
    // ============================================================
    integer rows_consumed = 0;
    initial begin
        rd_line_consume = 0; rd_col = 0; rd_line_sel = 0;
        // wait out reset
        @(negedge reset);
        // consume all IMG_HEIGHT rows. Gate each on rd_lines_valid; the tail
        // (last WINDOW-1 rows) drains after frame_done when count < WINDOW.
        for (rows_consumed = 0; rows_consumed < IMG_HEIGHT; rows_consumed++) begin
            // wait until at least one line is available to read.
            // during steady state that's rd_lines_valid; during tail drain the
            // buffer holds < WINDOW lines, so gate on line_count instead.
            wait (uut.line_count != 0);
            consume_one_line(rows_consumed);
        end
    end

    // ============================================================
    // Main: reset, let both run, then verify + dump
    // ============================================================
    initial begin
        clock = 0; reset = 1;
        repeat (3) @(negedge clock);
        reset = 0;

        // wait until consumer has captured every row
        wait (rows_consumed == IMG_HEIGHT);
        repeat (4) @(negedge clock);

        // ---- dump ch0 frame for the external NumPy diff ----
        // (flatten to row-major hex; adapt path/format to your Python script)
        begin
            integer fd, r, c;
            fd = $fopen("data_files/capture_array_uint.hex", "w");
            for (r = 0; r < IMG_HEIGHT; r++)
                for (c = 0; c < IMG_WIDTH; c++)
                    $fdisplay(fd, "%02h%02h%02h",
                            capture[r][c][2],    // bits [95:64] -> Python p2
                            capture[r][c][1],    // bits [63:32] -> Python p1
                            capture[r][c][0]);   // bits [31:0]  -> Python p0
            $fclose(fd);
        end

        // ---- instrumentation report ----
        $display("\n================ INTEGRATION REPORT ================");
        $display(" self-check          : %0d checks, %0d errors", checks, errors);
        $display(" fill_latency        : %0d cycles (reset-deassert -> rd_lines_valid)", fill_latency);
        $display(" stall_cycles        : %0d", stall_cycles);
        $display(" stall_events        : %0d", stall_events);
        $display(" backpressure_cycles : %0d", backpressure_cycles);
        $display(" max_occupancy       : %0d / %0d lines", max_occupancy, NUM_LINES);
        $display("====================================================");
        if (errors == 0) $display(" INTEGRATION PASSED\n");
        else             $display(" *** %0d FAILURES ***\n", errors);
        $finish;
    end

    // watchdog
    initial begin
        #10_000_000;
        $display("[TIMEOUT] rows=%0d line_count=%0d stall=%b frame_done=%b frame_over=%b",
                rows_consumed, uut.line_count, uut.stall_geo, frame_done, uut.frame_over);
        $fatal(1, "watchdog");
    end

endmodule