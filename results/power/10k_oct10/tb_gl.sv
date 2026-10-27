`timescale 1ns/1ps
module tb();
    localparam int WARMUP_PIX = 1280;    // ~5 rows: past reset + line-buffer fill
    localparam int WINDOW_CYC = 10000;   // ~270 px @ 37 cyc/px

    logic clock = 0, reset = 1;
    always #6.25 clock = ~clock;         // 80 MHz

    logic frame_done, out_valid, out_ready;
    logic [23:0] final_rgb_flat;

    top_full_synth dut (
        .clock(clock), .reset(reset), .out_ready(out_ready),
        .out_valid(out_valid), .frame_done(frame_done),
        .final_rgb_flat(final_rgb_flat)
    );

    // same sink-stall pattern as RTL TB
    logic [15:0] lfsr;
    always @(posedge clock)
        if (reset) lfsr <= 16'hACE1;
        else       lfsr <= {lfsr[14:0], lfsr[15]^lfsr[13]^lfsr[12]^lfsr[10]};
    assign out_ready = !reset && lfsr[0];

    logic out_ready_d, out_ready_dd;
    always @(posedge clock) begin
        out_ready_d  <= out_ready;      // matches wrapper's out_ready_r
        out_ready_dd <= out_ready_d;    // matches wrapper's registered outputs
    end
    // dump outputs to compare against RTL
    int out_count = 0;
    integer fout;
    initial fout = $fopen("gl_out.hex", "w");
    always @(posedge clock)
        if (!reset && out_valid && out_ready_dd) begin
            $fwrite(fout, "%02x\n%02x\n%02x\n",
                    final_rgb_flat[7:0], final_rgb_flat[15:8], final_rgb_flat[23:16]);
            out_count <= out_count + 1;
        end

    logic vcd_en = 0;                    // do-file turns VCD on when this goes high
    initial begin
        repeat (20) @(posedge clock);
        reset <= 0;
        wait (out_count == WARMUP_PIX);
        @(posedge clock) vcd_en = 1;
        repeat (WINDOW_CYC) @(posedge clock);
        $display("VCD window done, out_count=%0d", out_count);
        $fclose(fout);
        $finish;
    end

    // watchdog: catches an X'd or dead netlist
    initial begin
        repeat (300000) @(posedge clock);
        if (!vcd_en) begin
            $display("TIMEOUT out_count=%0d", out_count);
            $finish;
        end
    end
endmodule