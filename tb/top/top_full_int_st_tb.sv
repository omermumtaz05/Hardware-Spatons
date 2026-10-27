`timescale 1ns/1ps
module top_full_int_st_tb();

    localparam int BIT_WIDTH_FP = 32;
    localparam int BIT_WIDTH_UINT = 8;

    localparam int IND_WIDTH    = 8;
    localparam int IMG_WIDTH    = 256;
    localparam int IMG_HEIGHT   = 256;
    localparam int NUM_LINES    = 4;
    localparam int WINDOW       = 3;
    localparam int NPIX         = IMG_WIDTH*IMG_HEIGHT;
    localparam int TIMEOUT      = NPIX*200;

    logic clock=0, reset=1;
    always #6.495 clock=~clock;

    logic frame_done, out_valid, out_ready;
    logic [BIT_WIDTH_UINT-1:0] final_rgb [2:0];

    top_full_int_st #(
        .BIT_WIDTH_FP(BIT_WIDTH_FP), .BIT_WIDTH_UINT(8), .IND_WIDTH(IND_WIDTH),
        .IMG_WIDTH(IMG_WIDTH), .IMG_HEIGHT(IMG_HEIGHT),
        .NUM_LINES(NUM_LINES), .WINDOW(WINDOW), .IN_DIM(29)
    ) uut (
        .clock(clock), .reset(reset), .frame_done(frame_done),
        .out_valid(out_valid), .out_ready(out_ready), .final_rgb(final_rgb)
    );

    logic [15:0] lfsr;
    int sink_stall=0, idle=0;
    always @(posedge clock)
        if (reset) lfsr<=16'hACE1;
        else lfsr<={lfsr[14:0], lfsr[15]^lfsr[13]^lfsr[12]^lfsr[10]};
    assign out_ready = !reset && lfsr[0];   // ~50% random stalls

    integer fout; int out_count=0, cyc=0; bit done=0;
    initial fout=$fopen("data_files/full_sys_st.hex","w");
    //initial fout=$fopen("data_files/full_sys_vcd_run.hex","w");
    always @(posedge clock) begin
        if (!reset) begin
            cyc <= cyc+1;
            if (out_valid && out_ready) begin
                $fwrite(fout, "%02x\n%02x\n%02x\n",  // WAS %08x
                        final_rgb[0], final_rgb[1], final_rgb[2]);
                out_count <= out_count+1;
            end
            if (out_count==NPIX && !done) begin
                done<=1; $fclose(fout);
                $display("DONE out_count=%0d/%0d cyc=%0d  sink_stall=%0d idle=%0d",
                out_count, NPIX, cyc, sink_stall, idle);
                $finish;
            end
            if (cyc>TIMEOUT) begin
                $fclose(fout);
                $display("TIMEOUT out_count=%0d/%0d", out_count, NPIX);
                $finish;
            end
            if (out_valid && !out_ready) sink_stall <= sink_stall+1;
            if (!out_valid)              idle       <= idle+1;
        end
    end

    initial begin
        repeat(3) @(posedge clock);
        reset<=0;
    end
    // ---- VCD for power analysis ----
    // ---- VCD capture run. Comment out for normal sims. ----
    /*
    initial begin
        $dumpfile("power_new.vcd");
        wait (out_count == 200);            // let the pipeline reach steady state
        $dumpvars(0, uut);     // MLP only -- the part that changed
        repeat (10000) @(posedge clock);   // ~270 pixels
        $display("VCD window captured");
        $finish;                            // stop immediately, nothing more written
    end*/

/*
    initial begin
        $dumpfile("learn.vcd");
        $dumpvars(0, uut);
        repeat (3000) @(posedge clock);
        $display("learn VCD captured, out_count=%0d", out_count);
        $finish;
    end*/
endmodule