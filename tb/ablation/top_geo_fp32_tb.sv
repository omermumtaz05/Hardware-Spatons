module top_geo_fp32_tb();

    localparam int NPIX = 256*256, TIMEOUT = NPIX*20;
    localparam IND_WIDTH = 8;
    localparam BIT_WIDTH_FP   = 32;

    logic clock = 0;
    logic reset = 1;

    always #5 clock = ~clock;

    logic ov, fd, orl;

    logic [IND_WIDTH - 1:0] ox, oy;
    logic [BIT_WIDTH_FP - 1:0] P_out [2:0];


    top_geo uut (.clock(clock), .reset(reset), .stall(1'b0),
                .out_valid(ov), .out_x(ox), .out_y(oy), 
                .P_out(P_out), .frame_done(fd), .out_row_last(orl));

    integer fout; int n = 0, cyc = 0;
    
    initial begin
        fout = $fopen("data_files/capture_geo_fp32.hex", "w");
        repeat (3) @ (posedge clock); reset <= 0;
    end

    always @ (posedge clock)
        if(!reset)
        begin
            cyc <= cyc + 1;
            if(ov)
                begin
                    $fwrite(fout, "%08h%08h%08h\n", P_out[2], P_out[1], P_out[0]);
                    n <= n + 1;

            
                    if((n + 1) == NPIX)
                        begin
                            $fclose(fout);
                            $display("GEO FP32 DONE pixels=%0d cyc=%0d", n + 1, cyc);
                            $finish;
                        end
                end
            if(cyc > TIMEOUT)
                begin
                    $fclose(fout);
                    $display("TIMEOUT AT %0d cycles, %0d pixels", cyc, n);
                    $finish;
                end
        end

endmodule