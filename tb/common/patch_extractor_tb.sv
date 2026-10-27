`timescale 1ns/1ps
module patch_extractor_tb();

    localparam int BW = 32, IMG_WIDTH = 8, IMG_HEIGHT = 6, NUM_LINES = 4;
    localparam int LC_W = $clog2(NUM_LINES+1);

    logic clock = 0, reset = 1;
    always #5 clock = ~clock;

    logic [LC_W-1:0] line_count;
    logic rd_line_consume, out_valid, out_ready, frame_done;
    logic [$clog2(IMG_WIDTH)-1:0] rd_col;
    logic [$clog2(NUM_LINES)-1:0] rd_line_sel;
    logic [BW-1:0] rd_pixel [3];
    logic [BW-1:0] f [28:0];

    int lines_written, errors, checked;
    int exp_x, exp_y;
    logic [BW-1:0] f_hold [28:0];
    logic holding;

    patch_extractor #(
        .BW(BW), .IMG_WIDTH(IMG_WIDTH), .IMG_HEIGHT(IMG_HEIGHT),
        .NUM_LINES(NUM_LINES), .USE_COORD_FILE(0)
    ) dut (
        .clock(clock), .reset(reset),
        .line_count(line_count), .rd_line_consume(rd_line_consume),
        .rd_col(rd_col), .rd_line_sel(rd_line_sel), .rd_pixel(rd_pixel),
        .out_valid(out_valid), .out_ready(out_ready), .f(f),
        .frame_done(frame_done)
    );

    // fake buffer: 1-cycle registered read
    always_ff @(posedge clock) begin
        rd_pixel[0] <= 32'd0      + rd_line_sel*1000 + rd_col;
        rd_pixel[1] <= 32'd100000 + rd_line_sel*1000 + rd_col;
        rd_pixel[2] <= 32'd200000 + rd_line_sel*1000 + rd_col;
    end

    // line_count model: greedy producer + consume from DUT
    logic do_wr;
    always_comb do_wr = (lines_written < IMG_HEIGHT) && (line_count < NUM_LINES);

    always @(posedge clock) begin
        if (reset) begin
            line_count <= '0; lines_written <= 0;
        end else begin
            if (do_wr) lines_written <= lines_written + 1;
            unique case ({do_wr, rd_line_consume})
                2'b10: line_count <= line_count + 1'b1;
                2'b01: line_count <= line_count - 1'b1;
                default: ;
            endcase
        end
    end

    // random backpressure
    always_ff @(posedge clock) out_ready <= reset ? 1'b0 : ($urandom_range(0,2) != 0);

    function automatic int exp_sel(input int yy, input int wr);
        if (yy == 0)                 exp_sel = (wr == 1) ? 0 : 1;
        else if (yy == IMG_HEIGHT-1) exp_sel = (wr == 1) ? 1 : 0;
        else                         exp_sel = wr;
    endfunction

    function automatic int exp_col(input int xx, input int wc);
        if (wc == 0)      exp_col = (xx == 0) ? 1 : xx-1;
        else if (wc == 1) exp_col = xx;
        else              exp_col = (xx == IMG_WIDTH-1) ? IMG_WIDTH-2 : xx+1;
    endfunction

    function automatic int exp_v(input int yy, input int xx,
                                 input int wr, input int wc, input int ch);
        exp_v = ch*100000 + exp_sel(yy,wr)*1000 + exp_col(xx,wc);
    endfunction

    task automatic chk(input int base, input int wr, input int wc, input string nm);
        for (int c = 0; c < 3; c++) begin
            checked++;
            if (f[base+c] !== exp_v(exp_y, exp_x, wr, wc, c)) begin
                errors++;
                $display("ERR (%0d,%0d) %s ch%0d: got %0d exp %0d",
                         exp_x, exp_y, nm, c, f[base+c], exp_v(exp_y,exp_x,wr,wc,c));
            end
        end
    endtask

    // stability under backpressure
    always @(posedge clock) begin
        if (!reset) begin
            if (out_valid && !out_ready) begin
                if (holding)
                    for (int i = 0; i < 29; i++)
                        if (f[i] !== f_hold[i]) begin
                            errors++;
                            $display("ERR f[%0d] moved during backpressure", i);
                        end
                for (int i = 0; i < 29; i++) f_hold[i] <= f[i];
                holding <= 1'b1;
            end else holding <= 1'b0;
        end
    end

    always @(posedge clock) begin
        if (!reset && out_valid && out_ready) begin
            chk(2,  1,1, "C");  chk(5,  0,0, "NW"); chk(8,  0,1, "N");
            chk(11, 0,2, "NE"); chk(14, 1,0, "W");  chk(17, 1,2, "E");
            chk(20, 2,0, "SW"); chk(23, 2,1, "S");  chk(26, 2,2, "SE");
            checked += 2;
            if (f[0] !== exp_x) begin errors++; $display("ERR f[0] got %0d exp %0d", f[0], exp_x); end
            if (f[1] !== exp_y) begin errors++; $display("ERR f[1] got %0d exp %0d", f[1], exp_y); end

            if (exp_x == IMG_WIDTH-1) begin exp_x <= 0; exp_y <= exp_y + 1; end
            else exp_x <= exp_x + 1;
        end
    end

    initial begin
        errors = 0; checked = 0; exp_x = 0; exp_y = 0;
        holding = 0; lines_written = 0;
        repeat (4) @(posedge clock);
        reset <= 1'b0;

        fork
            begin
                wait (frame_done);
                repeat (5) @(posedge clock);
            end
            begin
                repeat (200000) @(posedge clock);
                $display("TIMEOUT at pixel (%0d,%0d)", exp_x, exp_y);
                errors++;
            end
        join_any
        disable fork;

        if (exp_y != IMG_HEIGHT || exp_x != 0) begin
            $display("ERR incomplete: stopped at (%0d,%0d)", exp_x, exp_y);
            errors++;
        end
        $display("checked=%0d errors=%0d  %s", checked, errors,
                 (errors == 0) ? "PASS" : "FAIL");
        $finish;
    end

endmodule