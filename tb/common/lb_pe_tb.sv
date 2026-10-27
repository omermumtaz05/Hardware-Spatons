`timescale 1ns/1ps
module lb_pe_tb();

    localparam int BW = 32, IMG_WIDTH = 8, IMG_HEIGHT = 6, NUM_LINES = 4;
    localparam int XW = $clog2(IMG_WIDTH), LC_W = $clog2(NUM_LINES+1);

    logic clock = 0, reset = 1;
    always #5 clock = ~clock;

    logic wr_line_valid, wr_line_ready, wr_en_c, last_col;
    logic [XW-1:0] wr_col_r;
    int wr_row;
    logic [BW-1:0] wr_pixel [3];

    logic rd_lines_valid, rd_line_consume, out_valid, out_ready, frame_done;
    logic [XW-1:0] rd_col;
    logic [$clog2(NUM_LINES)-1:0] rd_line_sel;
    logic [BW-1:0] rd_pixel [3], f [28:0];
    logic [LC_W-1:0] line_count;

    int errors, checked, exp_x, exp_y;

    // ---- writer (geo model) ----
    assign last_col      = (wr_col_r == XW'(IMG_WIDTH-1));
    assign wr_en_c       = (wr_row < IMG_HEIGHT) && (!last_col || wr_line_ready);
    assign wr_line_valid = wr_en_c && last_col;
    assign wr_pixel[0]   = 32'd0      + wr_row*1000 + wr_col_r;
    assign wr_pixel[1]   = 32'd100000 + wr_row*1000 + wr_col_r;
    assign wr_pixel[2]   = 32'd200000 + wr_row*1000 + wr_col_r;

    always @(posedge clock) begin
        if (reset) begin wr_col_r <= '0; wr_row <= 0; end
        else if (wr_en_c) begin
            if (last_col) begin wr_col_r <= '0; wr_row <= wr_row + 1; end
            else wr_col_r <= wr_col_r + 1'b1;
        end
    end

    inter_line_buffer #(.BIT_WIDTH_FP(BW), .IMG_WIDTH(IMG_WIDTH),
                        .NUM_LINES(NUM_LINES), .WINDOW(3)) u_buf (
        .clock(clock), .reset(reset),
        .wr_line_valid(wr_line_valid), .wr_line_ready(wr_line_ready),
        .wr_pixel_en(wr_en_c), .wr_col(wr_col_r), .wr_pixel(wr_pixel),
        .rd_lines_valid(rd_lines_valid), .rd_line_consume(rd_line_consume),
        .rd_col(rd_col), .rd_line_sel(rd_line_sel), .rd_pixel(rd_pixel),
        .line_count(line_count)
    );

    patch_extractor #(.BW(BW), .IMG_WIDTH(IMG_WIDTH), .IMG_HEIGHT(IMG_HEIGHT),
                      .NUM_LINES(NUM_LINES), .USE_COORD_FILE(0)) u_pe (
        .clock(clock), .reset(reset),
        .line_count(line_count), .rd_line_consume(rd_line_consume),
        .rd_col(rd_col), .rd_line_sel(rd_line_sel), .rd_pixel(rd_pixel),
        .out_valid(out_valid), .out_ready(out_ready), .f(f),
        .frame_done(frame_done)
    );

    always_ff @(posedge clock) out_ready <= reset ? 1'b0 : ($urandom_range(0,2) != 0);

    // absolute row after reflection
    function automatic int abs_row(input int yy, input int wr);
        if (yy == 0)                 abs_row = (wr == 1) ? 0 : 1;
        else if (yy == IMG_HEIGHT-1) abs_row = (wr == 1) ? IMG_HEIGHT-1 : IMG_HEIGHT-2;
        else                         abs_row = yy - 1 + wr;
    endfunction

    function automatic int abs_col(input int xx, input int wc);
        if (wc == 0)      abs_col = (xx == 0) ? 1 : xx-1;
        else if (wc == 1) abs_col = xx;
        else              abs_col = (xx == IMG_WIDTH-1) ? IMG_WIDTH-2 : xx+1;
    endfunction

    task automatic chk(input int base, input int wr, input int wc, input string nm);
        int e;
        for (int c = 0; c < 3; c++) begin
            e = c*100000 + abs_row(exp_y,wr)*1000 + abs_col(exp_x,wc);
            checked++;
            if (f[base+c] !== e) begin
                errors++;
                $display("ERR (%0d,%0d) %s ch%0d: got %0d exp %0d",
                         exp_x, exp_y, nm, c, f[base+c], e);
            end
        end
    endtask

    always @(posedge clock) begin
        if (!reset && out_valid && out_ready) begin
            chk(2,  1,1,"C");  chk(5,  0,0,"NW"); chk(8,  0,1,"N");
            chk(11, 0,2,"NE"); chk(14, 1,0,"W");  chk(17, 1,2,"E");
            chk(20, 2,0,"SW"); chk(23, 2,1,"S");  chk(26, 2,2,"SE");
            checked += 2;
            if (f[0] !== exp_x) begin errors++; $display("ERR f[0] %0d exp %0d", f[0], exp_x); end
            if (f[1] !== exp_y) begin errors++; $display("ERR f[1] %0d exp %0d", f[1], exp_y); end
            if (exp_x == IMG_WIDTH-1) begin exp_x <= 0; exp_y <= exp_y + 1; end
            else exp_x <= exp_x + 1;
        end
    end

    initial begin
        errors = 0; checked = 0; exp_x = 0; exp_y = 0;
        repeat (4) @(posedge clock);
        reset <= 1'b0;
        fork
            begin wait (frame_done); repeat (5) @(posedge clock); end
            begin
                repeat (200000) @(posedge clock);
                $display("TIMEOUT at (%0d,%0d) line_count=%0d wr_row=%0d",
                         exp_x, exp_y, line_count, wr_row);
                errors++;
            end
        join_any
        disable fork;
        if (exp_y != IMG_HEIGHT || exp_x != 0) begin
            $display("ERR incomplete: stopped at (%0d,%0d)", exp_x, exp_y);
            errors++;
        end
        $display("checked=%0d errors=%0d  %s", checked, errors,
                 (errors==0) ? "PASS" : "FAIL");
        $finish;
    end

endmodule