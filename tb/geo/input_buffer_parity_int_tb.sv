`timescale 1ns/1ps

// input_buffer_parity_int_tb.sv
//
// Ported from the FP32 version. Three interface changes:
//   BIT_WIDTH_FP parameter    -> IND_WIDTH
//   addr_* 2*32 bits          -> 2*IND_WIDTH bits
//   pack_addr $shortrealtobits-> plain concatenation
//
// The test STRUCTURE is unchanged, because it was already the right shape:
//
//   - the full-image sweep drives a proper 2x2 quad, which by construction
//     has exactly ONE coordinate per parity. That is the case the address
//     and output crossbars are designed for.
//   - the clamped-edge quads drive DEGENERATE quads where several coords
//     share a parity. That is where a crossbar bug actually lives: with two
//     coords wanting the same bank, "last case wins" has to land on the right
//     one, and the output side has to read it back through the right select.
//   - the corner cases collapse all four coords onto one, a 4-way collision.
//
// Both crossbars are exercised end to end by comparing against the golden
// image, so a mis-route shows up as a wrong pixel rather than needing the
// internal selects to be probed.

module input_buffer_parity_int_tb();

    // params for DUT and golden ref
    parameter int BIT_WIDTH_UINT = 8;
    parameter int IND_WIDTH      = 8;      // was BIT_WIDTH_FP
    parameter int IMG_WIDTH      = 256;
    parameter int IMG_HEIGHT     = 256;

    localparam NUM_PIXELS = IMG_WIDTH * IMG_HEIGHT;
    localparam UINT8_MAX  = (1 << BIT_WIDTH_UINT) - 1;

    // DUT read latency: ROM word register -> channel split register = 2 cycles
    localparam int LATENCY = 2;

    parameter IMAGE_FILE = "data_files/image.hex";
    parameter IMAGE_EE   = "data_files/rom_ee.hex";
    parameter IMAGE_EO   = "data_files/rom_eo.hex";
    parameter IMAGE_OE   = "data_files/rom_oe.hex";
    parameter IMAGE_OO   = "data_files/rom_oo.hex";

    logic clock;
    logic reset;
    logic stall;

    // packed {y, x}, integer now
    logic [2*IND_WIDTH-1:0] addr_00, addr_01, addr_10, addr_11;

    // outputs are raw uint8 channels now (dequant LUT removed from RTL)
    logic [BIT_WIDTH_UINT-1:0] P_00 [2:0], P_01 [2:0], P_10 [2:0], P_11 [2:0];

    int errors = 0, checks = 0;
    logic [BIT_WIDTH_UINT-1:0] held [2:0];      // <-- add here

    input_buffer_parity_int #(
        .BIT_WIDTH_UINT(BIT_WIDTH_UINT),
        .IND_WIDTH     (IND_WIDTH),
        .IMG_WIDTH     (IMG_WIDTH),
        .IMG_HEIGHT    (IMG_HEIGHT),
        .IMAGE_EE      (IMAGE_EE),
        .IMAGE_EO      (IMAGE_EO),
        .IMAGE_OE      (IMAGE_OE),
        .IMAGE_OO      (IMAGE_OO)
    ) uut (
        .clock  (clock),
        .reset  (reset),
        .stall  (stall),
        .addr_00(addr_00),
        .addr_01(addr_01),
        .addr_10(addr_10),
        .addr_11(addr_11),
        .P_00   (P_00),
        .P_01   (P_01),
        .P_10   (P_10),
        .P_11   (P_11)
    );

    // golden image (same bytes the DUT banks were built from)
    logic [BIT_WIDTH_UINT-1:0] gold_img [0:NUM_PIXELS-1][0:2];
    initial $readmemh(IMAGE_FILE, gold_img);

    initial clock = '0;
    always #5 clock = ~clock;

    // pack {y, x} -- y in the high half, matching ffn_int
    function automatic logic [2*IND_WIDTH-1:0] pack_addr(input int x, input int y);
        return {IND_WIDTH'(y), IND_WIDTH'(x)};
    endfunction

    // drive the four corner coords, then wait out the pipeline latency
    task automatic do_read(input logic [2*IND_WIDTH-1:0] addr00_i,
                           input logic [2*IND_WIDTH-1:0] addr01_i,
                           input logic [2*IND_WIDTH-1:0] addr10_i,
                           input logic [2*IND_WIDTH-1:0] addr11_i);
        addr_00 = addr00_i;
        addr_01 = addr01_i;
        addr_10 = addr10_i;
        addr_11 = addr11_i;

        repeat (LATENCY) @(posedge clock);
        #1;    // small delay so outputs are stable before checking
    endtask

    // compare one corner's 3 channels against the golden image (exact, uint8)
    task automatic check_corner(input string name,
                                input logic [3*BIT_WIDTH_UINT-1:0] got,   // {ch0,ch1,ch2}
                                input int idx);
        logic [BIT_WIDTH_UINT-1:0] exp_c, got_c;
        for (int i = 0; i < 3; i++) begin
            exp_c = gold_img[idx][i];
            got_c = got[BIT_WIDTH_UINT*(2-i) +: BIT_WIDTH_UINT];
            if (got_c !== exp_c) begin
                errors++;
                if (errors <= 20)
                    $display("ERROR: %0s ch%0d @idx=%0d (x=%0d,y=%0d): got=%0d exp=%0d",
                             name, i, idx, idx % IMG_WIDTH, idx / IMG_WIDTH, got_c, exp_c);
                if (errors == 20)
                    $display("  ... further errors suppressed");
            end
        end
    endtask

    task automatic do_check(input int x00, input int y00,
                            input int x01, input int y01,
                            input int x10, input int y10,
                            input int x11, input int y11);
        int idx00, idx01, idx10, idx11;

        do_read(pack_addr(x00, y00), pack_addr(x01, y01),
                pack_addr(x10, y10), pack_addr(x11, y11));

        idx00 = y00*IMG_WIDTH + x00;
        idx01 = y01*IMG_WIDTH + x01;
        idx10 = y10*IMG_WIDTH + x10;
        idx11 = y11*IMG_WIDTH + x11;

        checks++;
        check_corner("P_00", {P_00[0], P_00[1], P_00[2]}, idx00);
        check_corner("P_01", {P_01[0], P_01[1], P_01[2]}, idx01);
        check_corner("P_10", {P_10[0], P_10[1], P_10[2]}, idx10);
        check_corner("P_11", {P_11[0], P_11[1], P_11[2]}, idx11);
    endtask

    initial begin
        reset = 1;
        stall = 0;
        addr_00 = '0; addr_01 = '0; addr_10 = '0; addr_11 = '0;
        repeat (2) @(posedge clock);
        reset = 0;
        #1;

        // ---- single manual read, sanity print ----
        addr_00 = pack_addr(5, 3);
        addr_01 = pack_addr(6, 3);
        addr_10 = pack_addr(5, 4);
        addr_11 = pack_addr(6, 4);
        repeat (LATENCY) @(posedge clock);
        #1;

        $display("P_00=%0d P_01=%0d P_10=%0d P_11=%0d",
                 P_00[0], P_01[0], P_10[0], P_11[0]);
        $display("expect: (5,3)=%0d (6,3)=%0d (5,4)=%0d (6,4)=%0d",
                 gold_img[3*IMG_WIDTH+5][0], gold_img[3*IMG_WIDTH+6][0],
                 gold_img[4*IMG_WIDTH+5][0], gold_img[4*IMG_WIDTH+6][0]);

        // ---- stall check: outputs must freeze while stall is high ----
        $display("--- stall hold check ---");
        begin
            for (int i = 0; i < 3; i++) held[i] = P_00[i];

            stall = 1;
            // change the inputs underneath the stall; nothing should move
            addr_00 = pack_addr(100, 100);
            addr_01 = pack_addr(101, 100);
            addr_10 = pack_addr(100, 101);
            addr_11 = pack_addr(101, 101);

            repeat (4) @(posedge clock);
            #1;
            for (int i = 0; i < 3; i++) begin
                checks++;
                assert (P_00[i] === held[i])
                    else begin
                        errors++;
                        $error("stall ch%0d: output changed, got=%0d held=%0d", i, P_00[i], held[i]);
                    end
            end

            stall = 0;
            repeat (LATENCY) @(posedge clock);
            #1;
            checks++;
            check_corner("P_00 post-stall", {P_00[0], P_00[1], P_00[2]}, 100*IMG_WIDTH + 100);
            check_corner("P_11 post-stall", {P_11[0], P_11[1], P_11[2]}, 101*IMG_WIDTH + 101);
        end

        // ---- degenerate / clamped quads: corners collapse onto the same coord ----
        // These are the interesting ones. A proper quad has one coord per
        // parity so both crossbars are trivially unambiguous; here several
        // coords want the SAME bank and the routing has to still be right.
        $display("--- clamped edge quads ---");

        // right edge: x_hi clamps to x_lo
        for (int y = 0; y < IMG_HEIGHT - 1; y++)
            do_check(255, y,  255, y,  255, y+1,  255, y+1);

        // bottom edge: y_hi clamps to y_lo
        for (int x = 0; x < IMG_WIDTH - 1; x++)
            do_check(x, 255,  x+1, 255,  x, 255,  x+1, 255);

        // four corners: all four coords identical (worst case -- 4-way parity collision)
        do_check(0,   0,     0,   0,     0,   0,     0,   0  );
        do_check(255, 0,     255, 0,     255, 0,     255, 0  );
        do_check(0,   255,   0,   255,   0,   255,   0,   255);
        do_check(255, 255,   255, 255,   255, 255,   255, 255);

        // ---- full image sweep ----
        // 255 x 255 quads, 2 clocks each -- expect roughly a minute.
        $display("--- full image sweep ---");
        for (int y = 0; y < IMG_HEIGHT - 1; y++)
            for (int x = 0; x < IMG_WIDTH - 1; x++)
                do_check(x, y,  x+1, y,  x, y+1,  x+1, y+1);

        if (errors == 0) $display("PASS, 0 errors, checks = %0d", checks);
        else             $display("FAIL, %0d errors, checks = %0d", errors, checks);

        if (errors != 0) $fatal(1, "input_buffer_parity_int_tb failed");
        $finish;
    end

endmodule