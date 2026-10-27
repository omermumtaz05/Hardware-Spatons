import fp32_conv_pkg::*;

module input_buffer_parity #(
    parameter BIT_WIDTH_FP   = 32,
    parameter BIT_WIDTH_UINT = 8,
    parameter IMG_WIDTH  = 256,
    parameter IMG_HEIGHT = 256,

    parameter DEQUANT_LUT = "data_files/dequant_lut.hex",
    parameter IMAGE_EE = "data_files/rom_ee.hex",
    parameter IMAGE_EO = "data_files/rom_eo.hex",
    parameter IMAGE_OE = "data_files/rom_oe.hex",
    parameter IMAGE_OO = "data_files/rom_oo.hex"
) (
    input  logic clock,
    input  logic reset,

    input logic stall,
    // each addr is a full absolute coord {y,x}, already resolved by fetch_four_near
    input  logic [2*BIT_WIDTH_FP-1:0] addr_00, addr_01, addr_10, addr_11,

    output logic [BIT_WIDTH_FP-1:0] P_00 [2:0], P_01 [2:0], P_10 [2:0], P_11 [2:0]
);

    localparam NUM_PIXELS = IMG_WIDTH * IMG_HEIGHT;
    localparam int quart  = NUM_PIXELS/4;        // pixels per bank (128x128 = 16384)
    localparam int HALF_W = IMG_WIDTH/2;         // 128
    localparam UINT8_MAX  = (1 << BIT_WIDTH_UINT) - 1;

    // four quarter banks, 3 channels each
    logic [3*BIT_WIDTH_UINT-1:0] ROM_EE [0:quart-1];
    logic [3*BIT_WIDTH_UINT-1:0] ROM_EO [0:quart-1];
    logic [3*BIT_WIDTH_UINT-1:0] ROM_OE [0:quart-1];
    logic [3*BIT_WIDTH_UINT-1:0] ROM_OO [0:quart-1];

    logic [BIT_WIDTH_FP-1:0] dequant_LUT [0:UINT8_MAX];

// load ROMs from data files
    initial $readmemh(IMAGE_EE,   ROM_EE);
    initial $readmemh(IMAGE_EO,   ROM_EO);
    initial $readmemh(IMAGE_OE,   ROM_OE);
    initial $readmemh(IMAGE_OO,   ROM_OO);
    initial $readmemh(DEQUANT_LUT, dequant_LUT);

    // ---- 1. decode each coordinate to integer x,y ----
    logic [$clog2(IMG_WIDTH)-1:0]  x_ind_00, x_ind_01, x_ind_10, x_ind_11;
    logic [$clog2(IMG_HEIGHT)-1:0] y_ind_00, y_ind_01, y_ind_10, y_ind_11;

    assign x_ind_00 = fp32_to_int(addr_00[  BIT_WIDTH_FP-1:0]);
    assign y_ind_00 = fp32_to_int(addr_00[2*BIT_WIDTH_FP-1:BIT_WIDTH_FP]);
    assign x_ind_01 = fp32_to_int(addr_01[  BIT_WIDTH_FP-1:0]);
    assign y_ind_01 = fp32_to_int(addr_01[2*BIT_WIDTH_FP-1:BIT_WIDTH_FP]);
    assign x_ind_10 = fp32_to_int(addr_10[  BIT_WIDTH_FP-1:0]);
    assign y_ind_10 = fp32_to_int(addr_10[2*BIT_WIDTH_FP-1:BIT_WIDTH_FP]);
    assign x_ind_11 = fp32_to_int(addr_11[  BIT_WIDTH_FP-1:0]);
    assign y_ind_11 = fp32_to_int(addr_11[2*BIT_WIDTH_FP-1:BIT_WIDTH_FP]);

    // ---- 2. per coord: parity (which bank) + half_addr (slot in bank) ----
    // parity code: {y[0], x[0]}  -> 00=EE 01=EO 10=OE 11=OO
    logic [1:0] par_00, par_01, par_10, par_11;
    assign par_00 = {y_ind_00[0], x_ind_00[0]};
    assign par_01 = {y_ind_01[0], x_ind_01[0]};
    assign par_10 = {y_ind_10[0], x_ind_10[0]};
    assign par_11 = {y_ind_11[0], x_ind_11[0]};

    // half_addr = (y>>1)*128 + (x>>1)  -> index into the 128x128 quarter image
    logic [$clog2(quart)-1:0] half_00, half_01, half_10, half_11;
    assign half_00 = (y_ind_00>>1)*HALF_W + (x_ind_00>>1);
    assign half_01 = (y_ind_01>>1)*HALF_W + (x_ind_01>>1);
    assign half_10 = (y_ind_10>>1)*HALF_W + (x_ind_10>>1);
    assign half_11 = (y_ind_11>>1)*HALF_W + (x_ind_11>>1);

    // ---- 3. ADDRESS crossbar (selection form) ----
    // each bank grabs whichever coord carries its parity.
    // a valid quad has exactly one coord per parity, so each select is unique.
    localparam logic [1:0] EE=2'b00, EO=2'b01, OE=2'b10, OO=2'b11;

    // ---- ADDRESS crossbar: each bank picks the coord matching its parity ----
    logic [$clog2(quart)-1:0] addr_EE, addr_EO, addr_OE, addr_OO;

    always_comb begin
        // default to half_00 (any valid coord) to avoid latch/X
        addr_EE = half_00; addr_EO = half_00; addr_OE = half_00; addr_OO = half_00;

        // route each coord to the bank named by its parity
        case (par_00) EE:addr_EE=half_00; EO:addr_EO=half_00; OE:addr_OE=half_00; OO:addr_OO=half_00; endcase
        case (par_01) EE:addr_EE=half_01; EO:addr_EO=half_01; OE:addr_OE=half_01; OO:addr_OO=half_01; endcase
        case (par_10) EE:addr_EE=half_10; EO:addr_EO=half_10; OE:addr_OE=half_10; OO:addr_OO=half_10; endcase
        case (par_11) EE:addr_EE=half_11; EO:addr_EO=half_11; OE:addr_OE=half_11; OO:addr_OO=half_11; endcase
    end


    // ---- 4. synchronous reads (1 cycle) + register parities to align output ----
    logic [1:0] par_00_r, par_01_r, par_10_r, par_11_r;
    logic [1:0] par_00_r2, par_01_r2, par_10_r2, par_11_r2;

    // reading full words then splitting later into colors to avoid three reads in one cycle!
    logic [3*BIT_WIDTH_UINT-1:0] word_EE, word_EO, word_OE, word_OO;

    always_ff @(posedge clock) begin
        if(!stall)
            begin
                        word_EE <= ROM_EE[addr_EE];
                        word_EO <= ROM_EO[addr_EO];
                        word_OE <= ROM_OE[addr_OE];
                        word_OO <= ROM_OO[addr_OO];
                    
                    // remember which bank each corner went to, aligned with the read latency
                    par_00_r <= par_00;
                    par_01_r <= par_01;
                    par_10_r <= par_10;
                    par_11_r <= par_11;

                    par_00_r2 <= par_00_r;
                    par_01_r2 <= par_01_r;
                    par_10_r2 <= par_10_r;
                    par_11_r2 <= par_11_r;
                end
    end


//split words combinationally
    logic [BIT_WIDTH_FP-1:0] deq_EE [2:0], deq_EO [2:0], deq_OE [2:0], deq_OO [2:0];

    always_ff @(posedge clock) begin
        if (!stall) begin
            for (int c = 0; c < 3; c++) begin
                deq_EE[c] <= dequant_LUT[ word_EE[BIT_WIDTH_UINT*(2-c) +: BIT_WIDTH_UINT] ];
                deq_EO[c] <= dequant_LUT[ word_EO[BIT_WIDTH_UINT*(2-c) +: BIT_WIDTH_UINT] ];
                deq_OE[c] <= dequant_LUT[ word_OE[BIT_WIDTH_UINT*(2-c) +: BIT_WIDTH_UINT] ];
                deq_OO[c] <= dequant_LUT[ word_OO[BIT_WIDTH_UINT*(2-c) +: BIT_WIDTH_UINT] ];
            end
        end
end
    // ---- OUTPUT crossbar: each corner reads back from its bank ----
    always_comb begin
        for (int i = 0; i < 3; i++) begin
            // P_00 reads whichever bank par_00_r points to
            case (par_00_r2) EE:P_00[i]=deq_EE[i]; EO:P_00[i]=deq_EO[i]; OE:P_00[i]=deq_OE[i]; OO:P_00[i]=deq_OO[i]; endcase
            case (par_01_r2) EE:P_01[i]=deq_EE[i]; EO:P_01[i]=deq_EO[i]; OE:P_01[i]=deq_OE[i]; OO:P_01[i]=deq_OO[i]; endcase
            case (par_10_r2) EE:P_10[i]=deq_EE[i]; EO:P_10[i]=deq_EO[i]; OE:P_10[i]=deq_OE[i]; OO:P_10[i]=deq_OO[i]; endcase
            case (par_11_r2) EE:P_11[i]=deq_EE[i]; EO:P_11[i]=deq_EO[i]; OE:P_11[i]=deq_OE[i]; OO:P_11[i]=deq_OO[i]; endcase
        end
    end

endmodule