
module inter_line_buffer #(
    parameter int BIT_WIDTH_FP = 32,
    parameter int IMG_WIDTH    = 256,
    parameter int NUM_LINES    = 4,
    parameter int WINDOW       = 3
) (
    input  logic clock,
    input  logic reset,

    // --- write side (from geo) ---
    input  logic                              wr_line_valid,   // commit current line
    output logic                              wr_line_ready,   // a free slot exists
    input  logic                              wr_pixel_en,     // write one pixel this cycle
    input  logic [$clog2(IMG_WIDTH)-1:0]      wr_col,          // column within line
    input  logic [BIT_WIDTH_FP-1:0]           wr_pixel [3],    // 3 channels

    // --- read side (to photo) ---
    output logic                              rd_lines_valid,  // >= WINDOW lines live
    input  logic                              rd_line_consume, // release oldest line
    input  logic [$clog2(IMG_WIDTH)-1:0]      rd_col,          // column within line
    input  logic [$clog2(NUM_LINES)-1:0]      rd_line_sel,     // which window line (0..WINDOW-1)
    output logic [BIT_WIDTH_FP-1:0]           rd_pixel [3],    // 3 channels (registered)

    // --- observability ---
    output logic [$clog2(NUM_LINES+1)-1:0]    line_count
);

    localparam int LPTR_W = $clog2(NUM_LINES);
    localparam int COL_W  = $clog2(IMG_WIDTH);
    localparam int ADDR_W = LPTR_W + COL_W;          // {line, col}
    localparam int WORD_W = 3 * BIT_WIDTH_FP;        // 3 channels packed

    // ring pointers
    logic [LPTR_W-1:0] wr_line_ptr, rd_line_ptr;

    // handshakes
    localparam int MAX_LIVE = NUM_LINES - 1;
    assign wr_line_ready  = (line_count < MAX_LIVE);
    assign rd_lines_valid = (line_count >= WINDOW[LPTR_W:0]);

    wire do_commit  = wr_line_valid   && wr_line_ready;
    wire do_release = rd_line_consume && (line_count != '0);

    // ---- line memory: ONE flat array, 3 channels packed per word ----
    logic [WORD_W-1:0] mem [0:(1<<ADDR_W)-1];

    // read address: physical line = (rd_line_ptr + rd_line_sel) mod NUM_LINES
    logic [LPTR_W-1:0] rd_phys;
    always_comb begin
        logic [LPTR_W:0] s;
        s = rd_line_ptr + rd_line_sel;
        rd_phys = (s >= NUM_LINES[LPTR_W:0]) ? (s - NUM_LINES[LPTR_W:0]) : s[LPTR_W-1:0];
    end

    wire [ADDR_W-1:0] wr_addr = {wr_line_ptr, wr_col};
    wire [ADDR_W-1:0] rd_addr = {rd_phys,     rd_col};

    // write port: one word, one address
    always_ff @(posedge clock) begin
        if (wr_pixel_en)
            mem[wr_addr] <= {wr_pixel[2], wr_pixel[1], wr_pixel[0]};
    end

    // read port: registered, NO reset (inference rule)
    logic [WORD_W-1:0] rd_word;
    always_ff @(posedge clock)
        rd_word <= mem[rd_addr];

    // unpack -- free, just wire slicing
    always_comb begin
        rd_pixel[0] = rd_word[0*BIT_WIDTH_FP +: BIT_WIDTH_FP];
        rd_pixel[1] = rd_word[1*BIT_WIDTH_FP +: BIT_WIDTH_FP];
        rd_pixel[2] = rd_word[2*BIT_WIDTH_FP +: BIT_WIDTH_FP];
    end

    // ---- occupancy + pointer bookkeeping (unchanged) ----
    always_ff @(posedge clock) begin
        if (reset) begin
            line_count  <= '0;
            wr_line_ptr <= '0;
            rd_line_ptr <= '0;
        end else begin
            if (do_commit)
                wr_line_ptr <= (wr_line_ptr == NUM_LINES[LPTR_W-1:0]-1) ? '0 : wr_line_ptr + 1'b1;
            if (do_release)
                rd_line_ptr <= (rd_line_ptr == NUM_LINES[LPTR_W-1:0]-1) ? '0 : rd_line_ptr + 1'b1;

            unique case ({do_commit, do_release})
                2'b10:   line_count <= line_count + 1'b1;
                2'b01:   line_count <= line_count - 1'b1;
                default: line_count <= line_count;
            endcase
        end
    end

endmodule