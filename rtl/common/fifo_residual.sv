// fifo.sv
// Synchronous FIFO, first-word-fall-through (FWFT) style read.
// Holds DEPTH entries of WIDTH-bit data. Standard full/empty handshake.
//
// Handshake:
//   push: assert wr_en when !full; data latched on the clock edge.
//   pop:  assert rd_en when !empty; rd_data is the head, consumed on the edge.
//
// full/empty tracked with a count register (simplest correct scheme;
// avoids the pointer-wraparound ambiguity of same-address full-vs-empty).

module fifo_residual #(
    parameter int WIDTH = 32,
    parameter int DEPTH = 4               // power of two recommended
) (
    input  logic clock,
    input  logic reset,

    // write side
    input  logic             wr_en,
    input  logic [WIDTH-1:0] wr_data,
    output logic             full,

    // read side
    input  logic             rd_en,
    output logic [WIDTH-1:0] rd_data,
    output logic             empty
);

    localparam int AW = $clog2(DEPTH);

    logic [WIDTH-1:0] mem [DEPTH-1:0];
    logic [AW-1:0]    wr_ptr, rd_ptr;
    logic [AW:0]      count;              // 0..DEPTH, needs one extra bit

    assign full  = (count == DEPTH);
    assign empty = (count == 0);

    // read data is combinational head-of-queue (first-word-fall-through):
    // rd_data is valid whenever !empty, no read latency.
    assign rd_data = mem[rd_ptr];

    // accepted transfers this cycle
    wire do_wr = wr_en && !full;
    wire do_rd = rd_en && !empty;

    always_ff @(posedge clock) begin
        if (reset) begin
            wr_ptr <= '0;
            rd_ptr <= '0;
            count  <= '0;
        end else begin
            // write
            if (do_wr) begin
                mem[wr_ptr] <= wr_data;
                wr_ptr <= (wr_ptr == DEPTH-1) ? '0 : wr_ptr + 1'b1;
            end
            // read
            if (do_rd) begin
                rd_ptr <= (rd_ptr == DEPTH-1) ? '0 : rd_ptr + 1'b1;
            end
            // count update (handles simultaneous push+pop correctly)
            case ({do_wr, do_rd})
                2'b10: count <= count + 1'b1;   // push only
                2'b01: count <= count - 1'b1;   // pop only
                default: count <= count;         // both or neither: net zero
            endcase
        end
    end

endmodule