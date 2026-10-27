// fifo_tb.sv
// Self-checking TB for the synchronous FWFT FIFO.
// Uses a SystemVerilog queue as the golden reference model: every push/pop on
// the DUT is mirrored on the queue, and rd_data is checked against the queue head.
// Covers: basic push/pop, fill-to-full, drain-to-empty, full/empty flags,
// simultaneous push+pop, and wraparound (write past DEPTH).

module fifo_residual_tb();

    localparam int WIDTH = 32;
    localparam int DEPTH = 4;

    logic clock, reset;
    logic             wr_en, rd_en;
    logic [WIDTH-1:0] wr_data, rd_data;
    logic             full, empty;

    fifo #(.WIDTH(WIDTH), .DEPTH(DEPTH)) dut (
        .clock(clock), .reset(reset),
        .wr_en(wr_en), .wr_data(wr_data), .full(full),
        .rd_en(rd_en), .rd_data(rd_data), .empty(empty)
    );

    initial clock = 0;
    always #5 clock = ~clock;

    // golden model
    logic [WIDTH-1:0] model [$];      // SV queue
    int errors = 0, checks = 0;

    // push one word (only if not full); mirror to model
    task automatic push(input logic [WIDTH-1:0] d);
        @(negedge clock);              // drive on negedge, sample on posedge
        if (full) begin
            $display("[INFO] push skipped, FIFO full (d=%h)", d);
            return;
        end
        wr_en = 1'b1; wr_data = d;
        model.push_back(d);
        @(posedge clock);              // transfer happens here
        @(negedge clock);
        wr_en = 1'b0;
    endtask

    // pop one word (only if not empty); check against model head
    task automatic pop();
        logic [WIDTH-1:0] expected;
        @(negedge clock);
        if (empty) begin
            $display("[INFO] pop skipped, FIFO empty");
            return;
        end
        // FWFT: rd_data should already equal model head BEFORE the pop edge
        expected = model[0];
        checks++;
        if (rd_data !== expected) begin
            errors++;
            $display("[FAIL] pop mismatch: exp=%h got=%h", expected, rd_data);
        end
        rd_en = 1'b1;
        model.pop_front();
        @(posedge clock);
        @(negedge clock);
        rd_en = 1'b0;
    endtask

    // simultaneous push+pop
    task automatic push_pop(input logic [WIDTH-1:0] d);
        logic [WIDTH-1:0] expected;
        @(negedge clock);
        if (empty || full) begin
            $display("[INFO] push_pop skipped (empty=%b full=%b)", empty, full);
            return;
        end
        expected = model[0];
        checks++;
        if (rd_data !== expected) begin
            errors++;
            $display("[FAIL] push_pop read mismatch: exp=%h got=%h", expected, rd_data);
        end
        wr_en = 1'b1; wr_data = d; rd_en = 1'b1;
        model.pop_front();
        model.push_back(d);
        @(posedge clock);
        @(negedge clock);
        wr_en = 1'b0; rd_en = 1'b0;
    endtask

    task automatic pulse_reset;
        reset = 1; wr_en = 0; rd_en = 0; wr_data = '0;
        repeat (2) @(posedge clock);
        reset = 0; @(negedge clock);
        model.delete();
    endtask

    initial begin
        pulse_reset();

        // --- empty flag at start ---
        if (!empty) begin errors++; $display("[FAIL] not empty after reset"); end

        // --- fill to full, checking full asserts exactly at DEPTH ---
        for (int i = 0; i < DEPTH; i++) push(32'hA000_0000 + i);
        if (!full) begin errors++; $display("[FAIL] not full after %0d pushes", DEPTH); end

        // --- one more push should be rejected (stays full, no corruption) ---
        push(32'hDEAD_BEEF);           // should be skipped

        // --- drain fully, checking every value in FIFO order ---
        for (int i = 0; i < DEPTH; i++) pop();
        if (!empty) begin errors++; $display("[FAIL] not empty after draining all"); end

        // --- wraparound: push/pop past the end of the physical memory ---
        for (int i = 0; i < DEPTH; i++) push(32'hB000_0000 + i);
        pop(); pop();                  // advance rd_ptr
        push(32'hC000_0001); push(32'hC000_0002);  // wr_ptr wraps around
        while (!empty) pop();          // drain, order-checked

        // --- simultaneous push+pop keeps count steady, order correct ---
        push(32'h1111_0000);
        push(32'h1111_0001);
        push_pop(32'h2222_0000);       // pop head, push new tail, same cycle
        push_pop(32'h2222_0001);
        while (!empty) pop();

        if (errors == 0) $display("PASS, 0 errors, %0d read-checks", checks);
        else             $display("FAIL, %0d errors over %0d checks", errors, checks);
        $finish;
    end

    initial begin
        #100000; $display("[FAIL] TIMEOUT"); $finish;
    end

endmodule