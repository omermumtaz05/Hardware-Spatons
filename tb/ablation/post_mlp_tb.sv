import fp32_pkg::*;

module post_mlp_tb();
    localparam int BW = 32;

    logic [BW-1:0] z [2:0];
    logic [BW-1:0] center [2:0];
    logic [BW-1:0] final_rgb [2:0];

    post_mlp #(.BW(BW), .LUT_FILE("data_files/sigmoid_lut.hex")) uut (
        .z(z), .center_rgb(center), .final_rgb(final_rgb)
    );

    // bit-domain compare (no shortreal)
    function automatic logic [31:0] uord(input logic [31:0] a, input logic [31:0] b);
        logic [31:0] ao, bo;
        ao = a[31] ? (32'h8000_0000 - a) : (a | 32'h8000_0000);
        bo = b[31] ? (32'h8000_0000 - b) : (b | 32'h8000_0000);
        return (ao > bo) ? (ao - bo) : (bo - ao);
    endfunction

    integer fd, r, t, errors;
    logic [31:0] z0,z1,z2, c0,c1,c2, g0,g1,g2;

    initial begin
        errors = 0; t = 0;
        fd = $fopen("data_files/post_mlp_golden.txt", "r");
        if (fd == 0) begin $display("cant open golden"); $finish; end

        while (!$feof(fd)) begin
            while (1) begin
                r = $fscanf(fd, "%h %h %h %h %h %h | %h %h %h\n",
                            z0,z1,z2, c0,c1,c2, g0,g1,g2);
                if (r != 9) break;   // EOF or parse fail -> stop, don't spin
                // ... rest of body ...
                t++;
            end
            if (r != 9) continue;

            z[0]=z0; z[1]=z1; z[2]=z2;
            center[0]=c0; center[1]=c1; center[2]=c2;
            #1;  // settle combinational

            for (int c = 0; c < 3; c++) begin
                logic [31:0] got, exp, d;
                got = final_rgb[c];
                exp = (c==0)?g0:(c==1)?g1:g2;
                d = uord(got, exp);
                if (d > 32'd200000) begin
                    errors++;
                    $display("[FAIL] vec %0d ch %0d z=%h c=%h exp=%h got=%h uord=%0d",
                             t, c, z[c], center[c], exp, got, d);
                end
            end
            t++;
        end
        $fclose(fd);
        if (errors==0) $display("POST_MLP PASS (%0d vectors)", t);
        else           $display("POST_MLP FAIL: %0d errors / %0d vectors", errors, t);
        $finish;
    end
endmodule