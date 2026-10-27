module bilinear_interp_tb();
    parameter BIT_WIDTH = 32;

    logic [BIT_WIDTH-1:0] P_00[2:0], P_01[2:0], P_10[2:0], P_11[2:0];
    logic [BIT_WIDTH-1:0] dx, dy;
    logic [BIT_WIDTH-1:0] P_out[2:0];

    // module-level test inputs (no longer task args)
    shortreal p00[2:0], p01[2:0], p10[2:0], p11[2:0];
    shortreal dx_val, dy_val;

    int errors = 0, checks = 0;

    bilinear_interp #(.BIT_WIDTH(32)) uut
        (.P_00(P_00), .P_01(P_01), .P_10(P_10), .P_11(P_11),
         .dx(dx), .dy(dy), .P_out(P_out));

    task automatic do_check;     // no ports — reads module-level vars
        shortreal gold[2:0], got[2:0];
        shortreal W_00, W_01, W_10, W_11;

        // drive DUT pixel inputs: convert the float test values to FP32 bit patterns
        for (int i = 0; i < 3; i++) begin
            P_00[i] = $shortrealtobits(p00[i]); 
            P_01[i] = $shortrealtobits(p01[i]);
            P_10[i] = $shortrealtobits(p10[i]);
            P_11[i] = $shortrealtobits(p11[i]);
        end

        //DUT fractional component inputs set
        dx = $shortrealtobits(dx_val);
        dy = $shortrealtobits(dy_val);
        #1;


        // golden reference weights for bilin interp
        W_00 = (1.0-dx_val)*(1.0-dy_val);
        W_01 = (1.0-dx_val)*dy_val;
        W_10 = dx_val*(1.0-dy_val);
        W_11 = dx_val*dy_val;

        //for each color, set the golden bilinear interpolation value
        for (int i = 0; i < 3; i++)
            gold[i] = W_00*p00[i] + W_01*p01[i] + W_10*p10[i] + W_11*p11[i];

        //same as above but just the DUT output
        for (int i = 0; i < 3; i++)
            got[i] = $bitstoshortreal(P_out[i]);

        checks++;

        //compare DUT out to reference to see if they are 10^-3 off
        for (int i = 0; i < 3; i++)
            assert((got[i]-gold[i] < 1e-3) && (gold[i]-got[i] < 1e-3))
                else begin errors++;
                    $error("ch%0d: got=%f exp=%f (dx=%f dy=%f)",
                           i, got[i], gold[i], dx_val, dy_val); end
    endtask

    initial begin

        //set fixed A00 - A11 color values
        p00 = '{1.0,0.5,0.25}; p01 = '{0.0,1.0,0.5};
        p10 = '{0.5,0.0,1.0};  p11 = '{0.25,0.75,0.0};

        // vary dx and dy values and perform check
        dx_val=0.0; dy_val=0.0; do_check();   // → P00 (all weight on corner 00)
        dx_val=1.0; dy_val=0.0; do_check();    // → P10
        dx_val=0.0; dy_val=1.0; do_check();    // → P01
        dx_val=1.0; dy_val=1.0; do_check();    // → P11
        dx_val=0.5; dy_val=0.5; do_check();    // → Average of all four corners

        // test random input color and dx dy values for 1000 iterations
        for (int t = 0; t < 1000; t++) begin
            for (int c = 0; c < 3; c++) begin
                p00[c]=$urandom_range(0,1000)/1000.0;
                p01[c]=$urandom_range(0,1000)/1000.0;
                p10[c]=$urandom_range(0,1000)/1000.0;
                p11[c]=$urandom_range(0,1000)/1000.0;
            end
            dx_val=$urandom_range(0,999)/1000.0;
            dy_val=$urandom_range(0,999)/1000.0;
            do_check();
        end

        if (errors==0) $display("PASS, 0 errors, checks = %0d", checks);
        else           $display("FAIL, %0d errors, checks = %0d", errors, checks);
        $finish;
    end
endmodule