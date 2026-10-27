module LUT_tb();

  parameter BIT_WIDTH = 32;   // width of each coordinate output (Q-format fixed-point)
  parameter IND_WIDTH = 8;    // width of each index -> 256 values per axis
  parameter LUT_FILE = "data_files/lut.hex";
  // DUT input drivers
  logic [IND_WIDTH - 1:0] x_ind;
  logic [IND_WIDTH - 1:0] y_ind;
  logic clock;
  logic reset;

  // DUT outputs
  logic [BIT_WIDTH - 1:0] x_out;
  logic [BIT_WIDTH - 1:0] y_out;

  // instantiate the device under test
  MLP_geo_LUT # (
    .BIT_WIDTH(BIT_WIDTH),
    .IND_WIDTH(IND_WIDTH),
    .LUT_FILE(LUT_FILE)
  ) uut
  (
    .x_ind(x_ind),
    .y_ind(y_ind),
    .clock(clock),
    .reset(reset),

    .x_out(x_out),
    .y_out(y_out)

  );


  // golden reference: 256*256 entries, each 64 bits ({y_coord, x_coord})
  logic [2*BIT_WIDTH - 1:0] golden [0:(1 << (2*IND_WIDTH)) - 1]; // initialize reg array of 64 bit width and 256^2 no. of elements
  initial $readmemh(LUT_FILE, golden);

  // for final report of total checks and errors
  int checks = 0;
  int errors = 0;
 
  //initial #1 golden[0] = 64'hDEADBEEF_DEADBEEF;

  //above line is to make the golden LUT and the DUT different by one line to see if it fails
  
  // clock generation: 10 time-unit period
  initial clock = '0;

  always #5 clock = ~clock;


  // drive the indices, then wait one clock edge + settle time for the result
  task automatic do_read(input [IND_WIDTH - 1:0] xi, input [IND_WIDTH - 1:0] yi);
    x_ind = xi;
    y_ind = yi;

    @ (posedge clock);
    @ (posedge clock);
    #1;    // small delay so x_out/y_out are stable before checking

  endtask

  // does read then displays to ensure LUT outputs consistent with python script LUT values
  task automatic do_print(input [IND_WIDTH-1:0] xi, input [IND_WIDTH-1:0] yi);
    real xf, yf;
    do_read(xi, yi); // read the value from LUT 

    //convert from bits to short real to display!
    xf = $bitstoshortreal(x_out);
    yf = $bitstoshortreal(y_out);

    //print the LUT output values
    $display("(x=%0d, y=%0d) -> x_out=%f  y_out=%f", xi, yi, xf, yf);
  endtask

  // read one address and compare both outputs against the golden model
  task automatic do_check(input [IND_WIDTH - 1:0] xi, input [IND_WIDTH - 1:0] yi);

    do_read(xi, yi);

    checks++;
    // lower 32 bits of golden entry = expected x
    assert(x_out === golden[{yi, xi}][BIT_WIDTH - 1:0])
           else begin errors++;
             $error("x mismatch at (%0d, %0d): got %0h, expecting %0h", yi, xi, x_out, golden[{yi, xi}][BIT_WIDTH - 1:0]); end

    // upper 32 bits of golden entry = expected y
    assert(y_out === golden[{yi, xi}][2*BIT_WIDTH - 1:BIT_WIDTH])
           else begin errors++;
             $error("y mismatch at (%0d, %0d): got %0h, expecting %0h", yi, xi, y_out, golden[{yi, xi}][2*BIT_WIDTH - 1:BIT_WIDTH]);  end

           endtask

  initial begin

    // apply reset for 2 cycles, hold indices at 0
    reset = 1; x_ind = '0; y_ind = '0;

    repeat (2) @ (posedge clock);
    reset = 0;

    // comprehensive check for each element in LUT
    for(int i = 0; i < 256; i++)
      for(int j = 0; j < 256; j++)
        begin
          do_check(j[IND_WIDTH - 1:0], i[IND_WIDTH - 1:0]);
        end

  // print 5 output values to compare to python script
   $display("--- spot checks (compare to Python grid[y,x]) ---");
    do_print(0,   0); // expect x=18.582931519  y= 0.000000
    do_print(128, 128); // expect x=128.681274414  y= 96.407822
    do_print(255, 255); // expect x=255.000000000  y= 255.000000
    do_print(5,   3); // expect x=20.378252029  y= 0.000000
    do_print(200, 100); // expect x=207.071166992  y= 61.721947

    // report final result
    if(errors == 0)
      $display("PASS, 0 errors, checks = %0d", checks);
    else
      $display("FAIL somewhere, checks = %0d", checks);


    $finish;
  end


endmodule
