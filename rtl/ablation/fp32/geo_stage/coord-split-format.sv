import fp32_pkg::*;


module coordsplit_format# (
        parameter BIT_WIDTH = 32
    )(

    input logic [BIT_WIDTH - 1:0] x_in,
    input logic [BIT_WIDTH - 1:0] y_in,

    output logic [BIT_WIDTH - 1:0] dx,
    output logic [BIT_WIDTH - 1:0] dy,

    output logic [BIT_WIDTH - 1:0] x_int,
    output logic [BIT_WIDTH - 1:0] y_int

);

   

    always_comb 
    begin
        x_int = fp32_floor(x_in); // get floor of x
        y_int = fp32_floor(y_in); // get floor of y
        dx    = fp32_sub(x_in, x_int); // get fractional part of x
        dy    = fp32_sub(y_in, y_int);  // get fractional part of y
    end

endmodule