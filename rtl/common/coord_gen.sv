module coord_gen# (
    parameter IMG_HEIGHT = 256,
    parameter IMG_WIDTH = 256,
    parameter IND_WIDTH = 8
  )
  (
    input clock,
    input reset,
    input stall,                      // NEW: freeze the scan when high

    output [IND_WIDTH - 1:0] x_out,   // current x coordinate (0-255)
    output [IND_WIDTH - 1:0] y_out,   // current y coordinate (0-255)
    output logic frame_done,          // high when on the last pixel (255,255)
    output logic row_last             // NEW: high when x is on last column of a row
  );

    // internal registers holding the current coordinate
    logic [IND_WIDTH - 1:0] x_buf, y_buf;

    always_ff @ (posedge clock)
      begin
        if(reset)
          begin
            x_buf <= '0;
            y_buf <= '0;
          end
        else if (!frame_done && !stall)   // NEW: also freeze on stall
        begin
              if(x_buf == IMG_WIDTH - 1)
                begin
                    begin
                      x_buf <= '0;
                      y_buf <= y_buf + 8'b1;
                  end
                end
              else
                begin
                  x_buf <= x_buf + 8'b1;
                end
            end
      end

    assign frame_done = (x_buf == (IMG_WIDTH - 1) && y_buf == (IMG_HEIGHT - 1));
    assign row_last   = (x_buf == (IMG_WIDTH - 1));   // NEW

    assign x_out = x_buf;
    assign y_out = y_buf;

  endmodule