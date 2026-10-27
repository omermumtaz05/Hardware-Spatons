create_clock -name clk -period 12.500 [get_ports clock]
derive_clock_uncertainty

set_false_path -from [get_ports reset]
set_false_path -to   [get_ports {out_valid frame_done out_row_last out_x[*] out_y[*] P_out_flat[*]}]