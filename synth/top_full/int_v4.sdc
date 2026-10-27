create_clock -name clk -period 12.500 [get_ports clock]
derive_clock_uncertainty

set_false_path -from [get_ports reset]
set_false_path -from [get_ports out_ready]
set_false_path -to   [get_ports {final_rgb_flat[*] out_valid frame_done}]