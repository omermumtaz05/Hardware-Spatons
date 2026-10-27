// Quantised integer design - full system
// Build:  verilator --binary --timing -Wno-fatal -Wno-WIDTH -j 0 -f helper/sim/int.f
// Run:    ./obj_dir/Vtop_full_int_st_tb
// Trace:  add --trace -fno-inline --trace-max-array 64 --trace-max-width 256

--top-module top_full_int_st_tb

// geometric stage
rtl/common/coord_gen.sv
rtl/geo/LUT_fxp.sv
rtl/geo/csf_int.sv
rtl/geo/fetch_four_nearest_int.sv
rtl/geo/input_buffer_parity_int.sv
rtl/geo/bilinear_interp_int.sv
rtl/geo/top_geo_int.sv

// line buffer + patch
rtl/common/inter_line_buffer.sv
rtl/photo/patch_extractor_int.sv

// photometric stage
rtl/photo/dot_prod_int.sv
rtl/photo/neuron_int.sv
rtl/photo/layer_int_v2.sv
rtl/common/fifo_residual.sv
rtl/photo/nn_pipelined_int_v4.sv
rtl/photo/post_mlp_int.sv
rtl/photo/photo_stage_int_v4.sv

// top + testbench
rtl/top_full_int_st.sv
tb/top/top_full_int_st_tb.sv