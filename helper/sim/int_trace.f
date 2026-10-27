// int.f + waveform tracing
// Build:  verilator --binary --timing -Wno-fatal -Wno-WIDTH -j 0 -f helper/sim/int_trace.f
// Run:    ./obj_dir/Vtop_full_int_st_tb   (learn block uncommented in TB)
--trace
-fno-inline
--trace-max-array 64
--trace-max-width 256
-f helper/sim/int.f