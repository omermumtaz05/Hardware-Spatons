create_clock -name clock -period 20.000 [get_ports clock]
derive_clock_uncertainty

set_false_path -from [remove_from_collection [all_inputs] [get_ports clock]] -to [all_registers]
set_false_path -from [all_registers] -to [all_outputs]