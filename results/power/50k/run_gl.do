vlib work
vlog -work work int_v4.vo
vlog -sv -work work tb_gl.sv
vsim -c -t 1ps -voptargs="+acc" work.tb -L altera_ver -L lpm_ver -L sgate_ver -L altera_mf_ver -L altera_lnsim_ver -L cyclonev_ver
source int_v4_dump_all_vcd_nodes.tcl
vcd off
when {/tb/vcd_en == 1} { vcd on }
onfinish stop
run -all
vcd flush
quit -f