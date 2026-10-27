# Hardware-Spatons

Hardware implementation of Spatons.

## Source layout

- `rtl/geo/`, `rtl/photo/`, and `rtl/common/` contain the final integer design.
- `rtl/top_full_int_st.sv` is its full-system top.
- `rtl/ablation/` contains the FP32 and legacy RTL retained for ablations.
- `tb/geo/`, `tb/photo/`, `tb/common/`, and `tb/top/` contain final-design tests;
  FP32 ablation benches are in `tb/ablation/`.
- `data_files/` retains model weights, lookup tables, and ROM inputs. Simulation
  captures and generated test-image sets are not checked in.

## Final-design simulations

With Verilator installed, run from the repository root:

```sh
verilator --binary --timing -Wno-fatal -Wno-WIDTH -j 0 -f helper/sim/geo_int.f
./obj_dir/Vtop_geo_int_tb

verilator --binary --timing -Wno-fatal -Wno-WIDTH -j 0 -f helper/sim/int.f
./obj_dir/Vtop_full_int_st_tb
```

The full-system testbench writes `data_files/full_sys_st.hex`. The geometric
test writes `data_files/capture_array_int.hex`.

## Geometric ablation simulations

`bash helper/sim/build_geo.sh` builds the integer, FP32, and FP32-LUT variants
from their filelists. Run the resulting binaries from the repository root:

```sh
./obj/geo_int/Vtop_geo_int_tb
./obj/geo_fp32/Vtop_geo_fp32_tb
./obj/geo_fp32_full/Vtop_geo_fp32_full_tb
```

The nine synthetic images used by the accuracy sweep can be regenerated with
`python3 helper/gen/gen_test_images.py`. Their source generator is retained;
the generated `.hex` images and comparison PNGs are not.
## Synthesis

Each Quartus project in `synth/` reads memory-initialization files from
`data_files/` relative to the project directory. Link it to the repository's
`data_files/` before compiling, e.g. `ln -s ../../data_files synth/<project>/data_files`
(on Windows: `mklink /J synth\<project>\data_files data_files`).
The final design is `synth/top_full` (revision `int_v4`).
