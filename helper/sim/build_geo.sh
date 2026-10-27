set -e
for v in geo_int geo_fp32 geo_fp32_full; do
    echo "=== building  $v ==="
    mkdir -p obj/$v
    mkdir -p obj/$v
    verilator --binary --timing -Wno-fatal -Wno-WIDTH -j 0 \
        --Mdir obj/$v -f helper/sim/$v.f
done