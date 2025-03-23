ocperf.py stat -e cycles,instructions <your_command>



sudo perf record -e cycles,instructions ./grind
perf report perf.data


sudo perf stat -e cycles,instructions ./grind
perf record -e LLC-load-misses ./baseline

~/github/pmu-tools/ocperf.py stat -e cycles,instructions ./mmul_bench
