perf record -g -e cycles:u ./baseline
perf report
see screen shot.

perf stat -e cycles,instructions,cache-misses,LLC-load-misses ./baseline
Performance counter stats for './baseline':

   193,569,386,295      cycles                                                             
    37,057,860,005      instructions                     #    0.19  insn per cycle         
        11,905,162      cache-misses                                                       
         7,859,239      LLC-load-misses                                                    

      11.984869292 seconds time elapsed

      45.798697000 seconds user
       0.047919000 seconds sys

perf list      
perf record -a -e LLC-load-misses  ./baseline

see sxren shot 2.


perf record -e cycles,stalled-cycles-backend ./baseline
perf report


perf stat -e cycles,instructions,cache-misses,LLC-load-misses ./baseline


perf record -e cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses ./baseline
