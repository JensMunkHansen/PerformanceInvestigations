## L1 Bound:

Load instructions are delayed due to:

 - Dependency on a previous store (store-to-load forwarding delay)

 - Not enough register reuse (too many loads from memory)

 - Register pressure (spill/reload to L1 instead of keeping values in registers)

 - Port contention (e.g., too many loads competing for the same issue port)

### How to mitigate

1. Use registers more efficiently
 - Store values in registers instead of reloading from memory
 - Use scalar variables or loop invariants directly in registers

2. Reduce memory traffic
 - Minimize repeated loads (hoist loads out of inner loops)
 - Apply loop unrolling smartly to help the compiler track values in registers

3. Leverage SIMD
  - Vector registers are large (e.g. 512 bits in AVX-512)
  - They reduce memory accesses by operating on multiple elements at once

4. Use compiler flags that increase register use

## L2 Bound:

The CPU pipeline is stalled waiting for data from the L2 cache,
because the data was not found in L1, and L2 access is taking too long
to satisfy the load.

It doesn’t necessarily mean L2 is missing — it means:

 - The load missed L1, went to L2
 - But the load wasn't serviced fast enough from L2
 - So the pipeline had a stall and wait 
 
### Causes
1. L1 Misses + Inefficient L2 Access
 - Poor temporal locality: data not reused soon enough to stay in L1
 - L2 isn't fast enough to keep up with demand

2. Limited L2 Bandwidth / Access Ports
 - L2 is slower than L1 (~10–20 cycles latency)
 - L2 can only service a limited number of loads at once

3. L2 Misses, but L3 Hits
 - VTune may still attribute this as L2 Bound if the stall happens at
   L2, even if L3 serves the data later
   
### How to mitigate

1. Keep data in L1 with better temporal reuse
 - Access the same data multiple times before moving on
 - Loop optimizations: blocking, tiling, unrolling (reuse values across iterations)

2. Use vectorization
 - SIMD loads reduce the number of load operations → fewer chances to miss L1 and stress L2

3. Reduce working set size
 - Try to fit active data into L1 cache (~32KB/core typically)
 - If not, then into L2 (~256–512KB)

4. Improve memory access patterns
 - Access data contiguously
 - Avoid large strides or irregular patterns that kill spatial locality

## L3 Bound

The CPU pipeline is stalled waiting for data that missed both L1 and
L2, and is now being served from the L3 cache, but not fast enough to
keep the pipeline fed.

So:
 - L1 = miss
 - L2 = miss
 - L3 = hit, but...
 - The CPU had to wait long enough on that L3 access to cause a
   measurable stall

This is still much better than being DRAM Bound, but worse than
hitting in L1 or L2.

### Causes
1. Large working set
 - Your active data exceeds L1+L2 capacity
 - Especially common in numerical kernels, streaming access patterns, or large matrix ops

2. Poor data reuse
 - Data is not accessed again soon enough to stay in L1/L2
 - CPU keeps evicting and reloading the same cache lines

3. High core-to-core contention
 - On multi-core systems, L3 is often shared across cores
 - If other cores are heavily using L3, you might see contention

4. L3 is simply too slow for your access pattern
 - L3 latency is ~30–70 cycles (vs. ~4–10 for L2)
 - If your loop issues tons of loads per cycle, L3 might not keep up

### How to mitigate

1. Reduce working set size
 - Use blocking/tiling to work on smaller chunks that fit into L1/L2

2. Improve temporal locality
 - Reuse the same data before it gets evicted from L1/L2
 - Reorder loops or fuse them if needed

3. Improve spatial locality
 - Access data sequentially, avoid strided or random loads
 - Align and pad arrays to cache line boundaries

4. Exploit L3 prefetching
 - Access patterns that are too irregular defeat hardware prefetchers
 - Try to access memory in predictable, linear patterns

## 📋 Summary: Cache-Level Bound Types

| **Bound Type**   | **What It Means**                                               | **Fixes**                                                            |
|------------------|-----------------------------------------------------------------|----------------------------------------------------------------------|
| **L1 Bound**     | Stalls even when data is in L1 (e.g., due to register pressure) | Use registers better, reduce redundant loads                         |
| **L2 Bound**     | Data missed L1, L2 is too slow to serve in time                 | Improve reuse, reduce working set, optimize loops                    |
| **L3 Bound**     | Missed L1+L2, stalled on L3 access                              | Improve locality, block/tile loops, reduce working set               |
| **DRAM Bound**   | Missed all caches, now waiting on main memory (very slow)       | Reduce memory traffic, prefetching, blocking, better access patterns |
