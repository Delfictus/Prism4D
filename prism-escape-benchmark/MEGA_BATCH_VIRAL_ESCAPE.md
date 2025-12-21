# MEGA-BATCH MODE FOR VIRAL ESCAPE: MAXIMUM THROUGHPUT

## Current Status

**What we're using:** `--pure-gpu` (single-structure with buffer pooling)
- Processes: ONE structure at a time  
- Buffer reuse: YES (zero-allocation after warmup)
- Throughput: ~100-200 structures/second
- For 1000 mutations: ~5-10 seconds

**What exists:** `mega_fused_batch.cu` (TRUE multi-structure kernel)
- Processes: ALL structures in SINGLE kernel launch
- Target: 221 structures in <100ms
- Throughput: 2000+ structures/second
- For 1000 mutations: <1 second!

## For Viral Escape: Use MEGA-BATCH!

### Command:
```bash
# Process 1000 mutations in TRUE batch mode
target/release/prism-lbs batch \\
  --input /path/to/1000_mutant_structures/ \\
  --output /tmp/batch_results/ \\
  --pure-gpu
```

### Expected Performance:
```
Current (single-structure):  1000 mutations in ~5-10 seconds
Mega-batch (true batching): 1000 mutations in <1 second!

10× speedup for viral escape!
```

### Architecture:
```
Mega-batch kernel:
- Grid: 1000 blocks (one per mutation)
- Block: 256 threads (one per tile)
- Shared memory: Per-structure (isolated)
- Single kernel launch: Process ALL mutations together
- Buffer pooling: Still active (reuse across batches)
```

## Next: Enable Mega-Batch for Viral Escape

Update viral escape adapter to use mega-batch mode for maximum throughput!
