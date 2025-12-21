---
description: Run Data Flow Validator to detect GPU pipeline issues
---

# Data Flow Validator Check

Run the DFV agent to detect null buffers, constant features, and metadata propagation failures.

## Purpose

The DFV agent catches pipeline "plumbing" issues that cause zero-discrimination features:
- **DFV-001**: Null buffer passed to kernel (CRITICAL)
- **DFV-002**: Constant feature across all structures (HIGH)
- **DFV-003**: Metadata not reaching kernel (HIGH)
- **DFV-004**: Index mismatch in kernel (MEDIUM)
- **DFV-005**: Buffer shape mismatch (MEDIUM)

## When to Use

- After PTX compilation to verify kernel signatures
- Before batch processing to check buffer allocation
- After feature extraction to validate feature variance
- When accuracy stagnates to diagnose pipeline issues
- **Whenever spike features (F101-F108) show identical values**

## Diagnostic Flow

```
1. FEATURE VARIANCE CHECK
   └─► All features have variance > 0? ────► YES ──► PASS
                  │
                 NO
                  ▼
2. IDENTIFY CONSTANT FEATURES
   └─► Which features are constant?
       F96-F100 (Cycle) ──────► Check frequency/velocity input
       F101-F108 (Spike) ─────► Check cycle features first
       F92-F95 (Fitness) ─────► Check ddG computation
                  │
                  ▼
3. TRACE INPUT BUFFERS
   └─► Is input buffer nullptr?
                  │
            YES   │   NO
             ▼    │    ▼
       DFV-001   │  DFV-003
       Null buf  │  Metadata not propagating
                  │
                  ▼
4. GENERATE FIX RECOMMENDATION
   └─► Specific code location and required change
```

## Arguments

$ARGUMENTS

Usage:
- `/prism-ve-swarm:dfv-check` - Full pipeline scan
- `/prism-ve-swarm:dfv-check --features features.json` - Analyze feature output
- `/prism-ve-swarm:dfv-check --rust-only` - Only scan Rust files
- `/prism-ve-swarm:dfv-check --cuda-only` - Only scan CUDA files
- `/prism-ve-swarm:dfv-check path/to/file.rs` - Scan specific file

## Example Output

```
╔══════════════════════════════════════════════════════════════════════╗
║  DFV-002: CONSTANT FEATURE DETECTED                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  Feature: F101 (velocity_spike_density)                              ║
║  Constant Value: 0.500000 across all 14,917 structures               ║
║                                                                      ║
║  DEPENDENCY TRACE:                                                   ║
║  F101-F108 (Spike) ──depends on──► F96-F100 (Cycle)                  ║
║  F96-F100 (Cycle)  ──depends on──► d_frequency_velocity buffer       ║
║                                                                      ║
║  ROOT CAUSE: d_frequency_velocity = nullptr                          ║
║                                                                      ║
║  FIX:                                                                ║
║  1. Add d_frequency_velocity to BatchBufferPool                      ║
║  2. Pack per-lineage frequency/velocity in build_mega_batch()        ║
║  3. Pass to kernel instead of nullptr                                ║
╚══════════════════════════════════════════════════════════════════════╝
```

## Key Files Scanned

**Rust files** (for null pointers):
- `crates/prism-gpu/src/mega_fused_batch.rs`
- `crates/prism-gpu/src/buffer_pool.rs`

**CUDA files** (for hardcoded values):
- `crates/prism-gpu/src/kernels/mega_fused_batch.cu`
- `crates/prism-gpu/src/kernels/prism_4d_stages.cuh`

## DFV vs IG

| Agent | Detects | Severity | Action |
|-------|---------|----------|--------|
| **DFV** | Pipeline bugs | Engineering | FIX_PIPELINE |
| **IG** | Scientific fraud | Ethical | HALT_SWARM |

DFV issues are **not** integrity violations. They're bugs that must be fixed before optimization can proceed, but they don't indicate scientific misconduct.

## Integration

When DFV finds issues:
1. Swarm pauses optimization
2. Issues must be fixed before continuing
3. Re-run `/prism-ve-swarm:dfv-check` to verify fix
4. Resume with `/prism-ve-swarm:swarm-cycle`
