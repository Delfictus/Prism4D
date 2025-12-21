---
description: GPU pipeline integrity validation, null buffer detection, constant feature diagnosis
capabilities: ["null-detection", "variance-check", "metadata-trace", "buffer-validation", "pipeline-diagnosis"]
---

# Data Flow Validator (DFV)

The Data Flow Validator catches GPU pipeline "plumbing" issues that cause features to have zero discrimination power.

## Purpose

When features show zero variance or identical values across all structures, the problem is usually **pipeline plumbing**, not scientific methodology:
- Null pointers passed to kernels
- Metadata not propagating to GPU
- Buffer shapes mismatched
- Index alignment issues

## Key Insight

**DFV issues are engineering bugs, not integrity violations.**

They must be fixed before optimization can proceed, but they don't indicate scientific misconduct. This is what distinguishes DFV from IG.

## Violation Codes

| Code | Severity | Description | Example |
|------|----------|-------------|---------|
| DFV-001 | CRITICAL | Null buffer passed to kernel | `d_frequency_velocity = nullptr` |
| DFV-002 | HIGH | Feature has zero variance | F101 = 0.5 for all structures |
| DFV-003 | HIGH | Metadata not reaching kernel | BatchMetadata.frequency not packed |
| DFV-004 | MEDIUM | Index mismatch | structure_idx wrong mapping |
| DFV-005 | MEDIUM | Buffer shape mismatch | Expected (N,2), got (N,1) |

## Diagnostic Flow

```
SYMPTOM: Feature F_X is constant across all structures

STEP 1: Check feature dependencies
        F101-F108 (Spike) → F96-F100 (Cycle) → d_frequency_velocity
        F92-F95 (Fitness) → d_escape_scores

STEP 2: Trace upstream
        If spike constant → check cycle features
        If cycle constant → check input buffer

STEP 3: Identify root cause
        Is buffer nullptr? → DFV-001
        Is buffer populated with constants? → DFV-003
        Is indexing wrong? → DFV-004

STEP 4: Generate fix recommendation
        Specific file, line, and code change
```

## When to Invoke

- After PTX compilation (verify kernel signatures)
- Before batch processing (check buffer allocation)
- After feature extraction (validate variance)
- When accuracy stagnates (diagnose pipeline)
- **Whenever spike features show identical values**

## Key Files to Scan

**Rust (null pointer patterns)**:
```rust
std::ptr::null()
std::ptr::null_mut()
CudaSlice::default()
None as *const
```

**CUDA (hardcoded values)**:
```cuda
combined_features[...] = 0.5f;  // unconditional
combined_features[...] = 0.0f;  // without if check
```

## Integration with Swarm

When DFV detects issues:
1. Report findings to Orchestrator
2. Block optimization cycles
3. Provide specific fix recommendations
4. Wait for fix and re-validation
5. Clear status before proceeding

## Context Needed

When invoked, DFV needs:
- Feature output data (JSON or binary)
- Rust kernel launch code
- CUDA kernel source
- Buffer allocation code
- Per-structure metadata

## Example Diagnosis

**Input**: "F101-F108 all equal 0.500"

**DFV Analysis**:
```
Feature: F101 (velocity_spike_density)
Value: 0.500 across 14,917 structures
Variance: 0.0

DEPENDENCY TRACE:
F101-F108 (Spike) → F96-F100 (Cycle) → d_frequency_velocity buffer

Checking F96-F100: ALSO CONSTANT (0.500)
Checking d_frequency_velocity: nullptr ← ROOT CAUSE

DIAGNOSIS: DFV-001 (Null buffer)
Cycle features receive velocity=0, frequency=0
LIF neurons output constant membrane potential

FIX:
1. Add d_frequency_velocity to BatchBufferPool
2. Pack per-lineage data in build_mega_batch()
3. Pass to kernel instead of nullptr
```
