# PRISM-4D Data Flow Validation Protocols

The Data Flow Validator (DFV) agent ensures GPU pipeline integrity by detecting null inputs, constant features, and metadata propagation failures that cause zero-discrimination features.

## The Problem DFV Solves

**Symptom**: Features show zero variance across all structures
```
F101: 0.333333 (identical for all 14,917 structures)
F102: 0.250000 (identical for all 14,917 structures)
...
All variants produce identical spike responses (0.500)
```

**Root Cause**: Pipeline plumbing issues
- `nullptr` passed to kernel where data expected
- Per-lineage metadata not indexed to correct structure
- Buffer shapes don't match kernel expectations

**Without DFV**: Manual debugging required (hours)
**With DFV**: Automatic detection and diagnosis (seconds)

---

## DFV Violation Codes

### DFV-001: Null Buffer Detection
**Severity**: CRITICAL
**Description**: nullptr or null array passed to GPU kernel where real data expected

**Detection**:
```python
def check_null_buffers(kernel_call):
    """
    Inspect kernel launch parameters for null pointers.
    """
    null_params = []
    
    for param_name, param_value in kernel_call.parameters.items():
        if param_value is None:
            null_params.append(param_name)
        elif hasattr(param_value, 'ptr') and param_value.ptr == 0:
            null_params.append(param_name)
    
    if null_params:
        return DFVViolation(
            code="DFV-001",
            severity="CRITICAL",
            message=f"Null buffer(s) passed to kernel: {null_params}",
            fix_hint="Allocate and populate buffer before kernel launch"
        )
    return None
```

**Example**: Your spike feature bug
```
Problem: cycle_velocity and cycle_frequency passed as nullptr
Effect: Stage 8 cycle features all receive velocity=0, frequency=0
Result: All spike responses identical (0.500)

Fix: Allocate d_frequency_velocity buffer, pack per-lineage data,
     pass to kernel with correct structure indexing
```

---

### DFV-002: Constant Feature Detection
**Severity**: HIGH
**Description**: Feature has zero variance across all structures (all identical values)

**Detection**:
```python
def check_feature_variance(features, feature_indices, min_variance=1e-10):
    """
    Check that features have non-zero variance.
    Constant features cannot discriminate and indicate pipeline issues.
    """
    violations = []
    
    for idx in feature_indices:
        feature_values = features[:, idx]
        variance = np.var(feature_values)
        
        if variance < min_variance:
            unique_value = feature_values[0]
            violations.append(DFVViolation(
                code="DFV-002",
                severity="HIGH",
                feature_index=idx,
                constant_value=unique_value,
                message=f"Feature F{idx} has zero variance (constant={unique_value})",
                diagnosis=diagnose_constant_feature(idx, unique_value),
                fix_hint="Check input buffer for this feature stage"
            ))
    
    return violations

def diagnose_constant_feature(idx, value):
    """
    Provide diagnostic hints based on feature index and constant value.
    """
    # Feature ranges from PRISM-4D architecture
    if 96 <= idx <= 100:  # Cycle features
        if value == 0.0:
            return "Cycle features all zero - frequency/velocity inputs likely nullptr"
        elif value == 0.5:
            return "Cycle features at midpoint - LIF neurons receiving constant input"
    
    if 101 <= idx <= 108:  # Spike features
        if value == 0.5:
            return "Spike features at 0.5 - cycle features providing constant input"
        return "Spike features constant - check Stage 8.5 input propagation"
    
    if 92 <= idx <= 95:  # Fitness features
        return "Fitness features constant - check ddG/expression computation"
    
    return "Unknown feature range - check corresponding kernel stage"
```

**Diagnostic Output Example**:
```
╔══════════════════════════════════════════════════════════════════════╗
║  DFV-002: CONSTANT FEATURE DETECTED                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  Feature: F101 (velocity spike density)                              ║
║  Value: 0.333333 across all 14,917 structures                        ║
║  Variance: 0.0                                                       ║
║                                                                      ║
║  DIAGNOSIS: Spike features constant - check Stage 8.5 input          ║
║             propagation. Cycle features (F96-F100) likely receiving  ║
║             nullptr for frequency/velocity inputs.                    ║
║                                                                      ║
║  RECOMMENDED ACTION:                                                 ║
║  1. Check mega_fused_batch.rs for cycle_frequency buffer allocation  ║
║  2. Verify BatchMetadata frequency/velocity packed into buffer       ║
║  3. Confirm kernel receives non-null d_cycle_data parameter          ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

### DFV-003: Metadata Propagation Failure
**Severity**: HIGH
**Description**: Per-lineage metadata (frequency, velocity, escape) not reaching kernel

**Detection**:
```python
def check_metadata_propagation(batch_metadata, kernel_inputs):
    """
    Verify that BatchMetadata fields are correctly packed into kernel inputs.
    """
    violations = []
    
    # Check frequency propagation
    metadata_frequencies = [m.frequency for m in batch_metadata]
    if 'd_frequency' in kernel_inputs:
        kernel_frequencies = kernel_inputs['d_frequency']
        
        if not np.allclose(metadata_frequencies, kernel_frequencies):
            violations.append(DFVViolation(
                code="DFV-003",
                severity="HIGH",
                field="frequency",
                message="BatchMetadata.frequency not matching kernel input",
                expected=metadata_frequencies[:5],
                actual=kernel_frequencies[:5] if kernel_frequencies else None
            ))
    else:
        violations.append(DFVViolation(
            code="DFV-003",
            severity="HIGH",
            field="frequency",
            message="d_frequency buffer missing from kernel inputs"
        ))
    
    # Check velocity propagation
    metadata_velocities = [m.frequency_velocity for m in batch_metadata]
    # Similar check...
    
    return violations
```

**Trace Protocol**:
```
METADATA PROPAGATION TRACE
==========================

Source: BatchMetadata struct (main.rs)
  ├── frequency: 0.0523
  ├── frequency_velocity: 0.0089
  └── escape_score: 0.78

Step 1: Pack into batch (build_mega_batch)
  └── packed_batch.metadata[structure_idx] = metadata

Step 2: Allocate GPU buffer (BatchBufferPool)
  └── d_frequency_velocity: CudaSlice<f32>  ← CHECK THIS EXISTS

Step 3: Copy to GPU (detect_pockets_batch)
  └── cuMemcpyHtoD(d_frequency_velocity, h_frequency_velocity)

Step 4: Pass to kernel (mega_fused_batch_detection_prism4d)
  └── kernel<<<blocks, threads>>>(
        ...,
        d_frequency_velocity,  ← CHECK NOT nullptr
        ...
      )

Step 5: Index in kernel (stage8_cycle_features)
  └── float velocity = frequency_velocity_in[structure_idx * 2 + 0];
      float frequency = frequency_velocity_in[structure_idx * 2 + 1];
      ← CHECK structure_idx CORRECT

FAILURE POINT: [Identify where trace breaks]
```

---

### DFV-004: Index Mismatch
**Severity**: MEDIUM
**Description**: structure_idx in kernel doesn't map to correct metadata

**Detection**:
```python
def check_index_alignment(n_structures, batch_output, batch_metadata):
    """
    Verify structure indices align between kernel output and metadata.
    """
    violations = []
    
    # Check that output count matches metadata count
    if len(batch_output.structures) != len(batch_metadata):
        violations.append(DFVViolation(
            code="DFV-004",
            severity="MEDIUM",
            message=f"Output count ({len(batch_output.structures)}) != "
                   f"metadata count ({len(batch_metadata)})"
        ))
    
    # Sample check: verify known properties align
    for i in [0, n_structures // 2, n_structures - 1]:
        output = batch_output.structures[i]
        meta = batch_metadata[i]
        
        # Check that structure sizes match expectation
        expected_residues = meta.n_residues
        actual_features = len(output.combined_features)
        
        if actual_features != expected_residues * 101:
            violations.append(DFVViolation(
                code="DFV-004",
                severity="MEDIUM",
                structure_idx=i,
                message=f"Feature count mismatch at structure {i}: "
                       f"expected {expected_residues * 101}, got {actual_features}"
            ))
    
    return violations
```

---

### DFV-005: Buffer Shape Mismatch
**Severity**: MEDIUM
**Description**: GPU buffer dimensions don't match kernel expectations

**Detection**:
```python
def check_buffer_shapes(kernel_config, allocated_buffers):
    """
    Verify buffer shapes match kernel expectations.
    """
    violations = []
    
    expected_shapes = {
        'd_atoms': (kernel_config.total_atoms, 4),  # x, y, z, element
        'd_ca_indices': (kernel_config.total_residues,),
        'd_combined_features': (kernel_config.total_residues, 101),
        'd_frequency_velocity': (kernel_config.n_structures, 2),  # freq, velocity per structure
    }
    
    for buffer_name, expected_shape in expected_shapes.items():
        if buffer_name not in allocated_buffers:
            violations.append(DFVViolation(
                code="DFV-005",
                severity="MEDIUM",
                buffer=buffer_name,
                message=f"Buffer {buffer_name} not allocated"
            ))
            continue
        
        actual_shape = allocated_buffers[buffer_name].shape
        if actual_shape != expected_shape:
            violations.append(DFVViolation(
                code="DFV-005",
                severity="MEDIUM",
                buffer=buffer_name,
                expected_shape=expected_shape,
                actual_shape=actual_shape,
                message=f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
            ))
    
    return violations
```

---

## DFV Execution Protocol

### When to Run DFV

1. **After PTX compilation** - Verify kernel signature matches Rust FFI
2. **Before first batch run** - Check buffer allocation
3. **After feature extraction** - Validate feature variance
4. **When accuracy stagnates** - Diagnose pipeline issues

### DFV Diagnostic Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DFV DIAGNOSTIC FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. FEATURE VARIANCE CHECK                                      │
│     └─► All features have variance > 0? ────► YES ──► PASS      │
│                    │                                            │
│                   NO                                            │
│                    ▼                                            │
│  2. IDENTIFY CONSTANT FEATURES                                  │
│     └─► Which features are constant?                            │
│         F96-F100 (Cycle) ──────► Check frequency/velocity input │
│         F101-F108 (Spike) ─────► Check cycle features first     │
│         F92-F95 (Fitness) ─────► Check ddG computation          │
│                    │                                            │
│                    ▼                                            │
│  3. TRACE INPUT BUFFERS                                         │
│     └─► Is input buffer nullptr?                                │
│                    │                                            │
│              YES   │   NO                                       │
│               ▼    │    ▼                                       │
│         DFV-001   │  DFV-003                                    │
│         Null buf  │  Metadata not propagating                   │
│                    │                                            │
│                    ▼                                            │
│  4. GENERATE FIX RECOMMENDATION                                 │
│     └─► Specific code location and required change              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Example: Diagnosing Your Spike Feature Bug

**Input**: Feature extraction shows F101-F108 all equal 0.5

**DFV Analysis**:
```
╔══════════════════════════════════════════════════════════════════════╗
║                  DFV DIAGNOSTIC REPORT                               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  SYMPTOM: Features F101-F108 constant (0.500) across all structures  ║
║                                                                      ║
║  STEP 1: Check spike feature inputs (F96-F100)                       ║
║          Result: F96-F100 also constant (0.500)                      ║
║          Diagnosis: Problem is upstream in cycle features            ║
║                                                                      ║
║  STEP 2: Check cycle feature inputs                                  ║
║          d_cycle_velocity: nullptr ← FOUND IT                        ║
║          d_cycle_frequency: nullptr ← FOUND IT                       ║
║          Diagnosis: DFV-001 (Null buffer detection)                  ║
║                                                                      ║
║  STEP 3: Trace metadata propagation                                  ║
║          BatchMetadata.frequency: 0.0523 (populated)                 ║
║          BatchMetadata.frequency_velocity: 0.0089 (populated)        ║
║          d_frequency_velocity buffer: NOT ALLOCATED                  ║
║          Diagnosis: DFV-003 (Metadata not reaching kernel)           ║
║                                                                      ║
║  ROOT CAUSE:                                                         ║
║  mega_fused_batch.rs does not allocate d_frequency_velocity buffer   ║
║  or pass it to the kernel launch.                                    ║
║                                                                      ║
║  RECOMMENDED FIX:                                                    ║
║  1. Add to BatchBufferPool:                                          ║
║     d_frequency_velocity: CudaSlice<f32>                             ║
║                                                                      ║
║  2. In build_mega_batch(), pack frequency/velocity:                  ║
║     for (i, meta) in batch_metadata.iter().enumerate() {             ║
║         h_freq_vel[i * 2 + 0] = meta.frequency;                      ║
║         h_freq_vel[i * 2 + 1] = meta.frequency_velocity;             ║
║     }                                                                ║
║                                                                      ║
║  3. In kernel launch, pass d_frequency_velocity instead of nullptr   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## Integration with Other Agents

### DFV → Statistical Validator (SV)
When DFV detects constant features, SV should:
- Skip discrimination analysis for those features
- Flag features as "pipeline issue, not scientific issue"
- Not count constant features in ablation studies

### DFV → Feature Engineering (FE)
When FE implements new features:
- DFV validates buffer allocation
- DFV confirms non-zero variance
- DFV traces data flow before hypothesis testing

### DFV → Integrity Guardian (IG)
DFV issues are **not** integrity violations:
- Pipeline bugs are engineering issues
- They don't indicate scientific fraud
- But they MUST be fixed before accuracy claims

---

## DFV Checklist for New Features

Before claiming a feature works:

```
□ Buffer allocated with correct size
□ Host data packed correctly
□ GPU transfer completed (cuMemcpyHtoD)
□ Kernel parameter is not nullptr
□ Structure indexing is correct in kernel
□ Feature variance > 0 across structures
□ RISE/FALL distributions differ (discrimination check)
```

If any check fails, fix the pipeline before proceeding.
