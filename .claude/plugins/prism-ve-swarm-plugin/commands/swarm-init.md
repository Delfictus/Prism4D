---
description: Initialize the PRISM-4D VE Swarm for optimization
---

# Initialize PRISM-4D Viral Evolution Swarm

Initialize the multi-agent swarm for optimizing VASIL benchmark accuracy.

## What this does

1. **Loads the Master Blueprint** from project knowledge
2. **Audits existing codebase** for integrity violations (IG agent)
3. **Validates GPU pipeline** data flow (DFV agent)
4. **Establishes baseline metrics** from current benchmark run
5. **Creates experiment_log.json** for audit trail

## Pre-flight checks

Before initializing, ensure:
- [ ] PRISM project is accessible at expected path
- [ ] PTX files are compiled and up-to-date
- [ ] VASIL data is available in `/mnt/f/VASIL_Data/ByCountry/`
- [ ] Rust toolchain is configured

## Initialization Protocol

```
PHASE 1: CODEBASE AUDIT
=======================
1. Run Integrity Guardian (IG) scan for:
   - Forbidden coefficients (0.65, 0.35, 0.92)
   - Look-ahead bias patterns
   - Train/test leakage

2. Run Data Flow Validator (DFV) scan for:
   - Null buffer detections (DFV-001)
   - Constant features (DFV-002)
   - Metadata propagation failures (DFV-003)

PHASE 2: BASELINE ESTABLISHMENT
===============================
3. Execute current benchmark:
   cargo run --release -p prism-ve-bench -- --countries all

4. Record baseline metrics:
   - Mean accuracy across 12 countries
   - Per-country accuracy breakdown
   - Feature discrimination analysis

PHASE 3: STATE INITIALIZATION
=============================
5. Create swarm_state.json with:
   - baseline_accuracy: <measured>
   - cycle_number: 0
   - integrity_status: CLEAN | VIOLATIONS_FOUND
   - pipeline_status: HEALTHY | ISSUES_FOUND
```

## Arguments

$ARGUMENTS

If no arguments provided, uses default PRISM path and full initialization.

Options:
- `--path <project_path>` - Override PRISM project location
- `--skip-baseline` - Skip benchmark run (use existing results)
- `--quick` - Only run DFV, skip full IG audit

## Output

Creates in project root:
- `swarm_state.json` - Current optimization state
- `experiment_log.json` - Audit trail
- `dfv_report.md` - Pipeline health report
- `integrity_report.md` - Codebase integrity report

## Next Steps

After initialization:
1. Review any DFV violations and fix pipeline issues first
2. Run `/prism-ve-swarm:swarm-cycle` to begin hypothesis testing
3. Use `/prism-ve-swarm:swarm-status` to monitor progress
