---
name: prism-ve-swarm
description: "Multi-agent swarm for PRISM-4D viral evolution optimization with scientific integrity guarantees. Use when: (1) Optimizing VASIL benchmark accuracy, (2) Testing feature engineering hypotheses, (3) Validating prediction methodology, (4) Preparing publishable results. Coordinates 9 specialized agents with integrity-first architecture ensuring defensible, reproducible science. Trigger: 'optimize VE', 'run swarm', 'hypothesis test', 'validate accuracy', 'check data flow'."
---

# PRISM-4D Viral Evolution Multi-Agent Optimization Swarm

Coordinated 9-agent system for achieving 92%+ VASIL accuracy with publication-grade scientific integrity.

## Swarm Architecture Overview

```
                         ┌─────────────────────────────┐
                         │   ORCHESTRATOR AGENT (OA)   │
                         │   Coordinates all agents    │
                         └──────────────┬──────────────┘
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         │                              │                              │
         ▼                              ▼                              ▼
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│   INTEGRITY     │◄────────►│   HYPOTHESIS    │◄────────►│    FEATURE      │
│   GUARDIAN      │          │   GENERATOR     │          │   ENGINEERING   │
│   (IG)          │          │   (HG)          │          │   (FE)          │
│   ⚠️ VETO POWER │          └─────────────────┘          └─────────────────┘
└─────────────────┘                    │                           │
         │                             ▼                           │
         │    ┌────────────────────────────────────────────────┐   │
         │    │            DATA FLOW VALIDATOR (DFV)           │   │
         │    │   🔍 Pipeline integrity & null detection       │   │
         │    │   Catches: nullptr inputs, zero variance,      │   │
         │    │   missing metadata, buffer mismatches          │   │
         │    └────────────────────────┬───────────────────────┘   │
         │                             │                           │
         │                             ▼                           │
         │                   ┌─────────────────┐                   │
         └──────────────────►│   STATISTICAL   │◄──────────────────┘
                             │   VALIDATOR     │
                             │   (SV)          │
                             └────────┬────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│   ABLATION      │        │ CROSS-VALIDATION│        │   LITERATURE    │
│   STUDY         │        │    AGENT        │        │   ALIGNMENT     │
│   (AS)          │        │    (CV)         │        │   (LA)          │
└─────────────────┘        └─────────────────┘        └─────────────────┘
```

## Agent Specifications

### 1. Orchestrator Agent (OA)
**Role**: Central coordinator, workflow manager
**Responsibilities**:
- Sequence agent activations
- Aggregate results across agents
- Maintain experiment state
- Enforce stopping conditions

**Activation**: Always active during swarm execution

### 2. Integrity Guardian (IG)
**Role**: Scientific fraud prevention, bias detection
**Responsibilities**:
- Verify no look-ahead bias in features
- Detect data leakage between train/test
- Validate temporal integrity
- Flag coefficient hardcoding
- Audit reproducibility

**Activation**: Before and after every other agent
**Veto Power**: YES - can halt entire swarm

See `references/INTEGRITY_PROTOCOLS.md` for detailed rules.

### 3. Data Flow Validator (DFV) 🆕
**Role**: GPU pipeline integrity, null/constant detection
**Responsibilities**:
- Detect nullptr inputs where data expected
- Validate feature variance > 0 for non-constant features
- Trace data from BatchMetadata to kernel outputs
- Verify per-structure indexing correctness
- Catch buffer shape mismatches

**Activation**: After any feature extraction, before statistical validation
**Diagnostic Power**: YES - can identify root cause of zero-discrimination features

**Critical Checks**:
```
DFV-001: Null buffer detection (nullptr passed to kernel)
DFV-002: Constant feature detection (variance = 0 across structures)
DFV-003: Metadata propagation failure (per-lineage data not reaching kernel)
DFV-004: Index mismatch (structure_idx not mapping to correct metadata)
DFV-005: Buffer shape mismatch (dimensions don't match expected layout)
```

See `references/DATA_FLOW_PROTOCOLS.md` for detailed checks.

### 4. Hypothesis Generator (HG)
**Role**: Propose testable scientific hypotheses
**Responsibilities**:
- Analyze feature distributions
- Generate ranked hypotheses
- Define acceptance criteria
- Track hypothesis outcomes

**Activation**: Beginning of optimization cycle
**Output Format**: Structured hypothesis with null/alternative

See `references/HYPOTHESIS_FRAMEWORK.md` for templates.

### 5. Feature Engineering Agent (FE)
**Role**: Implement and test new features
**Responsibilities**:
- Code feature implementations
- Validate feature computation
- Measure feature discrimination
- Document feature semantics

**Activation**: After hypothesis approval
**Constraint**: Must pass IG validation before testing

### 6. Statistical Validator (SV)
**Role**: Ensure statistical rigor
**Responsibilities**:
- Compute confidence intervals
- Run significance tests
- Detect overfitting
- Validate sample sizes

**Activation**: After every experiment
**Output**: p-values, effect sizes, CI bounds

### 7. Ablation Study Agent (AS)
**Role**: Isolate feature contributions
**Responsibilities**:
- Single-feature ablations
- Feature combination tests
- Measure marginal contributions
- Identify redundant features

**Activation**: After feature improvements verified
**Output**: Feature importance rankings

### 8. Cross-Validation Agent (CV)
**Role**: Ensure generalization
**Responsibilities**:
- K-fold cross-validation
- Leave-one-country-out validation
- Temporal split validation
- Distribution shift detection

**Activation**: Before any result is accepted
**Requirement**: All 12 countries must validate

### 9. Literature Alignment Agent (LA)
**Role**: Ensure VASIL methodology alignment
**Responsibilities**:
- Compare methodology to VASIL paper
- Identify deviations
- Suggest corrections
- Validate fair comparison

**Activation**: Before finalizing any publishable result

## Execution Protocol

### Phase 1: Initialization
```
1. OA: Load current benchmark state
2. IG: Audit existing codebase for integrity violations
3. DFV: Validate GPU pipeline data flow
4. LA: Review VASIL paper methodology
5. OA: Establish baseline metrics
```

### Phase 2: Hypothesis Cycle (Repeat)
```
1. HG: Generate top-3 ranked hypotheses
2. IG: Validate hypotheses for integrity
3. FE: Implement highest-priority hypothesis
4. DFV: Verify data flow to new features
5. IG: Pre-execution integrity check
6. SV: Run experiment with proper validation
7. DFV: Check feature variance/discrimination
8. CV: Cross-validate across countries
9. AS: Ablation if improvement detected
10. IG: Post-execution integrity audit
11. OA: Record results, update state
```

### Phase 3: Finalization
```
1. LA: Final VASIL methodology alignment check
2. SV: Aggregate statistical summary
3. DFV: Final pipeline health report
4. IG: Final integrity certification
5. OA: Generate publication-ready report
```

## Integrity-First Architecture

**CRITICAL**: Scientific integrity is non-negotiable. The swarm implements:

1. **Temporal Firewall**: Test data from 2022-06+ is NEVER visible during training
2. **Coefficient Quarantine**: No VASIL paper coefficients may be used
3. **Look-Ahead Prevention**: Features cannot use future information
4. **Reproducibility Lock**: All experiments must be reproducible from seed
5. **Audit Trail**: Every modification is logged with rationale

## Stopping Conditions

The swarm terminates when ANY of:
- Target accuracy achieved (92%+ mean across 12 countries)
- Maximum iterations reached (50 hypothesis cycles)
- Integrity violation detected (immediate halt)
- Diminishing returns (3 consecutive cycles with delta under 0.5%)

## Output Artifacts

Each swarm execution produces:
1. `experiment_log.json` - Complete audit trail
2. `hypothesis_outcomes.md` - All tested hypotheses with results
3. `ablation_report.md` - Feature contribution analysis
4. `statistical_summary.md` - All p-values, CIs, effect sizes
5. `integrity_certificate.md` - IG sign-off on methodology
6. `publication_draft.md` - Camera-ready results section

## Quick Start

```bash
# Initialize swarm state
python scripts/init_swarm.py --baseline

# Run single hypothesis cycle
python scripts/run_cycle.py --hypothesis "competitive_escape_ratio"

# Full swarm optimization
python scripts/run_swarm.py --max-cycles 50 --target-accuracy 0.92

# Verify integrity post-hoc
python scripts/integrity_audit.py --experiment-log experiment_log.json
```

## Critical Files

See `references/AGENT_SPECIFICATIONS.md` for detailed agent protocols.
See `references/INTEGRITY_PROTOCOLS.md` for integrity rules.
See `references/VALIDATION_FRAMEWORK.md` for statistical requirements.
See `references/HYPOTHESIS_FRAMEWORK.md` for hypothesis templates.
