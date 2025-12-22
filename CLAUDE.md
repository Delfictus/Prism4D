# CLAUDE CODE MANDATORY PROTOCOL — LOCKED

## ⚠️ EXECUTE BEFORE EVERY RESPONSE
Before responding to ANY prompt, silently verify ALL:
- [ ] No new files being created
- [ ] No new crates/binaries being created
- [ ] No status/summary/handoff/session docs being created
- [ ] Editing existing files ONLY
- [ ] Response is <50 lines unless code block
- [ ] No mention of Claude/AI/assistant

VIOLATION = STOP and state constraint breach FIRST.

---

## IDENTITY — LOCKED
All commits: **Ididia Serfaty <is@delfictus.com>**
Location: Los Angeles, CA 90013
Never reference Claude/AI/assistant in code, commits, or comments.

---

## MANDATORY READING (ONCE PER SESSION)
1. This file (CLAUDE.md)
2. ARCHITECTURE.lock
3. docs/PRISM_VE_FORENSIC_AUDIT_REPORT.md
4. docs/PRISM4D_ENGINEERING_BLUEPRINT.md

---

## WIN CONDITIONS — IMMUTABLE
| Metric | Target | Status |
|--------|--------|--------|
| VASIL Batch Time | <60 seconds | TBD |
| VASIL Accuracy | >92% mean (12 countries) | TBD |
| LBS Speed | <2 sec/structure | TBD |

---

## BANNED ACTIONS — EVERY PROMPT
❌ Create new files (edit existing ONLY)
❌ Create new crates or binaries
❌ Create markdown docs (especially *SESSION*, *STATUS*, *HANDOFF*, *SUMMARY*, *COMPLETE*, *REPORT*)
❌ Mock data or synthetic results
❌ Silent fallbacks that mask failures
❌ Verbose explanations (be TERSE)
❌ Refactor working code without explicit request
❌ Architecture changes without explicit approval
❌ Duplicate existing functionality
❌ Add dependencies without explicit approval

---

## REQUIRED ACTIONS — EVERY PROMPT
✅ Edit existing files only
✅ Fail loudly with clear error messages
✅ Run `cargo check` before claiming success
✅ Cite specific file:line:column when discussing code
✅ Keep responses concise (<50 lines unless code block)
✅ One solution, not multiple options
✅ Execute, don't explain

---

## CURRENT OBJECTIVE
VASIL 12-country benchmark → diagnose accuracy gap → achieve >92%

---

## COMPETITIVE CONTEXT
- **EVEscape**: BEATEN (19,400x faster, deeper physics with ΔΔG)
- **VASIL**: Target (they use hardcoded linear weights; we use FluxNet RL)

---

## CANONICAL KERNEL INVENTORY — DO NOT DUPLICATE

### Mega-Fused Kernels (5)
| Kernel | Size | Location |
|--------|------|----------|
| mega_fused_vasil_fluxnet.cu | 28KB | crates/prism-gpu/src/kernels/ |
| mega_fused_batch.cu | 142KB | crates/prism-gpu/src/kernels/ |
| mega_fused_pocket_kernel.cu | 45KB | crates/prism-gpu/src/kernels/ |
| mega_fused_lbs_complete.cu | — | crates/prism-gpu/src/kernels/ |
| prism_lbs_fused.cu | — | crates/prism-gpu/src/kernels/ |

### VE-Swarm Kernels (3)
| Kernel | Function |
|--------|----------|
| ve_swarm_agents.cu | 32-agent swarm inference |
| ve_swarm_dendritic_reservoir.cu | Neuromorphic reservoir |
| ve_swarm_temporal_conv.cu | Temporal convolution |

---

## ARCHITECTURE CONSTRAINTS — LOCKED

### The Engine
Custom "Mega-Fused" CUDA kernel optimized for RTX 3060, processing >300 structures/second.

### The Brain
FluxNet RL — Q-learning agent using discretized 6D state space from physics features.
**NOT hardcoded regression formulas.**

### The Differentiator
Integration of Time (Cycle Module) and Spiking Dynamics (Neuromorphic input) into structural prediction.

---

## CRATE INVENTORY — DO NOT ADD NEW

### Core (19 crates in crates/)
prism, prism-cli, prism-core, prism-escape-extract, prism-fluxnet, prism-geometry, prism-gnn, prism-gpu, prism-lbs, prism-mec, prism-ontology, prism-phases, prism-physics, prism-pipeline, prism-ve, prism-ve-bench, prism-whcr

### Foundation (in foundation/)
mathematics, neuromorphic, prct-core, quantum, shared-types

---

## KNOWN BUGS — FIX THESE, DON'T WORK AROUND

1. **LBS Blob Bug**: merge_overlapping_pockets/expand_pocket disabled (lib.rs.blob_bug)
2. **API Mismatch**: detect_pockets expects 9 args, some calls pass 7

---

## DATA LOCATIONS — CANONICAL

| Data | Path |
|------|------|
| VASIL 12-country | data/VASIL/ByCountry/ |
| CryptoBench golden | data/cryptobench_2025_GOLDEN_pockets.csv |
| Trained Q-table | validation_results/fluxnet_ve_qtable.json |
| Thresholds | validation_results/fluxnet_threshold_optimized.json |
| PTX kernels | kernels/ptx/*.ptx |

---

## RESPONSE FORMAT — ENFORCED
```
[ACTION]: <what you're doing in ≤10 words>
[FILE]: <path:line if editing>
[CODE]: <code block if any>
[RESULT]: <outcome in ≤10 words>
```

No preamble. No summary. No "Let me explain." Execute.

---

**THIS FILE IS IMMUTABLE. DO NOT MODIFY.**
