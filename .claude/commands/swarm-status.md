---
description: Check current status of the PRISM-4D VE Swarm optimization
---

# Swarm Status

Display current optimization status, progress, and next actions.

## What this shows

1. **Current accuracy** vs target (92%)
2. **Cycles completed** / max (50)
3. **Active hypothesis** being tested
4. **Recent results** from last 3 cycles
5. **Integrity status** (clean/violations)
6. **Pipeline status** (healthy/issues)
7. **Estimated cycles to target**

## Status Dashboard

```
╔══════════════════════════════════════════════════════════════════════╗
║                    PRISM-4D VE SWARM STATUS                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  TARGET: 92.0% accuracy              CURRENT: XX.X% accuracy         ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   ║
║  Progress: [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] XX%           ║
║                                                                      ║
║  Cycles: XX/50                       Delta from baseline: +XX.X pp   ║
║  Integrity: ✓ CLEAN                  Pipeline: ✓ HEALTHY             ║
║                                                                      ║
║  ──────────────────────────────────────────────────────────────────  ║
║  RECENT CYCLES                                                       ║
║  ──────────────────────────────────────────────────────────────────  ║
║  Cycle XX: HYP-XX-XXX  [ACCEPTED] +X.X pp  p=0.XXX                   ║
║  Cycle XX: HYP-XX-XXX  [REJECTED] -X.X pp  p=0.XXX                   ║
║  Cycle XX: HYP-XX-XXX  [ACCEPTED] +X.X pp  p=0.XXX                   ║
║                                                                      ║
║  ──────────────────────────────────────────────────────────────────  ║
║  PER-COUNTRY ACCURACY                                                ║
║  ──────────────────────────────────────────────────────────────────  ║
║  Germany:  XX.X%    USA:      XX.X%    UK:       XX.X%               ║
║  Japan:    XX.X%    Brazil:   XX.X%    France:   XX.X%               ║
║  Canada:   XX.X%    Denmark:  XX.X%    Australia: XX.X%              ║
║  Sweden:   XX.X%    Mexico:   XX.X%    S.Africa: XX.X%               ║
║                                                                      ║
║  ──────────────────────────────────────────────────────────────────  ║
║  NEXT ACTION                                                         ║
║  ──────────────────────────────────────────────────────────────────  ║
║  Hypothesis: HYP-XX-XXX (Title)                                      ║
║  Expected delta: +X-X pp                                             ║
║  Run: /prism-ve-swarm:swarm-cycle                                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

## Arguments

$ARGUMENTS

Usage:
- `/prism-ve-swarm:swarm-status` - Full dashboard
- `/prism-ve-swarm:swarm-status --brief` - One-line summary
- `/prism-ve-swarm:swarm-status --history` - Full cycle history
- `/prism-ve-swarm:swarm-status --countries` - Detailed per-country breakdown

## Status Indicators

| Indicator | Meaning |
|-----------|---------|
| ✓ CLEAN | No integrity violations |
| ✓ HEALTHY | No pipeline issues (DFV) |
| ⚠️ WARNINGS | Non-blocking issues detected |
| 🛑 BLOCKED | Critical issues must be fixed |
| 🎯 TARGET MET | 92%+ achieved |

## Files Read

- `swarm_state.json` - Current state
- `experiment_log.json` - Cycle history
- Latest benchmark results
