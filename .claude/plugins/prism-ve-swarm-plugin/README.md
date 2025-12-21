# PRISM-4D VE Swarm Plugin

Multi-agent swarm for PRISM-4D viral evolution optimization with scientific integrity guarantees.

## Overview

This Claude Code plugin provides a coordinated 9-agent system for achieving 92%+ accuracy on the VASIL benchmark while ensuring publication-grade scientific integrity.

## Installation

```bash
# From local directory
/plugin marketplace add ./prism-ve-swarm-plugin

# Install the plugin
/plugin install prism-ve-swarm
```

## Agents

| # | Agent | Role | Power |
|---|-------|------|-------|
| 1 | Orchestrator (OA) | Coordination | — |
| 2 | Integrity Guardian (IG) | Fraud prevention | **VETO** |
| 3 | Data Flow Validator (DFV) | Pipeline debugging | Diagnostic |
| 4 | Hypothesis Generator (HG) | Propose experiments | — |
| 5 | Feature Engineering (FE) | Implementation | — |
| 6 | Statistical Validator (SV) | Rigor | — |
| 7 | Ablation Study (AS) | Attribution | — |
| 8 | Cross-Validation (CV) | Generalization | — |
| 9 | Literature Alignment (LA) | Comparison | — |

## Slash Commands

| Command | Description |
|---------|-------------|
| `/prism-ve-swarm:swarm-init` | Initialize swarm and establish baseline |
| `/prism-ve-swarm:swarm-cycle` | Run a hypothesis testing cycle |
| `/prism-ve-swarm:swarm-status` | Check optimization progress |
| `/prism-ve-swarm:dfv-check` | Run Data Flow Validator |
| `/prism-ve-swarm:integrity-audit` | Run Integrity Guardian audit |
| `/prism-ve-swarm:hypothesis-test` | Test a specific hypothesis |

## Quick Start

```bash
# 1. Initialize the swarm
/prism-ve-swarm:swarm-init

# 2. Check for pipeline issues
/prism-ve-swarm:dfv-check

# 3. Run optimization cycle
/prism-ve-swarm:swarm-cycle

# 4. Check progress
/prism-ve-swarm:swarm-status
```

## The Integrity Oath

```
"I will not use future information to predict the past.
 I will not train on test data or test on train data.
 I will not hardcode coefficients from the paper I'm trying to beat.
 I will not cherry-pick results or hide failures.
 I will document every modification and its rationale.
 I will ensure every result is reproducible.
 I will report confidence intervals, not just point estimates.
 I will acknowledge when my method differs from VASIL."
```

## Win Conditions

- **Speed**: <60 seconds batch pipeline
- **Accuracy**: >92.0% mean on 12-country VASIL benchmark
- **Speedup**: 19,400x vs EVEscape

## Files Structure

```
prism-ve-swarm-plugin/
├── .claude-plugin/
│   ├── plugin.json          # Plugin manifest
│   └── marketplace.json     # Marketplace config
├── commands/
│   ├── swarm-init.md        # Initialize swarm
│   ├── swarm-cycle.md       # Run hypothesis cycle
│   ├── swarm-status.md      # Check status
│   ├── dfv-check.md         # Data flow validation
│   ├── integrity-audit.md   # Integrity check
│   └── hypothesis-test.md   # Test hypothesis
├── agents/
│   ├── integrity-guardian.md
│   ├── data-flow-validator.md
│   ├── hypothesis-generator.md
│   └── statistical-validator.md
├── skills/
│   └── prism-ve-swarm/
│       ├── SKILL.md
│       ├── references/
│       └── scripts/
└── README.md
```

## License

MIT
