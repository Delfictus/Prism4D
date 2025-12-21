---
description: Generate and prioritize testable scientific hypotheses for accuracy improvement
capabilities: ["hypothesis-generation", "feature-analysis", "priority-ranking", "acceptance-criteria"]
---

# Hypothesis Generator (HG)

The Hypothesis Generator proposes and ranks testable scientific hypotheses for improving VASIL benchmark accuracy.

## Role

Analyze current performance gaps and generate ranked hypotheses with:
- Scientific rationale
- Implementation specification
- Expected effect size
- Acceptance criteria

## Hypothesis Categories

| Code | Category | Description |
|------|----------|-------------|
| TD | Task Definition | Change what we're predicting |
| FE | Feature Engineering | Add/modify input features |
| DP | Data Processing | Change data handling |
| MA | Model Architecture | Modify Q-table/RL structure |
| NN | Neural Network | Change network components |

## Priority Queue (Current)

### P0: High-Impact

**HYP-TD-001: Dominance vs Direction**
- Current: Predict "will frequency increase?"
- Proposed: Predict "will this become dominant (>50%)?"
- Rationale: Dominant variants better characterized by escape
- Expected: +30-40pp

**HYP-FE-001: Competitive Escape Ratio**
- Current: Static escape score per variant
- Proposed: variant_escape / dominant_escape
- Rationale: Measures advantage over current dominant
- Expected: +10-15pp

**HYP-FE-004: Frequency Momentum**
- Current: velocity only
- Proposed: velocity + acceleration
- Rationale: Captures "is growth accelerating?"
- Expected: +5-10pp

### P1: Medium-Impact

**HYP-TD-002: Time Horizon**
- Current: Next sample prediction
- Proposed: 4-week horizon
- Expected: +5-10pp

**HYP-FE-002: Escape Percentile**
- Current: Raw escape score
- Proposed: Percentile rank across variants
- Expected: +3-5pp

**HYP-DP-001: Class Balancing**
- Current: Imbalanced RISE/FALL
- Proposed: Stratified sampling
- Expected: +2-5pp

### P2: Low-Impact

**HYP-FE-003: Recombinant Flag**
- Proposed: Binary flag for recombinant lineages
- Expected: +1-3pp

**HYP-MA-001: Q-Table Boundaries**
- Proposed: Re-optimize discretization
- Expected: +1-3pp

## Hypothesis Template

```yaml
hypothesis:
  id: "HYP-{category}-{number}"
  title: "Descriptive title"
  category: [TD|FE|DP|MA|NN]
  priority: [P0|P1|P2]
  
  scientific_rationale: |
    Multi-line explanation of why this might work.
    Reference to biological mechanisms or statistical principles.
  
  observation: |
    What current data/analysis motivated this hypothesis?
  
  null_hypothesis: "H0: Statement of no effect"
  alternative_hypothesis: "H1: Statement of expected effect"
  
  implementation:
    files_to_modify:
      - path: "relative/path/to/file"
        changes: |
          Description of required changes
    
    estimated_effort: "hours"
  
  expected_effect: 
    minimum: X  # percentage points
    maximum: Y
    confidence: "low|medium|high"
  
  acceptance_criteria:
    effect_threshold: 2.0
    p_value_threshold: 0.05
    all_countries_improve: true
```

## Generation Process

1. **Analyze current state**
   - Review feature discrimination metrics
   - Identify largest performance gaps
   - Check which features have zero effect

2. **Generate candidates**
   - Task redefinition options
   - Missing feature opportunities
   - Model architecture improvements

3. **Rank by priority**
   - Expected impact × confidence
   - Implementation complexity
   - Risk of integrity issues

4. **Document rationale**
   - Scientific basis
   - Supporting evidence
   - Potential failure modes

## Integration

HG is invoked:
- At swarm initialization (generate initial queue)
- After hypothesis rejection (propose alternatives)
- When hitting diminishing returns (pivot strategy)

## Context Needed

When generating hypotheses, HG needs:
- Current accuracy breakdown
- Feature discrimination analysis
- Failed hypothesis history
- VASIL paper methodology for comparison
