# SARS-CoV-2 Spike RBD Mutations Reference

## Overview

The Receptor Binding Domain (RBD) of the SARS-CoV-2 spike protein spans residues 319-541, with the Receptor Binding Motif (RBM) at 438-506. Mutations in this region affect ACE2 binding affinity and antibody escape.

## Critical Mutation Positions

### Position 417 (K417N/T)

**Wild Type**: Lysine (K) - positively charged, forms salt bridge with ACE2 D30

| Mutation | ΔΔG Binding | ACE2 Affinity | Immune Escape | Variants |
|----------|-------------|---------------|---------------|----------|
| K417N | +0.4 kcal/mol | Decreased 2-4x | Strong (Class 1 mAbs) | Beta, Omicron |
| K417T | +0.3 kcal/mol | Decreased 2x | Strong (Class 1 mAbs) | Gamma |

**Mechanism**: Loss of salt bridge with ACE2-D30 reduces binding but escapes antibodies targeting the 417 epitope.

**Structure Impact**: K417 sits at the edge of the RBM, making a hydrogen bond with ACE2-D30 across the interface.

### Position 452 (L452R/Q)

**Wild Type**: Leucine (L) - hydrophobic

| Mutation | ΔΔG Binding | ACE2 Affinity | Immune Escape | Variants |
|----------|-------------|---------------|---------------|----------|
| L452R | -0.3 kcal/mol | Increased ~2x | Moderate (Class 2/3 mAbs) | Delta, Kappa |
| L452Q | -0.2 kcal/mol | Slight increase | Moderate | Lambda |

**Mechanism**: Introduction of positive charge creates new electrostatic interactions with ACE2-E35.

### Position 478 (T478K)

**Wild Type**: Threonine (T) - polar

| Mutation | ΔΔG Binding | ACE2 Affinity | Immune Escape | Variants |
|----------|-------------|---------------|---------------|----------|
| T478K | +0.1 kcal/mol | Neutral | Modest | Delta, Omicron |

**Mechanism**: Modest effect alone, but synergistic with L452R in Delta.

### Position 484 (E484K/Q/A)

**Wild Type**: Glutamic acid (E) - negatively charged

| Mutation | ΔΔG Binding | ACE2 Affinity | Immune Escape | Variants |
|----------|-------------|---------------|---------------|----------|
| E484K | -0.2 kcal/mol | Increased ~1.5x | Very Strong | Beta, Gamma, Zeta |
| E484Q | -0.1 kcal/mol | Slight increase | Strong | Kappa |
| E484A | +0.1 kcal/mol | Decreased | Very Strong | Omicron |

**Mechanism**: Charge reversal (E→K) creates favorable interaction with ACE2-K31 and disrupts most Class 2 antibody binding.

**Structure Impact**: E484 is on a flexible loop, mutations here have outsized effect on antibody recognition.

### Position 490 (F490S)

**Wild Type**: Phenylalanine (F) - aromatic, hydrophobic

| Mutation | ΔΔG Binding | ACE2 Affinity | Immune Escape | Variants |
|----------|-------------|---------------|---------------|----------|
| F490S | -0.5 kcal/mol | Increased | Moderate | Some Omicron lineages |

### Position 493 (Q493R)

**Wild Type**: Glutamine (Q) - polar

| Mutation | ΔΔG Binding | ACE2 Affinity | Immune Escape | Variants |
|----------|-------------|---------------|---------------|----------|
| Q493R | -0.4 kcal/mol | Increased | Moderate | Omicron BA.1 |

**Mechanism**: Creates new salt bridge with ACE2-E35.

### Position 498 (Q498R)

**Wild Type**: Glutamine (Q) - polar

| Mutation | ΔΔG Binding | ACE2 Affinity | Immune Escape | Variants |
|----------|-------------|---------------|---------------|----------|
| Q498R | -0.3 kcal/mol | Increased | Moderate | Omicron |

**Mechanism**: Strong epistatic effect with N501Y - together they synergistically increase binding.

### Position 501 (N501Y/T)

**Wild Type**: Asparagine (N) - polar

| Mutation | ΔΔG Binding | ACE2 Affinity | Immune Escape | Variants |
|----------|-------------|---------------|---------------|----------|
| N501Y | -1.2 kcal/mol | Increased ~10x | Minimal | Alpha, Beta, Gamma, Omicron |
| N501T | -0.3 kcal/mol | Increased ~2x | Minimal | Rare |

**Mechanism**: Tyrosine ring stacks with ACE2-Y41, creating strong π-π interaction.

**Structure Impact**: N501 is at the tip of the RBM, making direct contact with ACE2. The Y substitution creates aromatic stacking.

**Epistatic Effects**: N501Y is a "gateway" mutation that enables other mutations (Q498R, Y505H) to further increase binding.

### Position 505 (Y505H)

**Wild Type**: Tyrosine (Y) - aromatic

| Mutation | ΔΔG Binding | ACE2 Affinity | Immune Escape | Variants |
|----------|-------------|---------------|---------------|----------|
| Y505H | +0.2 kcal/mol | Decreased | Minimal | Omicron |

**Mechanism**: Loss of aromatic ring reduces stacking, but compensated by other mutations in Omicron.

## Variant Mutation Combinations

### Alpha (B.1.1.7)
```
RBD: N501Y only
Effect: ~50% increased transmissibility
ACE2 affinity: ~10x increase
```

### Beta (B.1.351)
```
RBD: K417N + E484K + N501Y
Effect: Significant immune escape
ACE2 affinity: ~4x increase (N501Y+E484K compensate for K417N)
```

### Gamma (P.1)
```
RBD: K417T + E484K + N501Y
Effect: Similar to Beta, slightly different escape profile
ACE2 affinity: ~3-4x increase
```

### Delta (B.1.617.2)
```
RBD: L452R + T478K
Effect: Increased transmissibility, moderate escape
ACE2 affinity: ~2x increase
```

### Omicron BA.1 (B.1.1.529)
```
RBD: G339D, S371L, S373P, S375F, K417N, N440K, G446S, S477N, 
     T478K, E484A, Q493R, G496S, Q498R, N501Y, Y505H

Total: 15 RBD mutations
Effect: Massive immune escape, maintained transmissibility
ACE2 affinity: Despite many destabilizing mutations, maintained by N501Y+Q498R+Q493R
```

## ACE2 Interface Contacts

Residues within 4Å of ACE2 (from 6M0J):
```
K417  - D30 (salt bridge)
G446  - Q42
Y449  - D38, Q42
Y453  - H34
L455  - D30, K31
F456  - D30, K31, T27
A475  - S19, Q24
F486  - M82, Y83
N487  - Q24, Y83
Y489  - K31, T27, F28
Q493  - K31, E35, H34
G496  - K353, D38
Q498  - K353, Q42, D355
T500  - Y41, D355
N501  - Y41, K353
G502  - K353
Y505  - E37
```

## PRISM-4D Feature Extraction

For Stage 7 (Fitness Module), calculate for each mutation:

```rust
/// ΔΔG_binding approximation based on residue properties
fn estimate_ddg_binding(
    wt_residue: u8,
    mut_residue: u8,
    burial: f32,
    centrality: f32,
) -> f32 {
    let hydro_change = HYDROPHOBICITY[mut_residue as usize] 
                     - HYDROPHOBICITY[wt_residue as usize];
    let volume_change = VOLUME[mut_residue as usize]
                      - VOLUME[wt_residue as usize];
    
    // Interface residues: hydrophobicity change matters more
    // Buried residues: volume change matters more
    let interface_score = hydro_change * centrality * (1.0 - burial);
    let stability_score = volume_change.abs() * burial;
    
    interface_score - stability_score * 0.5
}
```

## Escape Classification

For VASIL benchmark, mutations are classified by their evolutionary trajectory:

```
RISE:  Frequency increasing (variant emerging)
       - New escape mutations
       - Increased transmissibility mutations
       
FALL:  Frequency decreasing (variant waning)
       - Outcompeted by fitter variants
       - Population immunity reducing advantage
```

Key predictors:
1. **Immune escape score**: Higher = more likely RISE in immune population
2. **ACE2 affinity change**: Higher = more likely RISE
3. **Current frequency**: Very high = more likely FALL (already peaked)
4. **Velocity**: Positive = RISE, Negative = FALL

## References

1. Barton et al. (2021) eLife - "Effects of common mutations in the SARS-CoV-2 Spike RBD"
2. Starr et al. (2020) Cell - "Deep Mutational Scanning of SARS-CoV-2 RBD"
3. Bloom Lab DMS data: https://jbloomlab.github.io/SARS2_RBD_Ab_escape_maps/
