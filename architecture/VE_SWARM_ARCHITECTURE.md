# PRISM VE-SWARM: Revolutionary Viral Variant Prediction Architecture

## Executive Summary

Current viral variant prediction achieves only ~53% accuracy (barely above random) due to:
1. **Averaging destroys signal**: 109-dim x N_residue tensor reduced to single averages
2. **Linear decision boundaries**: Cannot capture non-linear variant dynamics
3. **No spatial exploitation**: Ignoring which residues are near ACE2 binding sites
4. **No temporal exploitation**: Not using frequency time series patterns
5. **Inverted velocity signal**: High velocity = at PEAK, about to FALL

**VE-Swarm Solution**: A GPU-accelerated swarm intelligence architecture that:
- Preserves FULL 109-dim x N_residue tensor through graph neural operations
- Uses dendritic reservoir computing on residue contact graph
- Implements structural attention weighted by binding site proximity
- Applies temporal convolutions over frequency time series
- Coordinates swarm agents for emergent feature discovery

**Target**: 75-85% accuracy (vs current 53%)

---

## 1. Data Flow Architecture

```
                                    VE-SWARM ARCHITECTURE

    INPUT LAYER                     SWARM PROCESSING                      OUTPUT
    ============                    =================                      ======

    [Residue Features]              [Dendritic Graph Reservoir]
    109-dim x N_residues  --------> GPU: Propagate on contact graph
         |                          Multi-branch neuromorphic dynamics
         |                                    |
         v                                    v
    [Spatial Attention]             [Binding Site Attention Mask]
    Which residues matter? <------- Learned: ACE2 interface = high weight
         |                          RBD epitopes = high weight
         |                                    |
         v                                    v
    [Swarm Agents]                  [32 GPU Agent Threads]
    109-dim features x 32 --------> Each agent: different feature subset
         |                          Compete + cooperate via pheromones
         |                                    |
         v                                    v
    [Temporal Conv]                 [Frequency Time Series]        -----> [RISE/FALL]
    1D Conv over weeks   <--------- 52 weeks x variant features            Binary
         |                          Multi-scale: 1w, 2w, 4w, 8w            Prediction
         |                                    |
         v                                    v
    [FluxNet RL]                    [Reward: Prediction Accuracy]
    Adapt weights online  <-------- Update swarm coordination
```

---

## 2. Key Innovations

### 2.1 Dendritic Residue Graph Reservoir (DRGR)

**Problem**: Current approach averages 109-dim features across residues, losing spatial structure.

**Solution**: Treat the protein as a graph where:
- Nodes = Residues (each with 109-dim features)
- Edges = Contact map (< 8A between CA atoms)

**Dendritic Computation**:
```
For each residue i:
    - Proximal dendrite: Local features (self)
    - Basal dendrite: Neighbor features (1-hop contacts)
    - Apical dendrite: Global features (eigenvector centrality)
    - Spine: Historical state (recurrent)

Output: 32-dim "reservoir state" per residue that encodes
        BOTH local features AND graph topology
```

**GPU Implementation**: One warp per residue, parallel across all residues.

### 2.2 Binding Site Structural Attention

**Problem**: All residues treated equally, but only ~25 residues at ACE2 interface matter for binding.

**Solution**: Learned attention weights based on:
1. Distance to ACE2 interface (from 6M0J structure)
2. Epitope class membership (10 antibody classes)
3. Known escape mutation sites (from DMS data)

**Attention Formula**:
```
attention[i] = softmax(
    alpha * interface_dist[i] +
    beta * epitope_score[i] +
    gamma * escape_hotspot[i] +
    delta * reservoir_state[i]
)

where alpha, beta, gamma, delta are learned via FluxNet RL
```

### 2.3 Swarm Feature Discovery

**Problem**: 109 features but unknown which combinations predict RISE vs FALL.

**Solution**: 32 GPU "swarm agents" that each:
1. Focus on different feature subsets (random initialization)
2. Make independent predictions
3. Communicate via "pheromone" signals (shared memory)
4. Evolve: successful agents reproduce, failing agents die

**Swarm Dynamics**:
```
Agent j:
    - Feature mask: binary[109] (which features to use)
    - Prediction: sigmoid(dot(mask * features, weights))
    - Fitness: correct predictions / total predictions

Every 100 predictions:
    - Top 8 agents: Reproduce (crossover feature masks)
    - Bottom 8 agents: Die (replaced by offspring)
    - Middle 16 agents: Mutate (flip random features)

Pheromone update:
    - Successful feature combinations leave "pheromone trail"
    - All agents bias toward high-pheromone features
```

### 2.4 Temporal Convolution Stack

**Problem**: Using single-week snapshots, missing trajectory information.

**Solution**: 1D temporal convolutions over 52-week frequency time series:

```
Input: [52 weeks x variant_features]

Conv Layers:
    - Conv1D(k=3, d=1): 1-week patterns (noise)
    - Conv1D(k=3, d=2): 2-week patterns (early growth)
    - Conv1D(k=3, d=4): 4-week patterns (sustained growth)
    - Conv1D(k=3, d=8): 8-week patterns (wave dynamics)

Pooling: Global max + global average

Output: 64-dim temporal embedding
```

**Key Insight**: RISE variants show specific temporal signatures:
- Accelerating frequency increase (2nd derivative > 0)
- S-curve shape (logistic growth)
- Correlation with declining competitors

### 2.5 Physics-Informed Constraints

**Problem**: Pure ML ignores known viral evolution physics.

**Solution**: Hard-code constraints from evolutionary biology:

1. **Fitness Landscape Constraint**:
   - Variants cannot have arbitrarily high escape + binding
   - Trade-off: escape mutations often reduce binding

2. **Frequency Dynamics**:
   - Variants cannot exceed 100% frequency
   - Logistic growth: dF/dt = gamma * F * (1-F)

3. **Cross-Immunity**:
   - Variants sharing epitopes compete
   - Negative correlation in fitness

---

## 3. GPU Kernel Specifications

### 3.1 Kernel: ve_swarm_dendritic_reservoir

```cuda
/**
 * Dendritic Reservoir on Residue Contact Graph
 *
 * Input:
 *   - features[N_residues x 109]: Per-residue feature tensor
 *   - contact_csr_row[N_residues+1]: CSR row pointers
 *   - contact_csr_col[N_edges]: CSR column indices
 *   - contact_weights[N_edges]: Contact strengths
 *
 * Output:
 *   - reservoir_state[N_residues x 32]: Dendritic reservoir output
 *
 * GPU Layout:
 *   - 1 warp (32 threads) per residue
 *   - Each thread computes 1 of 32 reservoir neurons
 *   - Shared memory for neighbor aggregation
 */
extern "C" __global__ void ve_swarm_dendritic_reservoir(
    const float* __restrict__ features,      // [N x 109]
    const int* __restrict__ csr_row,         // [N+1]
    const int* __restrict__ csr_col,         // [E]
    const float* __restrict__ csr_weight,    // [E]
    float* __restrict__ reservoir_state,     // [N x 32]
    const int N_residues,
    const int iterations
);
```

### 3.2 Kernel: ve_swarm_structural_attention

```cuda
/**
 * Structural Attention over Residue Graph
 *
 * Computes attention weights based on:
 * 1. Distance to ACE2 interface
 * 2. Epitope class scores
 * 3. Reservoir state importance
 *
 * Input:
 *   - reservoir_state[N x 32]: From dendritic reservoir
 *   - interface_dist[N]: Distance to ACE2 interface
 *   - epitope_scores[N x 10]: Per-epitope escape
 *   - attention_params[4]: Learned alpha, beta, gamma, delta
 *
 * Output:
 *   - attention[N]: Softmax attention weights
 *   - attended_features[109]: Weighted sum of features
 */
extern "C" __global__ void ve_swarm_structural_attention(
    const float* __restrict__ reservoir_state,
    const float* __restrict__ interface_dist,
    const float* __restrict__ epitope_scores,
    const float* __restrict__ attention_params,
    float* __restrict__ attention_weights,
    float* __restrict__ attended_features,
    const int N_residues
);
```

### 3.3 Kernel: ve_swarm_agents

```cuda
/**
 * Swarm Agent Computation
 *
 * 32 agents compete to find best feature combinations.
 * Uses warp-level voting for efficient coordination.
 *
 * Input:
 *   - attended_features[109]: From structural attention
 *   - agent_masks[32 x 109]: Binary feature masks
 *   - agent_weights[32 x 109]: Per-agent linear weights
 *   - temporal_embedding[64]: From temporal conv
 *
 * Output:
 *   - agent_predictions[32]: Each agent's prediction
 *   - pheromone_update[109]: Feature importance signal
 */
extern "C" __global__ void ve_swarm_agents(
    const float* __restrict__ attended_features,
    const float* __restrict__ agent_masks,
    const float* __restrict__ agent_weights,
    const float* __restrict__ temporal_embedding,
    float* __restrict__ agent_predictions,
    float* __restrict__ pheromone_update,
    const float temperature
);
```

### 3.4 Kernel: ve_swarm_temporal_conv

```cuda
/**
 * Dilated Temporal Convolution Stack
 *
 * Multi-scale 1D convolutions over frequency time series.
 * Captures week, month, and quarter-level patterns.
 *
 * Input:
 *   - freq_series[52 x N_variants]: Weekly frequencies
 *   - feature_series[52 x N_variants x 109]: Weekly features
 *
 * Output:
 *   - temporal_embedding[N_variants x 64]: Temporal features
 */
extern "C" __global__ void ve_swarm_temporal_conv(
    const float* __restrict__ freq_series,
    const float* __restrict__ feature_series,
    float* __restrict__ temporal_embedding,
    const float* __restrict__ conv_weights,
    const int N_variants,
    const int N_weeks
);
```

### 3.5 Kernel: ve_swarm_consensus

```cuda
/**
 * Swarm Consensus and Final Prediction
 *
 * Aggregates agent predictions using fitness-weighted voting.
 * Applies physics constraints for final prediction.
 *
 * Input:
 *   - agent_predictions[32]: Per-agent RISE/FALL probabilities
 *   - agent_fitness[32]: Historical accuracy per agent
 *   - frequency[1]: Current variant frequency
 *   - velocity[1]: Current velocity
 *
 * Output:
 *   - final_prediction[1]: RISE/FALL probability
 *   - confidence[1]: Prediction confidence
 */
extern "C" __global__ void ve_swarm_consensus(
    const float* __restrict__ agent_predictions,
    const float* __restrict__ agent_fitness,
    const float frequency,
    const float velocity,
    float* __restrict__ final_prediction,
    float* __restrict__ confidence
);
```

---

## 4. Implementation Plan

### Phase 1: Dendritic Reservoir (Week 1)

1. Create `ve_swarm_dendritic_reservoir.cu` kernel
2. Implement CSR contact graph construction in Rust
3. Write PTX compilation and integration
4. Test on 6M0J spike RBD structure
5. Verify GPU utilization > 70%

**Deliverables**:
- `crates/prism-gpu/src/kernels/ve_swarm_dendritic_reservoir.cu`
- `crates/prism-gpu/src/ve_swarm/dendritic.rs`
- `kernels/ptx/ve_swarm_dendritic_reservoir.ptx`

### Phase 2: Structural Attention (Week 2)

1. Create `ve_swarm_structural_attention.cu` kernel
2. Load ACE2 interface residues from 6M0J
3. Implement epitope class scoring
4. Integrate with FluxNet for parameter learning
5. Test attention visualization

**Deliverables**:
- `crates/prism-gpu/src/kernels/ve_swarm_structural_attention.cu`
- `crates/prism-gpu/src/ve_swarm/attention.rs`
- `data/ace2_interface_residues.json`

### Phase 3: Swarm Agents (Week 3)

1. Create `ve_swarm_agents.cu` kernel
2. Implement swarm evolution algorithm
3. Design pheromone trail mechanism
4. Test on synthetic data
5. Tune hyperparameters (population size, mutation rate)

**Deliverables**:
- `crates/prism-gpu/src/kernels/ve_swarm_agents.cu`
- `crates/prism-gpu/src/ve_swarm/agents.rs`
- `crates/prism-gpu/src/ve_swarm/pheromone.rs`

### Phase 4: Temporal Convolution (Week 4)

1. Create `ve_swarm_temporal_conv.cu` kernel
2. Load VASIL frequency time series data
3. Implement dilated convolution stack
4. Test multi-scale pattern detection
5. Integrate with swarm agents

**Deliverables**:
- `crates/prism-gpu/src/kernels/ve_swarm_temporal_conv.cu`
- `crates/prism-gpu/src/ve_swarm/temporal.rs`
- `scripts/load_vasil_timeseries.py`

### Phase 5: Integration and Benchmark (Week 5)

1. Create end-to-end pipeline
2. Run 12-country VASIL benchmark
3. Compare against 53% baseline
4. Iterate on hyperparameters
5. Document results

**Deliverables**:
- `crates/prism-gpu/src/ve_swarm/mod.rs` (full pipeline)
- `scripts/benchmark_ve_swarm.py`
- `results/ve_swarm_benchmark.json`

---

## 5. Expected Accuracy Improvements

| Component | Baseline | Expected | Mechanism |
|-----------|----------|----------|-----------|
| Dendritic Reservoir | 53% | 58% | Preserves spatial structure |
| Structural Attention | 58% | 65% | Focuses on binding interface |
| Swarm Agents | 65% | 72% | Discovers feature combinations |
| Temporal Conv | 72% | 78% | Captures trajectory patterns |
| Physics Constraints | 78% | 82% | Eliminates impossible predictions |

**Combined Target**: 75-85% accuracy

---

## 6. Prometheus Metrics

```rust
// VE-Swarm specific metrics
prism_ve_swarm_agents_active{country="Germany"} 32
prism_ve_swarm_agent_fitness_mean{country="Germany"} 0.72
prism_ve_swarm_pheromone_entropy{country="Germany"} 2.4
prism_ve_swarm_attention_entropy{country="Germany"} 1.8
prism_ve_swarm_reservoir_activation_mean{country="Germany"} 0.45
prism_ve_swarm_temporal_conv_latency_ms{country="Germany"} 12.5
prism_ve_swarm_prediction_confidence{country="Germany"} 0.85
prism_ve_swarm_accuracy_7day{country="Germany"} 0.78
```

---

## 7. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Swarm convergence too slow | Medium | High | Pre-seed with known good features |
| Overfitting to training countries | High | High | Leave-one-country-out CV |
| GPU memory overflow | Low | Medium | Batch residues, stream processing |
| Temporal conv vanishing gradients | Medium | Medium | Skip connections, layer norm |
| Attention collapse | Low | High | Entropy regularization |

---

## 8. File Structure

```
crates/prism-gpu/src/
    ve_swarm/
        mod.rs              # Module exports
        dendritic.rs        # Dendritic reservoir wrapper
        attention.rs        # Structural attention wrapper
        agents.rs           # Swarm agent management
        pheromone.rs        # Pheromone trail dynamics
        temporal.rs         # Temporal convolution wrapper
        consensus.rs        # Final prediction aggregation
        metrics.rs          # Prometheus integration

crates/prism-gpu/src/kernels/
    ve_swarm_dendritic_reservoir.cu
    ve_swarm_structural_attention.cu
    ve_swarm_agents.cu
    ve_swarm_temporal_conv.cu
    ve_swarm_consensus.cu

kernels/ptx/
    ve_swarm_dendritic_reservoir.ptx
    ve_swarm_structural_attention.ptx
    ve_swarm_agents.ptx
    ve_swarm_temporal_conv.ptx
    ve_swarm_consensus.ptx

scripts/
    benchmark_ve_swarm.py
    visualize_attention.py
    analyze_swarm_evolution.py
```

---

## 9. Conclusion

The VE-Swarm architecture addresses the fundamental limitations of current viral variant prediction:

1. **Spatial**: Dendritic reservoir preserves residue-level structure
2. **Attention**: Learned focus on binding-relevant residues
3. **Combinatorial**: Swarm agents discover non-linear feature interactions
4. **Temporal**: Multi-scale convolutions capture trajectory patterns
5. **Physical**: Hard constraints from evolutionary biology

By combining these innovations in a fully GPU-accelerated pipeline, we expect to achieve 75-85% prediction accuracy, a significant improvement over the current 53% baseline.

All components require PTX kernels that are fully implemented, integrated, and operational on GPU - no CPU fallbacks permitted.
