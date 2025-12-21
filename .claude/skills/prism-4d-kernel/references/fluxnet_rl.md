# FluxNet RL Reference

## Core Concept

FluxNet RL learns optimal feature weights for RISE/FALL prediction **without hardcoding VASIL's γ formula**.

**VASIL (hardcoded)**:
```
γ = 0.65 × escape + 0.35 × transmit
```

**FluxNet (learned)**:
```
Q-table implicitly encodes optimal weights through experience
If we beat 92%, RL discovered better feature interactions!
```

## State Space Design

### 6 Input Features → 4 Bins Each → 4096 States

| Feature | Source | Binning |
|---------|--------|---------|
| escape | DMS data | [0, 0.25, 0.5, 0.75, 1.0] |
| frequency | GISAID | [0, 0.05, 0.15, 0.5, 1.0] |
| gamma | GPU feat 95 | [-1, -0.3, 0.3, 1] |
| growth_potential | gamma × (1-freq)² | [0, 0.1, 0.3, 0.6, 1.0] |
| escape_dominance | escape - pop_mean | [-1, -0.3, 0.3, 1.0] |

### Discretization Code
```rust
pub fn discretize(&self) -> usize {
    // 4 bins per feature = 4^5 = 1024 states (simplified)
    // Or 4^6 = 4096 with all 6 features
    
    let escape_bin = ((self.escape * 3.99) as usize).min(3);
    let freq_bin = if self.frequency < 0.05 { 0 }
                   else if self.frequency < 0.15 { 1 }
                   else if self.frequency < 0.50 { 2 }
                   else { 3 };
    let gp_bin = ((self.growth_potential * 3.99) as usize).min(3);
    let ed_bin = (((self.escape_dominance + 1.0) / 2.0 * 3.99) as usize).min(3);
    
    escape_bin * 64 + freq_bin * 16 + gp_bin * 4 + ed_bin
}
```

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `alpha` | 0.15 | Learning rate (higher = faster adaptation) |
| `gamma` | 0.0 | Discount factor (0 for single-step prediction) |
| `epsilon_start` | 0.2 | Initial exploration |
| `epsilon_min` | 0.02 | Minimum exploration |
| `epsilon_decay` | 0.998 | Decay per epoch |
| `epochs` | 300 | Training iterations |
| `batch_size` | 32 | Experience replay batch |
| `num_states` | 256 | Reduced from 4096 for generalization |

## Q-Learning Update

```rust
fn train_step(&mut self, exp: &VEExperience) {
    let state_idx = exp.state.discretize();
    let action_idx = exp.action as usize;
    
    // Q-learning: Q(s,a) += α × (r - Q(s,a))
    // Note: γ=0 means no next-state value (single-step)
    self.q_table[state_idx][action_idx] += 
        self.alpha * (exp.reward - self.q_table[state_idx][action_idx]);
    
    // Update visit count for UCB
    self.visit_counts[state_idx][action_idx] += 1;
}
```

## Class Imbalance Handling

**Problem**: ~64% FALL, ~36% RISE in VASIL data

**Solution**: Asymmetric rewards

```rust
// Compute class weights
let rise_count = data.iter().filter(|(_, o)| *o == "RISE").count();
let fall_count = data.len() - rise_count;
let rise_weight = data.len() as f32 / (2.0 * rise_count as f32);  // ~1.4
let fall_weight = data.len() as f32 / (2.0 * fall_count as f32);  // ~0.8

// Apply asymmetric rewards
let reward = if is_correct {
    if is_rise { rise_weight } else { fall_weight }
} else {
    // Extra penalty for missing RISE (minority class)
    if is_rise { -rise_weight * 1.5 } else { -fall_weight }
};
```

## Q-Value Prior Initialization

```rust
// Initialize with bias toward majority class (FALL)
let fall_base_rate = 0.64;
let rise_prior = fall_base_rate - 0.5;  // -0.14
let fall_prior = 0.5 - fall_base_rate;  // -0.14

// Q(RISE) initialized slightly negative
// Q(FALL) initialized slightly positive
let q_init = [rise_prior, -fall_prior];  // [-0.14, 0.14]
self.q_table = vec![q_init; num_states];
```

## Action Selection

### Epsilon-Greedy with UCB Bonus
```rust
fn select_action(&self, state: &VEState, explore: bool) -> VEAction {
    let state_idx = state.discretize();
    let q_values = self.q_table[state_idx];
    let visits = self.visit_counts[state_idx];
    
    if explore && rand::random::<f32>() < self.epsilon {
        // Random exploration
        if rand::random::<bool>() { VEAction::Rise } else { VEAction::Fall }
    } else {
        // Exploitation with UCB bonus for uncertainty
        let total_visits = visits[0] + visits[1] + 1;
        
        let ucb_rise = q_values[0] + 
            (2.0 * (total_visits as f32).ln() / (visits[0] + 1) as f32).sqrt();
        let ucb_fall = q_values[1] + 
            (2.0 * (total_visits as f32).ln() / (visits[1] + 1) as f32).sqrt();
        
        if ucb_rise > ucb_fall { VEAction::Rise } else { VEAction::Fall }
    }
}
```

## Training Loop

```rust
pub fn train_on_dataset(&mut self, data: &[(VEState, &str)], epochs: usize) {
    log::info!("Training on {} samples for {} epochs", data.len(), epochs);
    
    for epoch in 0..epochs {
        let mut correct = 0;
        let mut rise_correct = 0;
        let mut rise_total = 0;
        
        // Shuffle data each epoch
        let mut shuffled: Vec<_> = data.iter().collect();
        shuffled.shuffle(&mut rand::thread_rng());
        
        for (state, observed) in shuffled {
            let action = self.select_action(state, true);
            let is_rise = *observed == "RISE";
            let is_correct = action.to_str() == *observed;
            
            // Compute asymmetric reward
            let reward = compute_reward(is_correct, is_rise);
            
            // Create and process experience
            let exp = VEExperience {
                state: state.clone(),
                action,
                reward,
                next_state: state.clone(),
            };
            
            self.train_step(&exp);
            self.add_experience(exp);
            
            // Batch replay every 100 samples
            if self.training_samples % 100 == 0 {
                self.train_batch(32);
            }
            
            // Track metrics
            if is_correct { correct += 1; }
            if is_rise {
                rise_total += 1;
                if is_correct { rise_correct += 1; }
            }
        }
        
        // Decay exploration
        self.decay_epsilon();
        
        // Log progress
        if epoch % 10 == 0 {
            let accuracy = correct as f32 / data.len() as f32;
            let rise_recall = rise_correct as f32 / rise_total as f32;
            log::info!("Epoch {}: acc={:.3}, rise_recall={:.3}, ε={:.3}",
                       epoch, accuracy, rise_recall, self.epsilon);
        }
    }
}
```

## Experience Replay

```rust
fn add_experience(&mut self, exp: VEExperience) {
    if self.replay_buffer.len() >= 10000 {
        self.replay_buffer.remove(0);  // FIFO
    }
    self.replay_buffer.push(exp);
}

fn train_batch(&mut self, batch_size: usize) {
    if self.replay_buffer.len() < batch_size {
        return;
    }
    
    // Sample random batch
    let indices: Vec<usize> = (0..batch_size)
        .map(|_| rand::random::<usize>() % self.replay_buffer.len())
        .collect();
    
    for i in indices {
        self.train_step(&self.replay_buffer[i].clone());
    }
}
```

## Evaluation

```rust
pub fn evaluate(&self, data: &[(VEState, &str)]) -> f32 {
    let mut correct = 0;
    
    for (state, observed) in data {
        let action = self.predict(state);  // No exploration
        if action.to_str() == *observed {
            correct += 1;
        }
    }
    
    correct as f32 / data.len() as f32
}
```

## Debugging Q-Table

```rust
fn analyze_q_table(&self) {
    println!("=== Q-TABLE ANALYSIS ===");
    
    // Find states with strongest preferences
    let mut max_diff = 0.0f32;
    let mut max_state = 0;
    
    for (state_idx, q) in self.q_table.iter().enumerate() {
        let diff = (q[0] - q[1]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_state = state_idx;
        }
    }
    
    println!("Strongest preference: state {} (diff={:.3})", max_state, max_diff);
    println!("  Q(RISE)={:.3}, Q(FALL)={:.3}", 
             self.q_table[max_state][0], 
             self.q_table[max_state][1]);
    
    // Decode state back to features
    let escape_bin = max_state / 64;
    let freq_bin = (max_state % 64) / 16;
    let gp_bin = (max_state % 16) / 4;
    let ed_bin = max_state % 4;
    
    println!("  Decoded: escape_bin={}, freq_bin={}, gp_bin={}, ed_bin={}",
             escape_bin, freq_bin, gp_bin, ed_bin);
    
    // Count states preferring each action
    let rise_pref = self.q_table.iter().filter(|q| q[0] > q[1]).count();
    let fall_pref = self.q_table.len() - rise_pref;
    
    println!("States preferring RISE: {} ({:.1}%)", 
             rise_pref, rise_pref as f32 / self.q_table.len() as f32 * 100.0);
    println!("States preferring FALL: {} ({:.1}%)", 
             fall_pref, fall_pref as f32 / self.q_table.len() as f32 * 100.0);
}
```

## Expected Q-Value Patterns

After successful training, Q-table should show:

| State Profile | Expected Preference |
|---------------|---------------------|
| High escape + Low freq | RISE (emerging variant) |
| High escape + High freq | FALL (peaked, declining) |
| Low escape + Low freq | FALL (no selective advantage) |
| Low escape + High freq | FALL (incumbent, stable) |
| High growth_potential | RISE |
| Positive escape_dominance | RISE |

## Failure Modes

1. **All FALL predictions**: Class imbalance not handled
   - Fix: Increase rise_weight, add asymmetric penalty

2. **50% accuracy**: Features not discriminative
   - Fix: Check feature extraction, ensure non-zero values

3. **Training accuracy high, test low**: Overfitting
   - Fix: Use coarser binning (256 states vs 4096)

4. **Random predictions**: Q-values all near zero
   - Fix: Increase alpha, add more training epochs
