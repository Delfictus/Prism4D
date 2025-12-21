# PRISM-VE Cycle Module: Complete Implementation Blueprint

**Date:** 2025-12-08
**Target:** GPU-accelerated temporal emergence prediction
**Goal:** Predict WHEN mutations emerge (not just WHICH)

---

## EXECUTIVE SUMMARY

**What:** Evolutionary Cycle Detection and Temporal Emergence Prediction
**Why:** Novel capability beyond EVEscape and VASIL (answers "WHEN" not just "WHAT")
**How:** 6-phase classification + velocity-based timing prediction
**Speed:** GPU-accelerated, maintains 323 mutations/second throughput
**Validation:** Against VASIL's temporal accuracy benchmarks

---

## 1. CYCLE MODULE OVERVIEW

### 1.1 Core Concept: Viral Escape Cycles

**Key Insight:**
Viral escape is NOT linear - it's CYCLIC:

```
Phase 1: NAIVE
  ↓ (Antibody pressure applied)
Phase 2: EXPLORING (rising frequency)
  ↓ (Reaches dominance)
Phase 3: ESCAPED (high frequency, dominant)
  ↓ (Fitness cost accumulates)
Phase 4: COSTLY (still frequent but declining)
  ↓ (Reversion pressure wins)
Phase 5: REVERTING (falling frequency)
  ↓ (Returns to baseline)
Phase 6: FIXED (rare: stable with compensation)
  ↓ (Or back to NAIVE - cycle repeats)
```

**Example - Position 484 (E484K):**
```
2021-01: NAIVE      (0.1% frequency, not under selection)
2021-03: EXPLORING  (2% frequency, rising 1%/month) ← Beta variant
2021-06: ESCAPED    (60% frequency, dominant)
2021-09: COSTLY     (55% frequency, falling)
2021-12: REVERTING  (20% frequency, rapid decline)
2022-03: NAIVE      (0.5% frequency, cycle complete)
2022-06: EXPLORING  (3% frequency, rising again) ← Omicron E484A
```

### 1.2 What Cycle Module Predicts

**Primary Output:**
```rust
pub struct CycleState {
    position: i32,              // Spike position (331-531)
    current_phase: Phase,       // NAIVE/EXPLORING/ESCAPED/etc
    phase_confidence: f32,      // 0-1
    current_frequency: f32,     // Current mutation prevalence
    velocity: f32,              // Δfreq/month (positive = rising)
    time_in_phase_days: f32,    // How long in current phase
    predicted_emergence: String, // "1-3 months", "6-12 months", etc
    emergence_probability: f32,  // P(emerges in time window)
}
```

**Novel Capability:**
```
User: "When will E484K emerge?"

PRISM-VE: "E484K is in EXPLORING phase
           Current frequency: 5%
           Rising at 2%/month
           Predicted dominance: 1-3 months
           Emergence probability: 0.85"

EVEscape: [Cannot answer - no temporal prediction]
VASIL: [Can answer but uses different method]
```

---

## 2. ARCHITECTURE DESIGN

### 2.1 GPU Kernel Architecture

**File:** `crates/prism-gpu/src/kernels/viral_evolution_cycle.cu`

**Kernel Stages:**

```cuda
//=============================================================================
// STAGE 1: Frequency Trajectory Analysis
//=============================================================================

__global__ void stage1_frequency_analysis(
    const float* __restrict__ gisaid_frequencies,  // [n_positions × n_timepoints]
    const int n_positions,
    const int n_timepoints,
    float* __restrict__ current_freq_out,          // [n_positions]
    float* __restrict__ velocity_out,              // [n_positions] Δfreq/month
    float* __restrict__ acceleration_out           // [n_positions] Δvelocity
)

Purpose: Compute frequency dynamics from time-series
Output: Current frequency, velocity, acceleration per position
```

```cuda
//=============================================================================
// STAGE 2: Phase Classification
//=============================================================================

__global__ void stage2_phase_classification(
    const float* __restrict__ current_freq,        // [n_positions]
    const float* __restrict__ velocity,            // [n_positions]
    const float* __restrict__ acceleration,        // [n_positions]
    const float* __restrict__ fitness_gamma,       // [n_positions] from Fitness module
    int* __restrict__ phase_out,                   // [n_positions] 0-5
    float* __restrict__ phase_confidence_out       // [n_positions]
)

Purpose: Classify each position into 6 phases
Logic:
  if (freq < 0.01 && vel < 0.05): NAIVE
  if (freq < 0.50 && vel > 0.05): EXPLORING
  if (freq > 0.50 && vel >= 0): ESCAPED
  if (freq > 0.20 && vel < -0.02): REVERTING
  etc.
```

```cuda
//=============================================================================
// STAGE 3: Temporal Emergence Prediction
//=============================================================================

__global__ void stage3_emergence_prediction(
    const int* __restrict__ phase,                 // [n_positions]
    const float* __restrict__ velocity,
    const float* __restrict__ escape_prob,         // From Escape module
    const float* __restrict__ fitness_gamma,       // From Fitness module
    const float time_horizon_months,               // 3, 6, or 12 months
    float* __restrict__ emergence_prob_out,        // [n_positions]
    float* __restrict__ predicted_timing_out       // [n_positions] months
)

Purpose: Predict emergence probability and timing
Output: P(emerges in time window), predicted months to dominance
```

### 2.2 Shared Memory Structure

```cuda
struct __align__(16) CycleSharedMemory {
    // Input trajectories (per-position time-series)
    float frequencies[TILE_SIZE][MAX_TIMEPOINTS];  // Historical frequencies

    // Computed dynamics
    float current_freq[TILE_SIZE];
    float velocity[TILE_SIZE];          // 1st derivative
    float acceleration[TILE_SIZE];      // 2nd derivative

    // Phase detection
    int phase[TILE_SIZE];               // 0-5 (6 phases)
    float phase_confidence[TILE_SIZE];

    // Temporal predictions
    float emergence_prob[TILE_SIZE];
    float time_to_peak[TILE_SIZE];

    // Integration with other modules
    float escape_scores[TILE_SIZE];     // From Escape module
    float fitness_gamma[TILE_SIZE];     // From Fitness module
};
```

### 2.3 Integration with mega_fused

**Option A: Separate Kernel (Easier)**
```
mega_fused → 92-dim features → Escape prediction
     ↓
cycle_kernel → phase detection → Emergence timing

Two GPU calls, 250-300 mutations/second (still fast!)
```

**Option B: Integrated Kernel (Optimal)**
```
mega_fused_extended → Stages 1-8
  ├─ Stages 1-6.5: Existing (92-dim features)
  ├─ Stage 7: Fitness (4-dim)
  └─ Stage 8: Cycle (5-dim)

Single GPU call, 323 mutations/second (maximum speed!)
Output: 101-dim vector per position
```

**Recommendation:** Start with Option A (separate), merge to Option B in v2.0

---

## 3. DATA REQUIREMENTS

### 3.1 GISAID Frequency Data

**Source:** Raw GISAID sequences (NOT VASIL's fitted frequencies)

**Format Required:**
```csv
position,date,frequency,mutation,variant
484,2021-01-01,0.001,E484K,Beta
484,2021-02-01,0.003,E484K,Beta
484,2021-03-01,0.008,E484K,Beta
484,2021-04-01,0.015,E484K,Beta
...
```

**Processing from Raw GISAID:**
```python
# scripts/process_gisaid_frequencies.py

import pandas as pd
from collections import Counter

def process_gisaid_to_position_frequencies(gisaid_metadata_path):
    """
    Convert raw GISAID sequences to position-level frequencies.

    Args:
        gisaid_metadata_path: Path to GISAID metadata.tsv

    Returns:
        DataFrame with [position, date, frequency, mutation]
    """

    # Load GISAID metadata
    gisaid = pd.read_csv(gisaid_metadata_path, sep='\t')

    # Filter to Spike sequences
    spike_only = gisaid[gisaid['gene'] == 'S']

    # Group by date and position
    position_freqs = []

    for date in spike_only['date'].unique():
        date_seqs = spike_only[spike_only['date'] == date]
        total_seqs = len(date_seqs)

        # Count mutations per position
        for pos in range(331, 532):  # RBD positions
            mutations_at_pos = date_seqs[f'site_{pos}'].value_counts()

            for mutation, count in mutations_at_pos.items():
                frequency = count / total_seqs

                if frequency > 0.001:  # >0.1% threshold
                    position_freqs.append({
                        'position': pos,
                        'date': date,
                        'frequency': frequency,
                        'mutation': mutation
                    })

    df = pd.DataFrame(position_freqs)
    return df
```

**Output:** `data/processed/gisaid_position_frequencies.csv`

**Size:** ~100 MB for 2021-2024 data

### 3.2 Known Variant Emergence Dates (Ground Truth)

**For Validation:**
```csv
variant,key_mutation,emergence_date,peak_date,peak_frequency
Alpha,N501Y,2020-12-01,2021-04-01,0.65
Beta,E484K,2021-01-01,2021-06-01,0.45
Delta,L452R,2021-04-01,2021-09-01,0.75
Omicron BA.1,E484A,2021-11-01,2022-02-01,0.85
Omicron BA.2,S371F,2022-01-01,2022-05-01,0.70
BQ.1.1,K444T,2022-09-01,2022-12-01,0.35
XBB.1.5,F486P,2022-11-01,2023-03-01,0.45
```

**Use:** Validate temporal predictions retrospectively

---

## 4. PHASE DETECTION ALGORITHM

### 4.1 Decision Tree

```python
def classify_phase(
    current_freq: float,
    velocity: float,          # Δfreq/month
    acceleration: float,      # Δvelocity/month
    fitness_gamma: float,     # From Fitness module
    escape_prob: float,       # From Escape module
) -> tuple[Phase, float]:
    """
    Classify evolutionary phase with confidence.

    Returns:
        (Phase enum, confidence 0-1)
    """

    # Phase 1: NAIVE (never under selection)
    if current_freq < 0.01 and velocity < 0.01 and escape_prob < 0.5:
        confidence = 1.0 - current_freq  # More confident if truly absent
        return (Phase.NAIVE, confidence)

    # Phase 2: EXPLORING (actively rising)
    if velocity > 0.05 and current_freq < 0.50:
        confidence = min(1.0, velocity * 10)  # Higher velocity = more confident
        return (Phase.EXPLORING, confidence)

    # Phase 3: ESCAPED (dominant, stable or rising)
    if current_freq > 0.50 and velocity >= -0.02:
        confidence = current_freq  # More dominant = more confident
        return (Phase.ESCAPED, confidence)

    # Phase 4: COSTLY (high frequency but falling)
    if current_freq > 0.20 and velocity < -0.02 and fitness_gamma < 0:
        confidence = min(1.0, abs(velocity) * 10)
        return (Phase.COSTLY, confidence)

    # Phase 5: REVERTING (actively falling)
    if velocity < -0.05:
        confidence = min(1.0, abs(velocity) * 10)
        return (Phase.REVERTING, confidence)

    # Phase 6: FIXED (stable at high frequency, no fitness cost)
    if current_freq > 0.80 and abs(velocity) < 0.02 and fitness_gamma > -0.1:
        confidence = 1.0 - abs(velocity)
        return (Phase.FIXED, confidence)

    # Default: EXPLORING (uncertain)
    return (Phase.EXPLORING, 0.4)
```

### 4.2 GPU Implementation

```cuda
__device__ int classify_phase_gpu(
    float freq,
    float vel,
    float accel,
    float fitness,
    float* confidence_out
) {
    // Phase 1: NAIVE
    if (freq < 0.01f && vel < 0.01f) {
        *confidence_out = 1.0f - freq;
        return 0;  // NAIVE
    }

    // Phase 2: EXPLORING
    if (vel > 0.05f && freq < 0.50f) {
        *confidence_out = fminf(1.0f, vel * 10.0f);
        return 1;  // EXPLORING
    }

    // Phase 3: ESCAPED
    if (freq > 0.50f && vel >= -0.02f) {
        *confidence_out = freq;
        return 2;  // ESCAPED
    }

    // Phase 4: COSTLY
    if (freq > 0.20f && vel < -0.02f && fitness < 0.0f) {
        *confidence_out = fminf(1.0f, fabsf(vel) * 10.0f);
        return 3;  // COSTLY
    }

    // Phase 5: REVERTING
    if (vel < -0.05f) {
        *confidence_out = fminf(1.0f, fabsf(vel) * 10.0f);
        return 4;  // REVERTING
    }

    // Phase 6: FIXED
    if (freq > 0.80f && fabsf(vel) < 0.02f && fitness > -0.1f) {
        *confidence_out = 1.0f - fabsf(vel);
        return 5;  // FIXED
    }

    // Default: EXPLORING (low confidence)
    *confidence_out = 0.4f;
    return 1;
}
```

---

## 5. TEMPORAL PREDICTION ALGORITHM

### 5.1 Emergence Timing

**Core Formula:**

```
time_to_emergence = f(phase, velocity, escape_prob, fitness_gamma)

Where:
- EXPLORING phase + high velocity → 1-3 months
- NAIVE phase + high escape → 3-6 months
- ESCAPED phase → already happened (0 months)
- REVERTING phase → won't emerge (>12 months or never)
```

**Implementation:**

```python
def predict_emergence_timing(
    phase: Phase,
    velocity: float,
    escape_prob: float,
    fitness_gamma: float,
    time_horizon: str  # "3_months", "6_months", "12_months"
) -> tuple[str, float]:
    """
    Predict WHEN a mutation will emerge.

    Returns:
        (timing_category, emergence_probability)
    """

    # Cycle multiplier based on phase
    phase_multipliers = {
        Phase.NAIVE: 0.3,      # Can emerge but slow
        Phase.EXPLORING: 1.0,  # Actively emerging NOW
        Phase.ESCAPED: 0.1,    # Already happened
        Phase.COSTLY: 0.4,     # Might shift to different mutation
        Phase.REVERTING: 0.2,  # Unlikely to re-emerge soon
        Phase.FIXED: 0.05,     # Stable, won't change
    }

    cycle_mult = phase_multipliers.get(phase, 0.5)

    # Base emergence probability
    # Combines: escape (can it escape?) × fitness (will it survive?) × cycle (is it ready?)
    emergence_prob = escape_prob * (fitness_gamma + 1.0) / 2.0 * cycle_mult

    # Timing prediction
    if phase == Phase.EXPLORING:
        # Use velocity to predict when it reaches 50% (dominance)
        if velocity > 0.001:
            months_to_dominance = (0.50 - current_freq) / velocity

            if months_to_dominance < 3:
                timing = "1-3 months"
            elif months_to_dominance < 6:
                timing = "3-6 months"
            else:
                timing = "6-12 months"
        else:
            timing = "6-12 months"

    elif phase == Phase.NAIVE:
        if escape_prob > 0.8 and fitness_gamma > 0:
            timing = "3-6 months"  # High potential but not yet selected
        else:
            timing = ">12 months"

    elif phase == Phase.ESCAPED:
        timing = "Already emerged"
        emergence_prob = 0.95  # It's here NOW

    else:  # COSTLY, REVERTING, FIXED
        timing = ">12 months or unlikely"
        emergence_prob *= 0.5  # Penalize

    return (timing, emergence_prob)
```

### 5.2 GPU Implementation

```cuda
__device__ void predict_emergence_gpu(
    int phase,
    float velocity,
    float current_freq,
    float escape_prob,
    float fitness,
    float time_horizon_months,
    float* emergence_prob_out,
    float* months_to_peak_out
) {
    // Cycle multiplier
    float cycle_mult = 1.0f;
    if (phase == 0) cycle_mult = 0.3f;  // NAIVE
    if (phase == 1) cycle_mult = 1.0f;  // EXPLORING
    if (phase == 2) cycle_mult = 0.1f;  // ESCAPED
    if (phase == 4) cycle_mult = 0.2f;  // REVERTING

    // Emergence probability
    float fitness_normalized = (fitness + 1.0f) / 2.0f;  // Map -1,1 → 0,1
    *emergence_prob_out = escape_prob * fitness_normalized * cycle_mult;

    // Months to 50% dominance
    if (velocity > 0.001f && phase == 1) {  // EXPLORING
        *months_to_peak_out = (0.50f - current_freq) / velocity;
    } else if (phase == 2) {  // ESCAPED
        *months_to_peak_out = 0.0f;  // Already dominant
    } else {
        *months_to_peak_out = 999.0f;  // Unknown/unlikely
    }
}
```

---

## 6. DATA PROCESSING PIPELINE

### 6.1 GISAID Frequency Processing

**Script:** `scripts/process_gisaid_frequencies.py`

```python
#!/usr/bin/env python3
"""
Process raw GISAID sequences to position-level frequencies.

CRITICAL: Use RAW GISAID, not VASIL's processed frequencies!
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

def load_raw_gisaid(gisaid_metadata_path: Path) -> pd.DataFrame:
    """
    Load raw GISAID metadata.

    Download from: https://gisaid.org/
    OR: Use VASIL's if verified to be raw sequences
    """

    # Load full metadata
    gisaid = pd.read_csv(gisaid_metadata_path, sep='\t', low_memory=False)

    # Essential columns
    required = ['accession_id', 'date', 'country', 'pango_lineage', 'aa_substitutions']

    for col in required:
        if col not in gisaid.columns:
            raise ValueError(f"Missing required column: {col}")

    return gisaid

def extract_spike_mutations(aa_substitutions: str) -> list[tuple[int, str]]:
    """
    Parse AA substitutions to extract Spike mutations.

    Format: "S:E484K,S:N501Y,ORF1a:P3395H"
    Returns: [(484, 'E484K'), (501, 'N501Y')]
    """
    if pd.isna(aa_substitutions):
        return []

    mutations = []
    for sub in aa_substitutions.split(','):
        if sub.startswith('S:'):  # Spike protein
            mut = sub[2:]  # Remove 'S:'

            # Parse E484K → position 484
            if len(mut) >= 3:
                try:
                    pos = int(mut[1:-1])
                    if 331 <= pos <= 531:  # RBD range
                        mutations.append((pos, mut))
                except ValueError:
                    continue

    return mutations

def compute_position_frequencies(
    gisaid: pd.DataFrame,
    start_date: str = "2021-07-01",
    end_date: str = "2024-12-01",
    time_resolution: str = "weekly"  # or "daily", "monthly"
) -> pd.DataFrame:
    """
    Compute mutation frequency per position over time.

    This is OUR processing (not VASIL's).
    """

    # Filter date range
    gisaid['date'] = pd.to_datetime(gisaid['date'])
    mask = (gisaid['date'] >= start_date) & (gisaid['date'] <= end_date)
    gisaid_filtered = gisaid[mask].copy()

    # Resample to time resolution
    if time_resolution == "weekly":
        gisaid_filtered['time_bin'] = gisaid_filtered['date'].dt.to_period('W')
    elif time_resolution == "monthly":
        gisaid_filtered['time_bin'] = gisaid_filtered['date'].dt.to_period('M')
    else:
        gisaid_filtered['time_bin'] = gisaid_filtered['date']

    # Compute frequencies
    results = []

    for time_bin in gisaid_filtered['time_bin'].unique():
        bin_data = gisaid_filtered[gisaid_filtered['time_bin'] == time_bin]
        total_sequences = len(bin_data)

        if total_sequences < 100:  # Skip sparse bins
            continue

        # Count mutations per position
        position_counts = {}

        for _, row in bin_data.iterrows():
            mutations = extract_spike_mutations(row['aa_substitutions'])
            for pos, mut in mutations:
                key = (pos, mut)
                position_counts[key] = position_counts.get(key, 0) + 1

        # Compute frequencies
        for (pos, mut), count in position_counts.items():
            frequency = count / total_sequences

            results.append({
                'position': pos,
                'date': str(time_bin),
                'frequency': frequency,
                'mutation': mut,
                'n_sequences': total_sequences,
                'n_with_mutation': count
            })

    df_freq = pd.DataFrame(results)

    # Sort by position and date
    df_freq = df_freq.sort_values(['position', 'date'])

    return df_freq

def compute_velocity(freq_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute velocity (Δfreq/month) from frequency time-series.
    """

    freq_df = freq_df.copy()
    freq_df['date'] = pd.to_datetime(freq_df['date'])

    # Group by position
    results = []

    for position in freq_df['position'].unique():
        pos_data = freq_df[freq_df['position'] == position].sort_values('date')

        for i in range(len(pos_data)):
            if i == 0:
                velocity = 0.0  # No previous data
            else:
                prev_freq = pos_data.iloc[i-1]['frequency']
                curr_freq = pos_data.iloc[i]['frequency']

                prev_date = pos_data.iloc[i-1]['date']
                curr_date = pos_data.iloc[i]['date']

                days = (curr_date - prev_date).days
                if days > 0:
                    months = days / 30.0
                    velocity = (curr_freq - prev_freq) / months
                else:
                    velocity = 0.0

            results.append({
                **pos_data.iloc[i].to_dict(),
                'velocity': velocity
            })

    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gisaid", required=True, help="Path to GISAID metadata.tsv")
    parser.add_argument("--output", default="data/processed/position_frequencies.csv")
    args = parser.parse_args()

    print("Processing GISAID to position-level frequencies...")
    print("CRITICAL: Using RAW GISAID (not VASIL's processed)")
    print()

    # Load raw GISAID
    gisaid = load_raw_gisaid(Path(args.gisaid))
    print(f"Loaded {len(gisaid)} sequences")

    # Compute frequencies
    freq_df = compute_position_frequencies(gisaid)
    print(f"Computed frequencies: {len(freq_df)} position-date combinations")

    # Compute velocities
    freq_with_vel = compute_velocity(freq_df)

    # Save
    freq_with_vel.to_csv(args.output, index=False)
    print(f"✅ Saved to {args.output}")
    print()
    print("This is OUR processing (independent from VASIL)")
```

---

## 7. RUST IMPLEMENTATION

### 7.1 Cycle Module Structure

**File:** `crates/prism-ve/src/cycle.rs`

```rust
use std::collections::HashMap;
use std::path::Path;
use chrono::NaiveDate;
use csv::ReaderBuilder;
use prism_core::PrismError;

/// Evolutionary cycle phase
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Naive = 0,
    Exploring = 1,
    Escaped = 2,
    Costly = 3,
    Reverting = 4,
    Fixed = 5,
}

impl Phase {
    pub fn from_i32(val: i32) -> Self {
        match val {
            0 => Phase::Naive,
            1 => Phase::Exploring,
            2 => Phase::Escaped,
            3 => Phase::Costly,
            4 => Phase::Reverting,
            5 => Phase::Fixed,
            _ => Phase::Exploring,  // Default
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Phase::Naive => "Never under immune selection",
            Phase::Exploring => "Currently rising under selection",
            Phase::Escaped => "Dominant escape variant",
            Phase::Costly => "Fitness cost accumulating",
            Phase::Reverting => "Returning to wild-type",
            Phase::Fixed => "Stable compensated escape",
        }
    }

    pub fn emergence_ready(&self) -> bool {
        matches!(self, Phase::Naive | Phase::Exploring | Phase::Reverting)
    }
}

/// Current cycle state for a position
#[derive(Debug, Clone)]
pub struct CycleState {
    pub position: i32,
    pub phase: Phase,
    pub phase_confidence: f32,      // 0-1
    pub current_frequency: f32,     // Current mutation prevalence
    pub velocity: f32,              // Δfreq/month
    pub acceleration: f32,          // Δvelocity/month
    pub time_in_phase_days: f32,    // Estimated days in current phase
    pub emergence_probability: f32, // P(emerges in time window)
    pub predicted_timing: String,   // "1-3 months", etc
}

/// GISAID frequency trajectory data
pub struct FrequencyTrajectory {
    position_data: HashMap<i32, PositionTimeSeries>,
}

struct PositionTimeSeries {
    dates: Vec<NaiveDate>,
    frequencies: Vec<f32>,
    velocities: Vec<f32>,
}

impl FrequencyTrajectory {
    /// Load from processed GISAID data (OUR processing, not VASIL's!)
    pub fn load_from_processed(
        freq_file: &Path
    ) -> Result<Self, PrismError> {
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(freq_file)
            .map_err(|e| PrismError::io("cycle", format!("Load freq: {}", e)))?;

        let mut position_data: HashMap<i32, Vec<(NaiveDate, f32, f32)>> = HashMap::new();

        for result in reader.records() {
            let record = result
                .map_err(|e| PrismError::io("cycle", format!("Read: {}", e)))?;

            let position: i32 = record.get(0)
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| PrismError::data("cycle", "Invalid position"))?;

            let date_str = record.get(1)
                .ok_or_else(|| PrismError::data("cycle", "Missing date"))?;
            let date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
                .map_err(|e| PrismError::data("cycle", format!("Invalid date: {}", e)))?;

            let frequency: f32 = record.get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);

            let velocity: f32 = record.get(6)  // Assuming column 6 has velocity
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);

            position_data.entry(position)
                .or_insert_with(Vec::new)
                .push((date, frequency, velocity));
        }

        // Convert to PositionTimeSeries
        let position_ts: HashMap<i32, PositionTimeSeries> = position_data
            .into_iter()
            .map(|(pos, mut data)| {
                data.sort_by_key(|(date, _, _)| *date);

                let dates: Vec<_> = data.iter().map(|(d, _, _)| *d).collect();
                let frequencies: Vec<_> = data.iter().map(|(_, f, _)| *f).collect();
                let velocities: Vec<_> = data.iter().map(|(_, _, v)| *v).collect();

                (pos, PositionTimeSeries { dates, frequencies, velocities })
            })
            .collect();

        Ok(Self {
            position_data: position_ts
        })
    }

    /// Get current state for a position
    pub fn get_current_state(
        &self,
        position: i32,
        current_date: &NaiveDate
    ) -> Option<(f32, f32, f32)> {
        let ts = self.position_data.get(&position)?;

        // Find most recent date before or equal to current_date
        let idx = ts.dates.iter()
            .rposition(|d| d <= current_date)?;

        let freq = ts.frequencies[idx];
        let vel = ts.velocities[idx];

        // Acceleration (2nd derivative)
        let accel = if idx > 0 {
            ts.velocities[idx] - ts.velocities[idx - 1]
        } else {
            0.0
        };

        Some((freq, vel, accel))
    }
}

/// Cycle detector
pub struct CycleDetector {
    trajectories: FrequencyTrajectory,
    current_date: NaiveDate,
}

impl CycleDetector {
    pub fn new(
        freq_file: &Path,
        current_date: NaiveDate
    ) -> Result<Self, PrismError> {
        let trajectories = FrequencyTrajectory::load_from_processed(freq_file)?;

        Ok(Self {
            trajectories,
            current_date
        })
    }

    pub fn detect_phase(
        &self,
        position: i32,
        escape_prob: f32,
        fitness_gamma: f32
    ) -> CycleState {
        // Get current dynamics
        let (freq, vel, accel) = self.trajectories
            .get_current_state(position, &self.current_date)
            .unwrap_or((0.0, 0.0, 0.0));

        // Classify phase
        let (phase, confidence) = self.classify_phase_internal(
            freq, vel, accel, fitness_gamma, escape_prob
        );

        // Predict emergence
        let (timing, emergence_prob) = self.predict_emergence_internal(
            phase, vel, freq, escape_prob, fitness_gamma
        );

        CycleState {
            position,
            phase,
            phase_confidence: confidence,
            current_frequency: freq,
            velocity: vel,
            acceleration: accel,
            time_in_phase_days: 0.0,  // TODO: Compute from trajectory
            emergence_probability: emergence_prob,
            predicted_timing: timing,
        }
    }

    fn classify_phase_internal(
        &self,
        freq: f32,
        vel: f32,
        accel: f32,
        fitness: f32,
        escape_prob: f32
    ) -> (Phase, f32) {
        // Implement decision tree from section 4.1

        if freq < 0.01 && vel < 0.01 {
            return (Phase::Naive, 1.0 - freq);
        }

        if vel > 0.05 && freq < 0.50 {
            return (Phase::Exploring, (vel * 10.0).min(1.0));
        }

        if freq > 0.50 && vel >= -0.02 {
            return (Phase::Escaped, freq);
        }

        if vel < -0.05 {
            return (Phase::Reverting, (vel.abs() * 10.0).min(1.0));
        }

        (Phase::Exploring, 0.4)
    }

    fn predict_emergence_internal(
        &self,
        phase: Phase,
        velocity: f32,
        current_freq: f32,
        escape_prob: f32,
        fitness: f32
    ) -> (String, f32) {
        // Implement timing prediction from section 5.1

        let cycle_mult = match phase {
            Phase::Naive => 0.3,
            Phase::Exploring => 1.0,
            Phase::Escaped => 0.1,
            Phase::Reverting => 0.2,
            _ => 0.5,
        };

        let fitness_norm = (fitness + 1.0) / 2.0;
        let emergence_prob = escape_prob * fitness_norm * cycle_mult;

        let timing = if phase == Phase::Exploring && velocity > 0.001 {
            let months = (0.50 - current_freq) / velocity;
            if months < 3.0 {
                "1-3 months".to_string()
            } else if months < 6.0 {
                "3-6 months".to_string()
            } else {
                "6-12 months".to_string()
            }
        } else if phase == Phase::Escaped {
            "Already emerged".to_string()
        } else {
            ">12 months or unlikely".to_string()
        };

        (timing, emergence_prob)
    }
}
```

---

## 8. INTEGRATION WITH ESCAPE + FITNESS

### 8.1 Unified PRISM-VE Predictor

**File:** `crates/prism-ve/src/lib.rs`

```rust
use prism_gpu::mega_fused::MegaFusedGpu;
use prism_gpu::viral_evolution_fitness::ViralEvolutionFitnessGpu;

pub mod cycle;
pub mod data;

use cycle::{CycleDetector, CycleState};

/// Unified PRISM-VE predictor
pub struct PRISMVEPredictor {
    // Escape module (existing mega_fused)
    escape_gpu: MegaFusedGpu,

    // Fitness module (new VE fitness GPU)
    fitness_gpu: ViralEvolutionFitnessGpu,

    // Cycle module (CPU/GPU hybrid)
    cycle_detector: CycleDetector,
}

impl PRISMVEPredictor {
    pub fn predict_emergence(
        &mut self,
        mutations: &[String],  // ["E484K", "N501Y", ...]
        time_horizon: &str,    // "3_months", "6_months", "12_months"
    ) -> Result<Vec<EmergencePrediction>, PrismError> {

        let mut predictions = Vec::new();

        for mutation in mutations {
            // Parse mutation
            let (wt_aa, pos, mut_aa) = parse_mutation(mutation)?;

            // Step 1: Get escape probability (GPU)
            let escape_prob = self.escape_gpu.predict_single_mutation(mutation)?;

            // Step 2: Get fitness score (GPU)
            let fitness_gamma = self.fitness_gpu.compute_single_mutation_fitness(mutation)?;

            // Step 3: Get cycle state (CPU with GISAID data)
            let cycle_state = self.cycle_detector.detect_phase(
                pos,
                escape_prob,
                fitness_gamma
            );

            // Step 4: Combine into emergence prediction
            predictions.push(EmergencePrediction {
                mutation: mutation.clone(),
                position: pos,
                escape_probability: escape_prob,
                fitness_score: fitness_gamma,
                cycle_phase: cycle_state.phase,
                cycle_confidence: cycle_state.phase_confidence,
                emergence_probability: cycle_state.emergence_probability,
                predicted_timing: cycle_state.predicted_timing,
                current_frequency: cycle_state.current_frequency,
                velocity: cycle_state.velocity,
            });
        }

        // Sort by emergence probability
        predictions.sort_by(|a, b|
            b.emergence_probability.partial_cmp(&a.emergence_probability).unwrap()
        );

        Ok(predictions)
    }
}

#[derive(Debug, Clone)]
pub struct EmergencePrediction {
    pub mutation: String,
    pub position: i32,
    pub escape_probability: f32,
    pub fitness_score: f32,
    pub cycle_phase: Phase,
    pub cycle_confidence: f32,
    pub emergence_probability: f32,
    pub predicted_timing: String,
    pub current_frequency: f32,
    pub velocity: f32,
}
```

---

## 9. VALIDATION & BENCHMARKING

### 9.1 Retrospective Validation

**Test:** Can we predict Omicron before it emerged?

```python
# scripts/validate_retrospective.py

import pandas as pd
from prism_ve import PRISMVEPredictor

# Train on data BEFORE Omicron (pre-Nov 2021)
training_cutoff = "2021-10-31"
test_date = "2021-11-01"  # Day before Omicron announced

# Load predictor trained on pre-Omicron data
predictor = PRISMVEPredictor.load_checkpoint("checkpoints/pre_omicron")

# Predict which mutations will emerge in next 3 months
predictions = predictor.predict_emergence(
    mutations=all_possible_rbd_mutations,  # 3,819 single mutations
    time_horizon="3_months",
    prediction_date=test_date
)

# Check: Were Omicron mutations in top predictions?
omicron_mutations = ["K417N", "E484A", "N501Y", "S371F", "G446S", ...]

for mut in omicron_mutations:
    pred = [p for p in predictions if p.mutation == mut][0]
    print(f"{mut}: Rank {pred.rank}, Prob {pred.emergence_probability:.3f}")

# Success metric: >50% of Omicron mutations in top 10%
top_10_pct = len(predictions) // 10
omicron_in_top = sum(1 for m in omicron_mutations
                     if any(p.mutation == m for p in predictions[:top_10_pct]))

recall = omicron_in_top / len(omicron_mutations)

print(f"\nOmicron Recall: {recall:.1%} in top 10%")
print("Target: >50% (better than random)")

if recall > 0.50:
    print("✅ PRISM-VE successfully predicts Omicron emergence!")
else:
    print("⚠️ Need calibration")
```

### 9.2 VASIL Benchmark Comparison

```python
# scripts/benchmark_vs_vasil_temporal.py

import pandas as pd

# Load VASIL's temporal predictions (if available)
# OR: Replicate VASIL's protocol and compare

# Test period: 2023 (held-out)
test_data = load_gisaid_frequencies("2023-01-01", "2023-12-31")

# Get positions that had dynamics in 2023
active_positions = test_data[test_data['velocity'].abs() > 0.05]['position'].unique()

print(f"Testing on {len(active_positions)} active positions")

correct = 0
total = 0

for position in active_positions:
    # Get Jan 2023 state
    state_jan = test_data[(test_data['position'] == position) &
                          (test_data['date'] == '2023-01-01')]

    # Get Dec 2023 state (ground truth)
    state_dec = test_data[(test_data['position'] == position) &
                          (test_data['date'] == '2023-12-01')]

    if len(state_jan) == 0 or len(state_dec) == 0:
        continue

    # Observed direction
    freq_change = state_dec.iloc[0]['frequency'] - state_jan.iloc[0]['frequency']
    observed = "RISE" if freq_change > 0.05 else "FALL" if freq_change < -0.05 else "STABLE"

    if observed == "STABLE":
        continue  # Skip stable cases

    # PRISM-VE prediction
    cycle_state = predictor.cycle_detector.detect_phase(position, ...)

    predicted = "RISE" if cycle_state.velocity > 0 else "FALL"

    total += 1
    if predicted == observed:
        correct += 1

accuracy = correct / total if total > 0 else 0

print(f"\nPRISM-VE Temporal Accuracy: {accuracy:.3f}")
print(f"VASIL Temporal Accuracy: 0.92 (baseline)")

if accuracy > 0.92:
    print("🏆 BEAT VASIL on temporal prediction!")
elif accuracy > 0.85:
    print("✅ Competitive with VASIL")
else:
    print("⚠️ Need calibration")
```

---

## 10. IMPLEMENTATION ROADMAP

### Week 1: Data Infrastructure
```
Day 1-2: Download/verify raw GISAID
  - [ ] Register at GISAID.org
  - [ ] Download metadata.tsv (or verify VASIL's is raw)
  - [ ] Verify no model-fitted columns

Day 3-4: Process GISAID frequencies
  - [ ] Implement process_gisaid_frequencies.py
  - [ ] Compute position-level frequencies
  - [ ] Compute velocities
  - [ ] Output: position_frequencies.csv

Day 5: Validate data quality
  - [ ] Plot frequency trajectories for known positions (484, 501)
  - [ ] Verify matches known variant emergence patterns
  - [ ] Check against VASIL's published trajectories
```

### Week 2: Cycle Detection Implementation
```
Day 6-7: Rust cycle module
  - [ ] Implement cycle.rs (Phase enum, CycleDetector)
  - [ ] Implement frequency trajectory loading
  - [ ] Implement phase classification

Day 8-9: Phase detection logic
  - [ ] Implement decision tree (6 phases)
  - [ ] Add confidence scoring
  - [ ] Test on known variants (Alpha, Delta, Omicron)

Day 10: Emergence timing prediction
  - [ ] Implement timing categories
  - [ ] Add velocity-based forecasting
  - [ ] Test retrospectively
```

### Week 3: Integration
```
Day 11-12: PRISM-VE integration
  - [ ] Create unified PRISMVEPredictor
  - [ ] Integrate Escape + Fitness + Cycle
  - [ ] Test end-to-end pipeline

Day 13-14: Validation
  - [ ] Retrospective: Predict Omicron (trained pre-Nov 2021)
  - [ ] Temporal: 2023 rise/fall accuracy
  - [ ] Compare to VASIL benchmarks

Day 15: Documentation
  - [ ] API documentation
  - [ ] Usage examples
  - [ ] Benchmark results
```

---

## 11. SUCCESS CRITERIA

### Minimum Viable Product (MVP)
```
[ ] Load GISAID frequency data
[ ] Classify positions into 6 phases
[ ] Predict emergence timing (categories)
[ ] Achieve >70% accuracy on 2023 rise/fall
```

### Target Performance
```
[ ] >85% accuracy on temporal dynamics
[ ] >50% recall on Omicron retrospective prediction
[ ] Correct phase classification for known variants
[ ] <100ms processing time (with GPU acceleration)
```

### Stretch Goals
```
[ ] >90% temporal accuracy (beat VASIL if they have this metric)
[ ] Geographic specificity (USA vs Germany differences)
[ ] Multi-wave prediction (predict second wave of same mutation)
[ ] Real-time dashboard integration
```

---

## 12. SCIENTIFIC HONESTY CHECKLIST

### Before Using Any VASIL Data:

```
Data Source Verification:
[ ] Verified GISAID frequencies are RAW aggregates (not VASIL's fitted)
[ ] If model-processed, downloaded raw GISAID ourselves
[ ] Documented all data sources in methods section

Parameter Independence:
[ ] Started with physics-based or neutral defaults
[ ] Calibrated on OUR training data (2021-2022)
[ ] Did NOT copy VASIL's fitted parameters
[ ] Documented our parameter values

Processing Independence:
[ ] Implemented OUR frequency computation
[ ] Implemented OUR phase classification
[ ] Implemented OUR emergence prediction
[ ] No use of VASIL's model outputs

Validation Honesty:
[ ] Test on held-out data (2023+)
[ ] Compare to VASIL's published predictions (benchmark)
[ ] Compare to observed frequencies (ground truth)
[ ] Report: "Same inputs, independent methods, benchmark comparison"
```

---

## 13. FILES TO CREATE

### Critical Path (Must Create):

```
1. scripts/process_gisaid_frequencies.py (500 lines)
   Purpose: Convert raw GISAID to position frequencies

2. crates/prism-ve/src/cycle.rs (800 lines)
   Purpose: Phase detection, emergence prediction

3. crates/prism-ve/src/lib.rs (300 lines)
   Purpose: Unified PRISM-VE predictor (Escape + Fitness + Cycle)

4. scripts/validate_retrospective.py (200 lines)
   Purpose: Omicron retrospective validation

5. scripts/benchmark_vs_vasil_temporal.py (300 lines)
   Purpose: Temporal accuracy benchmark
```

### Optional (GPU Optimization):

```
6. crates/prism-gpu/src/kernels/viral_evolution_cycle.cu (600 lines)
   Purpose: GPU-accelerated cycle detection
   Note: Only if CPU version is too slow
```

---

## 14. EXECUTION CHECKLIST

### Phase 1: Data Preparation
```
[ ] Download/verify raw GISAID metadata.tsv
[ ] Run process_gisaid_frequencies.py
[ ] Validate output matches expected format
[ ] Verify no VASIL model outputs used
```

### Phase 2: Implementation
```
[ ] Create cycle.rs module
[ ] Implement Phase enum and CycleDetector
[ ] Implement phase classification logic
[ ] Implement emergence timing prediction
[ ] Write unit tests
```

### Phase 3: Integration
```
[ ] Create unified PRISMVEPredictor
[ ] Integrate Escape + Fitness + Cycle
[ ] Test end-to-end on sample mutations
[ ] Verify outputs make sense
```

### Phase 4: Validation
```
[ ] Run retrospective Omicron prediction
[ ] Run 2023 temporal accuracy test
[ ] Compare to VASIL (if applicable)
[ ] Achieve >70% temporal accuracy
```

### Phase 5: Documentation
```
[ ] Write API documentation
[ ] Create usage examples
[ ] Document benchmark results
[ ] Prepare for publication
```

---

## 15. EXPECTED OUTCOMES

### After Week 1:
```
✅ GISAID frequencies processed independently
✅ Data verified to be primary source (not VASIL outputs)
✅ Ready for cycle detection implementation
```

### After Week 2:
```
✅ Cycle detector implemented (6-phase classification)
✅ Emergence timing prediction working
✅ Tested on known variants
```

### After Week 3:
```
✅ Fully integrated PRISM-VE (Escape + Fitness + Cycle)
✅ Validated retrospectively (Omicron prediction)
✅ Benchmarked temporally (2023 accuracy)
✅ Ready for publication
```

---

## 16. DELIVERABLES

**Code:**
- cycle.rs module (~800 lines)
- Processing scripts (~1,000 lines Python)
- Integration layer (~300 lines)
- Tests (~500 lines)

**Data:**
- Position frequencies (100 MB processed)
- Known variant database (1 MB)
- Validation results (JSON)

**Documentation:**
- API reference
- Usage guide
- Benchmark report

**Total:** ~2,600 lines of code + 100 MB data + documentation

---

## 17. QUESTIONS FOR CLAUDE AGENT

**When implementing, verify:**

1. **Data Source:** Is GISAID data RAW or VASIL-processed?
   - If VASIL: Check for "fitted", "predicted", "model" in columns
   - If found: Download raw GISAID directly

2. **Parameter Independence:** Are we using VASIL's fitted values?
   - If yes: STOP, calibrate our own
   - If no: Proceed

3. **Validation:** Are we training on VASIL's predictions?
   - Should: Train on observed frequencies
   - Should NOT: Train on VASIL's predictions

4. **Benchmark:** Are we comparing honestly?
   - Should: Same inputs, independent methods
   - Should NOT: Use VASIL's outputs as inputs

---

## READY FOR CLAUDE AGENT EXECUTION

**This blueprint provides:**
- ✅ Complete architecture design
- ✅ Detailed algorithms (decision trees, formulas)
- ✅ Full code templates (Rust + Python)
- ✅ Data processing pipelines
- ✅ Validation protocols
- ✅ Scientific integrity checks
- ✅ Success criteria
- ✅ Implementation timeline

**Claude agent can execute this plan to build Cycle module in 2-3 weeks!**

**Total Blueprint:** 1,200 lines of specification

**Ready to hand off to implementation agent.** 🚀
