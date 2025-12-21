/**
 * Feature Merge CUDA Kernels
 *
 * Merges TDA features (48-dim) with base mega-fused features (32-dim)
 * to produce combined 80-dim feature vectors.
 *
 * Three variants:
 * 1. merge_features_simple: Direct concatenation
 * 2. merge_features_weighted: Weighted combination with learnable scales
 * 3. merge_features_normalized: Z-score normalized merge
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BASE_FEATURES 32
#define TDA_FEATURES 48
#define TOTAL_FEATURES 80
#define WARP_SIZE 32

/**
 * Simple feature merge (concatenation)
 *
 * @param base_features  [n_residues * BASE_FEATURES] Base features
 * @param tda_features   [n_residues * TDA_FEATURES] TDA features
 * @param output        [n_residues * TOTAL_FEATURES] Merged output
 * @param n_residues    Number of residues
 */
extern "C" __global__ void merge_features_simple(
    const float* __restrict__ base_features,
    const float* __restrict__ tda_features,
    float* __restrict__ output,
    unsigned int n_residues
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int residue = idx / TOTAL_FEATURES;
    const int feature = idx % TOTAL_FEATURES;

    if (residue >= n_residues) return;

    float value;
    if (feature < BASE_FEATURES) {
        value = base_features[residue * BASE_FEATURES + feature];
    } else {
        value = tda_features[residue * TDA_FEATURES + (feature - BASE_FEATURES)];
    }

    output[residue * TOTAL_FEATURES + feature] = value;
}

/**
 * Weighted feature merge
 *
 * output[i] = base[i] * base_weight + tda[i] * tda_weight
 *
 * @param base_features  [n_residues * BASE_FEATURES] Base features
 * @param tda_features   [n_residues * TDA_FEATURES] TDA features
 * @param base_weights   [BASE_FEATURES] Per-feature weights for base
 * @param tda_weights    [TDA_FEATURES] Per-feature weights for TDA
 * @param output        [n_residues * TOTAL_FEATURES] Merged output
 * @param n_residues    Number of residues
 */
extern "C" __global__ void merge_features_weighted(
    const float* __restrict__ base_features,
    const float* __restrict__ tda_features,
    const float* __restrict__ base_weights,
    const float* __restrict__ tda_weights,
    float* __restrict__ output,
    unsigned int n_residues
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int residue = idx / TOTAL_FEATURES;
    const int feature = idx % TOTAL_FEATURES;

    if (residue >= n_residues) return;

    float value;
    if (feature < BASE_FEATURES) {
        const float raw = base_features[residue * BASE_FEATURES + feature];
        const float weight = base_weights[feature];
        value = raw * weight;
    } else {
        const int tda_idx = feature - BASE_FEATURES;
        const float raw = tda_features[residue * TDA_FEATURES + tda_idx];
        const float weight = tda_weights[tda_idx];
        value = raw * weight;
    }

    output[residue * TOTAL_FEATURES + feature] = value;
}

/**
 * Z-score normalized feature merge
 *
 * Normalizes each feature using precomputed mean/std, then concatenates.
 *
 * @param base_features  [n_residues * BASE_FEATURES] Base features
 * @param tda_features   [n_residues * TDA_FEATURES] TDA features
 * @param base_mean      [BASE_FEATURES] Mean for base features
 * @param base_std       [BASE_FEATURES] Std for base features
 * @param tda_mean       [TDA_FEATURES] Mean for TDA features
 * @param tda_std        [TDA_FEATURES] Std for TDA features
 * @param output        [n_residues * TOTAL_FEATURES] Merged output
 * @param n_residues    Number of residues
 */
extern "C" __global__ void merge_features_normalized(
    const float* __restrict__ base_features,
    const float* __restrict__ tda_features,
    const float* __restrict__ base_mean,
    const float* __restrict__ base_std,
    const float* __restrict__ tda_mean,
    const float* __restrict__ tda_std,
    float* __restrict__ output,
    unsigned int n_residues
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int residue = idx / TOTAL_FEATURES;
    const int feature = idx % TOTAL_FEATURES;

    if (residue >= n_residues) return;

    float value;
    if (feature < BASE_FEATURES) {
        const float raw = base_features[residue * BASE_FEATURES + feature];
        const float mean = base_mean[feature];
        const float std = base_std[feature];
        // Z-score with epsilon for numerical stability
        value = (raw - mean) / fmaxf(std, 1e-8f);
    } else {
        const int tda_idx = feature - BASE_FEATURES;
        const float raw = tda_features[residue * TDA_FEATURES + tda_idx];
        const float mean = tda_mean[tda_idx];
        const float std = tda_std[tda_idx];
        value = (raw - mean) / fmaxf(std, 1e-8f);
    }

    output[residue * TOTAL_FEATURES + feature] = value;
}

/**
 * In-place TDA feature normalization
 *
 * Normalizes TDA features in-place using precomputed statistics.
 *
 * @param features      [n_residues * TDA_FEATURES] TDA features (in-place)
 * @param mean          [TDA_FEATURES] Mean per feature
 * @param std           [TDA_FEATURES] Std per feature
 * @param n_residues    Number of residues
 */
extern "C" __global__ void normalize_tda_inplace(
    float* __restrict__ features,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    unsigned int n_residues
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int residue = idx / TDA_FEATURES;
    const int feature = idx % TDA_FEATURES;

    if (residue >= n_residues) return;

    const int offset = residue * TDA_FEATURES + feature;
    const float raw = features[offset];
    const float m = mean[feature];
    const float s = std[feature];

    features[offset] = (raw - m) / fmaxf(s, 1e-8f);
}

/**
 * Compute running statistics for normalization (Welford's algorithm)
 *
 * Updates mean and M2 (sum of squared differences) in parallel.
 *
 * @param features      [n_residues * n_features] Input features
 * @param mean          [n_features] Running mean (updated)
 * @param m2            [n_features] Running M2 (updated)
 * @param count         Current sample count
 * @param n_residues    Number of residues in this batch
 * @param n_features    Number of features per residue
 */
extern "C" __global__ void update_running_stats(
    const float* __restrict__ features,
    float* __restrict__ mean,
    float* __restrict__ m2,
    unsigned long long count,
    unsigned int n_residues,
    unsigned int n_features
) {
    const int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= n_features) return;

    float local_mean = mean[feature];
    float local_m2 = m2[feature];
    unsigned long long n = count;

    for (int i = 0; i < n_residues; i++) {
        const float x = features[i * n_features + feature];
        n++;
        const float delta = x - local_mean;
        local_mean += delta / (float)n;
        const float delta2 = x - local_mean;
        local_m2 += delta * delta2;
    }

    mean[feature] = local_mean;
    m2[feature] = local_m2;
}

/**
 * Finalize statistics: compute std from M2
 *
 * @param m2            [n_features] M2 values
 * @param std           [n_features] Output std values
 * @param count         Total sample count
 * @param n_features    Number of features
 */
extern "C" __global__ void finalize_stats(
    const float* __restrict__ m2,
    float* __restrict__ std,
    unsigned long long count,
    unsigned int n_features
) {
    const int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= n_features) return;

    if (count > 1) {
        std[feature] = sqrtf(m2[feature] / (float)(count - 1));
    } else {
        std[feature] = 1.0f;
    }
}

/**
 * Apply readout weights to combined features
 *
 * score = sum(features[i] * weights[i]) + bias
 *
 * @param features      [n_residues * TOTAL_FEATURES] Combined features
 * @param weights       [TOTAL_FEATURES] Readout weights
 * @param bias          Scalar bias
 * @param scores        [n_residues] Output scores
 * @param n_residues    Number of residues
 */
extern "C" __global__ void apply_readout(
    const float* __restrict__ features,
    const float* __restrict__ weights,
    float bias,
    float* __restrict__ scores,
    unsigned int n_residues
) {
    const int residue = blockIdx.x;
    const int lane = threadIdx.x;

    if (residue >= n_residues) return;

    // Each warp handles one residue
    __shared__ float partial[32];

    // Compute partial sum
    float sum = 0.0f;
    for (int f = lane; f < TOTAL_FEATURES; f += WARP_SIZE) {
        sum += features[residue * TOTAL_FEATURES + f] * weights[f];
    }

    // Warp reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane == 0) {
        scores[residue] = sum + bias;
    }
}

/**
 * Sigmoid activation
 *
 * @param input         [n] Input values
 * @param output        [n] Output probabilities
 * @param n             Number of elements
 */
extern "C" __global__ void sigmoid(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float x = input[idx];
    // Numerically stable sigmoid
    if (x >= 0) {
        output[idx] = 1.0f / (1.0f + expf(-x));
    } else {
        const float e = expf(x);
        output[idx] = e / (1.0f + e);
    }
}
