// PRISM-4D: Mega-Fused VASIL Kernel with FluxNet-Trainable Parameters
//
// Architecture: Single kernel computes VASIL accuracy from FluxNet parameters
//
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║  FLUXNET-TRAINABLE PARAMETERS (optimized by RL)                           ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║  1. fluxnet_ic50[11]        - Epitope binding affinities                  ║
// ║  2. fluxnet_epitope_power[11] - Epitope contribution exponents            ║
// ║  3. fluxnet_rise_bias       - Threshold bias for RISE predictions         ║
// ║  4. fluxnet_fall_bias       - Threshold bias for FALL predictions         ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
//
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║  DATA-DRIVEN VALUES (from VASIL/GInPipe, NOT trainable)                   ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║  - frequencies[variant, day] - Observed variant frequencies πx(t)         ║
// ║  - incidence[day]            - Infection counts from GInPipe              ║
// ║  - epitope_escape[variant, epitope] - DMS escape fractions                ║
// ║  - actual_directions[variant, day] - Ground truth RISE/FALL               ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
//
// Formula with FluxNet parameters:
//   b_θ = c(t) / (FR · fluxnet_ic50[θ] + c(t))
//   P_neut = 1 - Π_θ (1 - b_θ)^fluxnet_epitope_power[θ]
//   
//   Prediction RISE if: gamma_min > fluxnet_rise_bias
//   Prediction FALL if: gamma_max < fluxnet_fall_bias

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// === DEBUG CONFIGURATION ===
#define DEBUG_VARIANT 0    // Debug first variant only
#define DEBUG_DAY 0        // Debug first eval day only
#define DEBUG_PK 37        // Debug middle PK combination
#define ENABLE_DEBUG 1     // Set to 0 to disable all debug output

constexpr int MAX_DELTA_DAYS = 1500;
constexpr int N_EPITOPES = 11;
constexpr int N_PK = 75;
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

constexpr float NEGLIGIBLE_CHANGE_THRESHOLD = 0.05f;
constexpr float MIN_FREQUENCY_THRESHOLD = 0.03f;

// PK grid (fixed per VASIL methodology)
__constant__ float c_tmax[5] = {14.0f, 17.5f, 21.0f, 24.5f, 28.0f};
__constant__ float c_thalf[15] = {
    25.0f, 28.14f, 31.29f, 34.43f, 37.57f,
    40.71f, 43.86f, 47.0f, 50.14f, 53.29f,
    56.43f, 59.57f, 62.71f, 65.86f, 69.0f
};

// Default IC50 (VASIL-calibrated baseline, FluxNet starts here)
__constant__ float c_baseline_ic50[11] = {
    0.85f, 1.12f, 0.93f, 1.05f, 0.98f,
    1.21f, 0.89f, 1.08f, 0.95f, 1.03f, 1.00f
};

// Default epitope power (uniform = 1.0, FluxNet learns deviations)
__constant__ float c_baseline_power[11] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
};

//=============================================================================
// FluxNet Parameters Structure (passed to kernel)
//=============================================================================
struct FluxNetParams {
    float ic50[N_EPITOPES];           // Trained binding affinities
    float epitope_power[N_EPITOPES];  // Trained contribution exponents
    float rise_bias;                  // Trained RISE threshold adjustment
    float fall_bias;                  // Trained FALL threshold adjustment
};

//=============================================================================
// DEVICE: P_neut with FluxNet-trained parameters
//=============================================================================
__device__ __forceinline__ float compute_p_neut_fluxnet(
    const float* __restrict__ escape_x,
    const float* __restrict__ escape_y,
    const float* __restrict__ fluxnet_ic50,
    const float* __restrict__ fluxnet_power,
    int delta_t,
    int pk_idx
) {
    if (delta_t <= 0 || delta_t >= MAX_DELTA_DAYS) return 0.0f;
    
    // PK pharmacokinetics (data-driven, not FluxNet-trained)
    const float tmax = c_tmax[pk_idx / 15];
    const float thalf = c_thalf[pk_idx % 15];
    const float ke = __logf(2.0f) / thalf;
    
    float ka;
    const float ke_tmax = ke * tmax;
    if (ke_tmax > __logf(2.0f) + 0.01f) {
        ka = __logf(ke_tmax / (ke_tmax - __logf(2.0f)));
    } else {
        ka = ke * 2.0f;
    }
    
    const float exp_ke_tmax = __expf(-ke * tmax);
    const float exp_ka_tmax = __expf(-ka * tmax);
    const float pk_denom = exp_ke_tmax - exp_ka_tmax;
    
    if (fabsf(pk_denom) < 1e-10f) return 0.0f;
    
    // Antibody concentration (data-driven from PK model)
    const float t = (float)delta_t;
    float c_t = (__expf(-ke * t) - __expf(-ka * t)) / pk_denom;
    c_t = fmaxf(0.0f, c_t);
    
    if (c_t < 1e-8f) return 0.0f;
    
    // P_neut with FluxNet-trained IC50 and epitope powers
    float log_product = 0.0f;
    
    #pragma unroll
    for (int e = 0; e < N_EPITOPES; e++) {
        // Fold resistance (data-driven from DMS escape)
        float fold_res;
        if (escape_x[e] > 0.01f) {
            fold_res = (1.0f + escape_y[e]) / (1.0f + escape_x[e]);
        } else {
            fold_res = 1.0f + escape_y[e];
        }
        fold_res = fmaxf(0.1f, fminf(fold_res, 100.0f));
        
        // FluxNet-trained IC50 (or baseline if NULL)
        const float ic50_e = (fluxnet_ic50 != nullptr) ? fluxnet_ic50[e] : c_baseline_ic50[e];
        
        // VASIL formula: b_θ = c(t) / (FR · IC50 + c(t))
        const float denom = fold_res * ic50_e + c_t;
        const float b_theta = (denom > 1e-10f) ? (c_t / denom) : 0.0f;
        
        // FluxNet-trained epitope power (contribution exponent)
        const float power_e = (fluxnet_power != nullptr) ? fluxnet_power[e] : c_baseline_power[e];
        
        // P_neut = 1 - Π_θ (1 - b_θ)^power_θ
        // Using log for numerical stability: log(1-b)^p = p * log(1-b)
        const float one_minus_b = fmaxf(1e-10f, 1.0f - b_theta);
        log_product += power_e * __logf(one_minus_b);
    }
    
    return 1.0f - __expf(log_product);
}

//=============================================================================
// DEVICE: Compute immunity with warp-parallel reduction
//=============================================================================
__device__ double compute_immunity_fluxnet(
    const float* __restrict__ epitope_escape,
    const float* __restrict__ frequencies,
    const double* __restrict__ incidence,
    const float* __restrict__ fluxnet_ic50,
    const float* __restrict__ fluxnet_power,
    const float* __restrict__ escape_y,
    int y_idx,
    int t_abs,
    int pk_idx,
    int n_variants,
    int max_history_days,
    cg::thread_block_tile<WARP_SIZE>& warp
) {
    double warp_sum = 0.0;
    const int lane = warp.thread_rank();
    
    // Integration over infection history (data-driven: freq, incidence from VASIL)
    for (int s = lane; s < t_abs && s < max_history_days; s += WARP_SIZE) {
        const int delta_t = t_abs - s;
        if (delta_t <= 0 || delta_t >= MAX_DELTA_DAYS) continue;
        
        const double inc = incidence[s];
        if (inc < 1.0) continue;
        
        // Sum over circulating variants (data-driven frequency weighting)
        for (int x = 0; x < n_variants; x++) {
            const float freq = frequencies[x * max_history_days + s];
            if (freq < 0.001f) continue;
            
            float escape_x[N_EPITOPES];
            #pragma unroll
            for (int e = 0; e < N_EPITOPES; e++) {
                escape_x[e] = epitope_escape[x * N_EPITOPES + e];
            }
            
            // P_neut with FluxNet-trained parameters
            const float p_neut = compute_p_neut_fluxnet(
                escape_x, escape_y, fluxnet_ic50, fluxnet_power, delta_t, pk_idx
            );
            
            if (p_neut > 1e-8f) {
                warp_sum += (double)freq * inc * (double)p_neut;
            }
        }
    }
    
    // Warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        warp_sum += warp.shfl_down(warp_sum, offset);
    }
    
    return warp_sum;
}

//=============================================================================
// MEGA FUSED KERNEL: FluxNet Parameters → VASIL Accuracy
//=============================================================================
extern "C" __global__ void mega_fused_vasil_fluxnet(
    // Data-driven inputs (from VASIL/GInPipe)
    const float* __restrict__ epitope_escape,    // [n_variants × 11]
    const float* __restrict__ frequencies,       // [n_variants × max_history_days]
    const double* __restrict__ incidence,        // [max_history_days]
    const int8_t* __restrict__ actual_directions,// [n_variants × n_eval_days]
    const float* __restrict__ freq_changes,      // [n_variants × n_eval_days]
    
    // FluxNet-trained parameters (RL-optimized)
    const float* __restrict__ fluxnet_ic50,      // [11] trained IC50
    const float* __restrict__ fluxnet_power,     // [11] trained epitope powers
    const float fluxnet_rise_bias,               // trained RISE threshold
    const float fluxnet_fall_bias,               // trained FALL threshold
    
    // Output counters
    unsigned int* __restrict__ correct_count,
    unsigned int* __restrict__ total_count,
    
    // Constants
    const double population,
    const int n_variants,
    const int n_eval_days,
    const int max_history_days,
    const int eval_start_offset
) {
    const int y_idx = blockIdx.x;
    const int t_eval = blockIdx.y;
    
    if (y_idx >= n_variants || t_eval >= n_eval_days) return;
    
    const int t_abs = eval_start_offset + t_eval;
    if (t_abs >= max_history_days || t_abs < 1) return;
    
    // Check VASIL exclusion criteria (data-driven)
    const int sample_idx = y_idx * n_eval_days + t_eval;
    const int8_t actual_dir = actual_directions[sample_idx];
    const float rel_change = freq_changes[sample_idx];
    
    if (actual_dir == 0) return;  // Pre-excluded
    if (fabsf(rel_change) < NEGLIGIBLE_CHANGE_THRESHOLD) return;
    
    const float current_freq = frequencies[y_idx * max_history_days + t_abs];
    if (current_freq < MIN_FREQUENCY_THRESHOLD) return;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);
    
    // Load FluxNet parameters into shared memory
    __shared__ float smem_ic50[N_EPITOPES];
    __shared__ float smem_power[N_EPITOPES];
    __shared__ float smem_escape_y[N_EPITOPES];
    
    if (threadIdx.x < N_EPITOPES) {
        smem_ic50[threadIdx.x] = (fluxnet_ic50 != nullptr) ? 
            fluxnet_ic50[threadIdx.x] : c_baseline_ic50[threadIdx.x];
        smem_power[threadIdx.x] = (fluxnet_power != nullptr) ? 
            fluxnet_power[threadIdx.x] : c_baseline_power[threadIdx.x];
        smem_escape_y[threadIdx.x] = epitope_escape[y_idx * N_EPITOPES + threadIdx.x];
    }
    block.sync();
    
    // Compute gamma envelope across all 75 PK combinations
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    
    __shared__ double smem_immunity_y[N_PK];
    
    // Phase 1: Compute immunity for variant y at each PK (FluxNet IC50 + power)
    for (int pk = warp_id; pk < N_PK; pk += warps_per_block) {
        double immunity_y = compute_immunity_fluxnet(
            epitope_escape, frequencies, incidence,
            smem_ic50, smem_power, smem_escape_y,
            y_idx, t_abs, pk, n_variants, max_history_days, warp
        );
        
        if (warp.thread_rank() == 0) {
            smem_immunity_y[pk] = immunity_y;
        }
    }
    block.sync();
    
    // Phase 2: Compute Σx Sx(t) per VASIL formula (Extended Data Fig 6a)
    // BUGFIX: Must compute immunity_x for EACH variant x (not reuse immunity_y!)

    // Warp-parallel computation of Σx Sx(t)
    double warp_sum_sx = 0.0;

    // Each warp processes variants in parallel
    for (int x_base = 0; x_base < n_variants; x_base += warps_per_block) {
        const int x = x_base + warp_id;
        if (x >= n_variants) break;

        const float freq_x = frequencies[x * max_history_days + t_abs];
        if (freq_x < 0.001f) continue;

        // Compute immunity_x using this warp (middle PK = 37)
        double immunity_x = 0.0;
        for (int s = warp.thread_rank(); s < t_abs && s < max_history_days; s += WARP_SIZE) {
            const int delta_t = t_abs - s;
            if (delta_t <= 0) continue;
            const double inc = incidence[s];
            if (inc < 1.0) continue;

            for (int x2 = 0; x2 < n_variants; x2++) {
                const float freq_x2 = frequencies[x2 * max_history_days + s];
                if (freq_x2 < 0.001f) continue;

                float escape_x2[N_EPITOPES], escape_x[N_EPITOPES];
                for (int e = 0; e < N_EPITOPES; e++) {
                    escape_x2[e] = epitope_escape[x2 * N_EPITOPES + e];
                    escape_x[e] = epitope_escape[x * N_EPITOPES + e];
                }

                const float p_neut = compute_p_neut_fluxnet(
                    escape_x2, escape_x, fluxnet_ic50, fluxnet_power, delta_t, 37
                );

                if (p_neut > 1e-8f) {
                    immunity_x += (double)freq_x2 * inc * (double)p_neut;
                }
            }
        }

        // Warp reduction
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            immunity_x += warp.shfl_down(immunity_x, offset);
        }

        if (warp.thread_rank() == 0 && x < n_variants) {
            const double susceptibility_x = fmax(0.0, population - immunity_x);
            warp_sum_sx += susceptibility_x;  // VASIL: Σx Sx(t)
        }
    }

    // Reduce across warps to get Σx Sx(t) (VASIL formula)
    __shared__ double smem_sum_sx;
    if (threadIdx.x == 0) {
        smem_sum_sx = (warp_sum_sx > 0.0) ? warp_sum_sx : (population * 0.5);
    }
    block.sync();

    // Phase 3: Compute gamma envelope (VASIL Extended Data Fig 6a)
    double local_min = 1e10;
    double local_max = -1e10;

    for (int pk = threadIdx.x; pk < N_PK; pk += BLOCK_SIZE) {
        const double immunity_y = smem_immunity_y[pk];
        const double susceptibility_y = fmax(0.0, population - immunity_y);

        // VASIL formula: γy(t) = Sy(t) / Σx Sx(t) (NO -1!)
        double gamma;
        if (smem_sum_sx > 0.1) {
            gamma = susceptibility_y / smem_sum_sx;  // FIXED: No -1
        } else {
            gamma = 0.0;
        }

        local_min = fmin(local_min, gamma);
        local_max = fmax(local_max, gamma);
    }
    
    // Block reduction for envelope
    __shared__ double smem_min[BLOCK_SIZE];
    __shared__ double smem_max[BLOCK_SIZE];
    smem_min[threadIdx.x] = local_min;
    smem_max[threadIdx.x] = local_max;
    block.sync();
    
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            smem_min[threadIdx.x] = fmin(smem_min[threadIdx.x], smem_min[threadIdx.x + stride]);
            smem_max[threadIdx.x] = fmax(smem_max[threadIdx.x], smem_max[threadIdx.x + stride]);
        }
        block.sync();
    }
    
    // Phase 4: Make prediction with FluxNet-trained biases
    if (threadIdx.x == 0) {
        const double gamma_min = smem_min[0];
        const double gamma_max = smem_max[0];
        
        int8_t predicted_dir;
        // FluxNet-trained thresholds (not fixed at 0.0)
        if (gamma_min > (double)fluxnet_rise_bias && gamma_max > (double)fluxnet_rise_bias) {
            predicted_dir = 1;  // RISE
        } else if (gamma_min < (double)fluxnet_fall_bias && gamma_max < (double)fluxnet_fall_bias) {
            predicted_dir = -1; // FALL
        } else {
            predicted_dir = 0;  // UNDECIDED
        }
        
        if (predicted_dir != 0) {
            atomicAdd(total_count, 1u);
            if (predicted_dir == actual_dir) {
                atomicAdd(correct_count, 1u);
            }
        }
    }
}

//=============================================================================
// Helper kernels
//=============================================================================
extern "C" __global__ void reset_counters(
    unsigned int* correct_count,
    unsigned int* total_count
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *correct_count = 0;
        *total_count = 0;
    }
}

extern "C" __global__ void get_accuracy(
    const unsigned int* correct_count,
    const unsigned int* total_count,
    float* accuracy_out
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *accuracy_out = (*total_count > 0) ? 
            ((float)*correct_count / (float)*total_count) : 0.0f;
    }
}
