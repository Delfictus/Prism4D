# CUDA Kernel Patterns for PRISM>4D

## Kernel Launch Configuration

### RTX 3060 Laptop (sm_75) Optimal Settings
```cuda
// In kernel signature
extern "C" __global__ void __launch_bounds__(256, 2)  // 256 threads, 2 blocks/SM
mega_fused_batch_detection(...)

// Host-side launch
dim3 grid(n_structures);  // One block per structure
dim3 block(256);          // 256 threads per block

cudaFuncSetCacheConfig(mega_fused_batch_detection, cudaFuncCachePreferL1);
```

### Memory Configuration
```
RTX 3060 Laptop GPU:
- Compute capability: 7.5 (Turing)
- SMs: 30
- Shared memory/SM: 64KB (48KB usable per block)
- L1 cache: 32KB per SM
- Registers: 65536 per SM
- Max threads/block: 1024
- Warp size: 32
```

## Shared Memory Structure

### Actual SharedMem from mega_fused_batch.cu
```cuda
struct __align__(16) BatchSharedMem {
    // Stage 1: Contact network (reused each tile)
    float distance_tile[TILE_SIZE][TILE_SIZE];  // 256×256×4 = 256KB (TOO BIG)
    // SOLUTION: Process in 32×32 tiles
    
    // Stage 2: Per-residue features
    float3 ca_coords[TILE_SIZE];        // 256×12 = 3KB
    float conservation[TILE_SIZE];       // 256×4 = 1KB
    float bfactor[TILE_SIZE];           // 256×4 = 1KB
    float burial[TILE_SIZE];            // 256×4 = 1KB
    
    // Stage 3: Centrality
    float degree[TILE_SIZE];            // 256×4 = 1KB
    float centrality[TILE_SIZE];        // 256×4 = 1KB
    
    // Stage 4: Reservoir (compressed)
    float reservoir_state[TILE_SIZE][8]; // 256×8×4 = 8KB
    
    // Stage 5: Consensus
    float consensus_score[TILE_SIZE];   // 256×4 = 1KB
    int confidence[TILE_SIZE];          // 256×4 = 1KB
    int signal_mask[TILE_SIZE];         // 256×4 = 1KB
    int pocket_assignment[TILE_SIZE];   // 256×4 = 1KB
    
    // Stage 7: Fitness (NEW)
    float fitness_features[TILE_SIZE][4]; // 256×4×4 = 4KB
    
    // Stage 8: Cycle (NEW)
    float cycle_features[TILE_SIZE][5];   // 256×5×4 = 5KB
    
    // Union-Find scratch
    int parent[TILE_SIZE];              // 256×4 = 1KB
};
// Total: ~30KB (fits in 48KB limit)
```

## Tiled Processing Pattern

### Processing Large Residue Sets
```cuda
#define TILE_SIZE 256
#define BLOCK_SIZE 256

__global__ void mega_fused_batch_detection(...) {
    int structure_idx = blockIdx.x;
    BatchStructureDesc desc = descriptors[structure_idx];
    
    __shared__ BatchSharedMem smem;
    
    // Process residues in tiles
    int n_tiles = (desc.n_residues + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < n_tiles; tile++) {
        int local_idx = threadIdx.x;
        int global_idx = tile * TILE_SIZE + local_idx;
        bool active = (global_idx < desc.n_residues);
        
        // Load tile data to shared memory
        if (active) {
            int atom_idx = ca_indices_packed[desc.residue_offset + global_idx];
            smem.ca_coords[local_idx] = make_float3(
                atoms_packed[(desc.atom_offset + atom_idx) * 3 + 0],
                atoms_packed[(desc.atom_offset + atom_idx) * 3 + 1],
                atoms_packed[(desc.atom_offset + atom_idx) * 3 + 2]
            );
        }
        __syncthreads();
        
        // Process stages...
        batch_stage1_contacts(desc, tile, &smem, params);
        batch_stage2_geometry(desc, tile, &smem, params);
        // ...
        
        // Write outputs
        if (active) {
            int out_idx = desc.residue_offset + global_idx;
            consensus_out[out_idx] = smem.consensus_score[local_idx];
        }
        __syncthreads();
    }
}
```

## L1 Cache Optimization

### Read-Only Data Access
```cuda
// Use __ldg() for read-only global memory (texture cache path)
BatchStructureDesc desc;
desc.atom_offset = __ldg(&descriptors[structure_idx].atom_offset);
desc.residue_offset = __ldg(&descriptors[structure_idx].residue_offset);
desc.n_atoms = __ldg(&descriptors[structure_idx].n_atoms);
desc.n_residues = __ldg(&descriptors[structure_idx].n_residues);

// For arrays, use __restrict__ and const
const float* __restrict__ atoms_packed
```

### Coalesced Memory Access
```cuda
// GOOD: Coalesced (threads access consecutive addresses)
float val = data[blockIdx.x * blockDim.x + threadIdx.x];

// BAD: Strided (threads access with stride)
float val = data[threadIdx.x * stride + offset];

// For AoS data, load as float3/float4
float3 coord = *reinterpret_cast<const float3*>(&atoms[idx * 3]);
```

## Stage Implementation Pattern

### Device Function Template
```cuda
__device__ void batch_stage7_fitness_features(
    const float* __restrict__ bfactor_packed,
    const int* __restrict__ residue_types_packed,
    int residue_offset,
    int n_residues,
    int tile_idx,
    BatchSharedMem* smem,
    const MegaFusedParams* params
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;
    bool active = (local_idx < TILE_SIZE && global_idx < n_residues);
    
    if (active) {
        // Load from global with __ldg
        float bfactor = __ldg(&bfactor_packed[residue_offset + global_idx]);
        int res_type = __ldg(&residue_types_packed[residue_offset + global_idx]);
        
        // Compute using shared memory from previous stages
        float burial = smem->burial[local_idx];
        float centrality = smem->centrality[local_idx];
        
        // Get amino acid properties from constant memory
        float hydro = c_residue_hydrophobicity[res_type];
        float volume = c_residue_volume[res_type];
        
        // Feature 92: ΔΔG_binding
        float ddg_binding = (hydro - 0.5f) * centrality * (1.0f - burial);
        
        // Feature 93: ΔΔG_stability
        float ddg_stability = burial * (volume - 0.5f) * (1.0f - bfactor);
        
        // Feature 94: Expression
        float expression = 0.3f + 0.5f * (1.0f - burial) + 0.2f * bfactor;
        
        // Feature 95: Transmissibility
        float transmit = __fdividef(1.0f, 1.0f + expf(ddg_binding)) *
                         __fdividef(1.0f, 1.0f + expf(ddg_stability)) *
                         expression;
        
        // Store to shared memory
        smem->fitness_features[local_idx][0] = ddg_binding;
        smem->fitness_features[local_idx][1] = ddg_stability;
        smem->fitness_features[local_idx][2] = expression;
        smem->fitness_features[local_idx][3] = transmit;
    }
    __syncthreads();
}
```

## Constant Memory for Amino Acid Properties

```cuda
// In mega_fused_batch.cu (top of file)
__constant__ float c_residue_hydrophobicity[20] = {
    0.616f, 0.000f, 0.236f, 0.028f, 0.680f,  // A,R,N,D,C
    0.251f, 0.043f, 0.501f, 0.165f, 0.943f,  // Q,E,G,H,I
    0.943f, 0.283f, 0.738f, 1.000f, 0.711f,  // L,K,M,F,P
    0.359f, 0.450f, 0.878f, 0.880f, 0.825f   // S,T,W,Y,V
};

__constant__ float c_residue_volume[20] = {
    0.152f, 0.476f, 0.243f, 0.220f, 0.190f,  // A,R,N,D,C
    0.302f, 0.280f, 0.100f, 0.333f, 0.341f,  // Q,E,G,H,I
    0.341f, 0.373f, 0.324f, 0.402f, 0.220f,  // L,K,M,F,P
    0.165f, 0.220f, 0.476f, 0.422f, 0.275f   // S,T,W,Y,V
};

__constant__ float c_residue_charge[20] = {
    0.0f,  1.0f,  0.0f, -1.0f,  0.0f,  // A,R,N,D,C
    0.0f, -1.0f,  0.0f,  0.5f,  0.0f,  // Q,E,G,H,I
    0.0f,  1.0f,  0.0f,  0.0f,  0.0f,  // L,K,M,F,P
    0.0f,  0.0f,  0.0f,  0.0f,  0.0f   // S,T,W,Y,V
};
```

## Writing Combined Features to Global Memory

### Prevent Dead Code Elimination
```cuda
__device__ void stage6_5_combine_features(
    int n_residues,
    int tile_idx,
    float* __restrict__ combined_features_out,
    BatchSharedMem* smem
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;
    
    if (local_idx < TILE_SIZE && global_idx < n_residues) {
        int base = global_idx * TOTAL_COMBINED_FEATURES;  // 101 features
        
        // Use inline PTX to force writes (prevent DCE)
        #define FORCE_STORE(addr, val) \
            asm volatile("st.global.f32 [%0], %1;" : : "l"(addr), "f"(val))
        
        // Features 0-47: TDA (from smem->tda_features)
        for (int i = 0; i < 48; i++) {
            FORCE_STORE(&combined_features_out[base + i], smem->tda_features[local_idx][i]);
        }
        
        // Features 48-79: Reservoir
        for (int i = 0; i < 32; i++) {
            FORCE_STORE(&combined_features_out[base + 48 + i], smem->reservoir_state[local_idx][i % 8]);
        }
        
        // Features 80-91: Physics
        for (int i = 0; i < 12; i++) {
            FORCE_STORE(&combined_features_out[base + 80 + i], smem->physics_features[local_idx][i]);
        }
        
        // Features 92-95: Fitness (Stage 7)
        FORCE_STORE(&combined_features_out[base + 92], smem->fitness_features[local_idx][0]);
        FORCE_STORE(&combined_features_out[base + 93], smem->fitness_features[local_idx][1]);
        FORCE_STORE(&combined_features_out[base + 94], smem->fitness_features[local_idx][2]);
        FORCE_STORE(&combined_features_out[base + 95], smem->fitness_features[local_idx][3]);
        
        // Features 96-100: Cycle (Stage 8)
        FORCE_STORE(&combined_features_out[base + 96], smem->cycle_features[local_idx][0]);
        FORCE_STORE(&combined_features_out[base + 97], smem->cycle_features[local_idx][1]);
        FORCE_STORE(&combined_features_out[base + 98], smem->cycle_features[local_idx][2]);
        FORCE_STORE(&combined_features_out[base + 99], smem->cycle_features[local_idx][3]);
        FORCE_STORE(&combined_features_out[base + 100], smem->cycle_features[local_idx][4]);
        
        #undef FORCE_STORE
    }
}
```

## Host Launcher Function

```cuda
extern "C" {

cudaError_t launch_mega_fused_batch_prism4d(
    // Input arrays (packed)
    const float* d_atoms_packed,
    const int* d_ca_indices_packed,
    const float* d_conservation_packed,
    const float* d_bfactor_packed,
    const float* d_burial_packed,
    const int* d_residue_types_packed,      // NEW for Stage 7
    const float* d_gisaid_frequencies,      // NEW for Stage 8
    const float* d_gisaid_velocities,       // NEW for Stage 8
    
    // Structure descriptors
    const BatchStructureDesc* d_descriptors,
    int n_structures,
    
    // Output arrays
    float* d_consensus_out,
    int* d_confidence_out,
    int* d_signal_mask_out,
    int* d_pocket_assignment_out,
    float* d_centrality_out,
    float* d_combined_features_out,         // NEW: 101-dim output
    
    // Parameters
    const MegaFusedParams* d_params,
    cudaStream_t stream
) {
    dim3 grid(n_structures);
    dim3 block(BLOCK_SIZE);
    
    cudaFuncSetCacheConfig(mega_fused_batch_detection_prism4d, cudaFuncCachePreferL1);
    
    mega_fused_batch_detection_prism4d<<<grid, block, 0, stream>>>(
        d_atoms_packed, d_ca_indices_packed, d_conservation_packed,
        d_bfactor_packed, d_burial_packed, d_residue_types_packed,
        d_gisaid_frequencies, d_gisaid_velocities,
        d_descriptors, n_structures,
        d_consensus_out, d_confidence_out, d_signal_mask_out,
        d_pocket_assignment_out, d_centrality_out, d_combined_features_out,
        d_params
    );
    
    return cudaGetLastError();
}

}  // extern "C"
```
