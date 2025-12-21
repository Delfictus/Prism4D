---
name: uspto-patent-drafting
description: "USPTO patent application drafting for AI/ML and neuromorphic computing inventions. Use when: (1) Drafting method and system claims for PRISM-4D, (2) Differentiating from EVEscape and VASIL prior art, (3) Structuring dependent claim hierarchies, (4) Navigating 35 USC 101 eligibility for AI inventions, (5) Writing specifications with enablement. Encodes 2024 USPTO AI guidance, MPEP requirements, and neuromorphic/RL claim examples."
---

# USPTO Patent Drafting Skill

## Purpose
Guide patent application drafting for PRISM-4D's novel integration of neuromorphic computing and reinforcement learning for biological prediction, ensuring claims survive 35 U.S.C. 101 examination and differentiate from prior art.

## When to Use This Skill
- Drafting independent and dependent claims
- Writing specifications with sufficient enablement
- Responding to 101 (eligibility) rejections
- Differentiating from EVEscape, VASIL, and other prior art
- Structuring claim hierarchies for prosecution flexibility

## Patent Application Structure

### Required Sections
```
1. Title
2. Cross-Reference to Related Applications
3. Statement Regarding Federally Sponsored Research
4. Background of the Invention
5. Brief Summary of the Invention
6. Brief Description of Drawings
7. Detailed Description
8. Claims
9. Abstract
```

## Claim Drafting for AI/ML Inventions

### 2024 USPTO AI Guidance Key Points

**Step 2A, Prong 1** - Does claim recite abstract idea?
- Mathematical concepts (formulas, calculations)
- Mental processes (observation, evaluation, judgment)
- Certain methods of organizing human activity

**Step 2A, Prong 2** - Practical application integration?
- Specific technological improvement
- Particular machine implementation
- Transformation of data into different state

**For PRISM-4D claims to pass 101:**
1. Recite specific hardware (GPU, neuromorphic processor)
2. Describe particular solution, not just goal
3. Show technical improvement (speed, accuracy)
4. Include real-world application (viral prediction)

### Independent Claim Templates

**Method Claim (Broadest)**
```
1. A computer-implemented method for predicting viral variant 
   emergence, comprising:
   
   (a) receiving, by a processor, structural data representing 
       a viral protein comprising a plurality of amino acid 
       residues;
       
   (b) extracting, by a graphics processing unit (GPU), a 
       feature vector for each residue using a fused kernel 
       executing stages including:
       (i) computing a contact network based on inter-residue 
           distances;
       (ii) generating topological features using persistent 
            homology;
       (iii) processing the topological features through a 
             dendritic reservoir network to produce neuromorphic 
             features;
             
   (c) determining, by a reinforcement learning agent, a 
       prediction of variant frequency trajectory based on the 
       feature vectors, wherein the reinforcement learning agent 
       comprises a Q-table mapping discretized states to actions; 
       and
       
   (d) outputting the prediction indicating whether the viral 
       variant will increase or decrease in frequency.
```

**System Claim**
```
10. A system for viral evolution prediction, comprising:
    
    a memory storing instructions;
    
    a graphics processing unit (GPU) configured to execute a 
    mega-fused kernel, the kernel comprising:
      a contact network stage computing inter-residue distances;
      a topological data analysis stage generating persistence 
        diagrams;
      a dendritic reservoir stage implementing spiking neural 
        dynamics; and
      a feature combination stage producing a multi-dimensional 
        feature vector;
    
    a processor coupled to the GPU and configured to:
      receive the feature vector from the GPU;
      discretize the feature vector into a state representation;
      query a Q-table using the state representation to obtain 
        action values; and
      output a prediction based on the action values.
```

**Computer-Readable Medium Claim**
```
20. A non-transitory computer-readable medium storing 
    instructions that, when executed by a processor, cause 
    the processor to perform operations comprising:
    [... mirror method claim steps ...]
```

### Dependent Claim Hierarchy

**Narrow the Technical Implementation**
```
2. The method of claim 1, wherein the fused kernel executes 
   all stages in a single GPU kernel launch without 
   intermediate memory transfers.

3. The method of claim 1, wherein the GPU comprises a 
   consumer-grade graphics card with at least 6 GB of memory.

4. The method of claim 1, wherein the contact network is 
   computed using a distance threshold of 12 Angstroms between 
   alpha-carbon atoms.
```

**Narrow the Neuromorphic Component**
```
5. The method of claim 1, wherein the dendritic reservoir 
   network comprises:
   a plurality of virtual neurons with membrane potential 
   dynamics; and
   recurrent connections with exponential decay weights.

6. The method of claim 5, wherein the membrane potential 
   dynamics follow a leaky integrate-and-fire model.

7. The method of claim 1, wherein the neuromorphic features 
   comprise 32 dimensions derived from reservoir state vectors.
```

**Narrow the RL Component**
```
8. The method of claim 1, wherein the Q-table comprises 256 
   discrete states derived from binning continuous features.

9. The method of claim 1, wherein the reinforcement learning 
   agent is trained using Q-learning with asymmetric reward 
   weighting for class imbalance.

10. The method of claim 1, wherein the actions comprise a 
    binary prediction of frequency increase or decrease.
```

**Narrow the Application Domain**
```
11. The method of claim 1, wherein the viral protein comprises 
    a receptor binding domain of a coronavirus spike protein.

12. The method of claim 11, wherein the prediction is generated 
    for a lineage defined by mutations at positions selected 
    from the group consisting of 417, 484, 490, 493, 498, and 501.

13. The method of claim 1, further comprising receiving 
    epidemiological data indicating historical frequency of the 
    viral variant across geographic regions.
```

## Prior Art Differentiation

### EVEscape (Bloom Lab, 2023)
```
Key features:
- Uses evolutionary sequence model (EVE)
- Fitness = binding + stability + expression
- Linear combination of escape scores

PRISM-4D differentiation:
- DOES NOT use pre-trained language models
- DOES use reinforcement learning to LEARN weights
- DOES incorporate temporal dynamics (Cycle Module)
- DOES use neuromorphic computing for feature extraction
- 19,400x faster execution
```

### VASIL (Validation framework)
```
Key features:
- Hardcoded formula: gamma = 0.65*escape + 0.35*transmit
- Linear regression baseline
- Per-country evaluation

PRISM-4D differentiation:
- NO hardcoded weights - RL learns optimal strategy
- Neuromorphic feature extraction
- GPU-accelerated batch processing
- Temporal cycle phase prediction
```

### Claim Language to Distinguish
```
"...wherein weights for combining features are learned by the 
reinforcement learning agent rather than predetermined..."

"...wherein the feature extraction comprises neuromorphic 
processing using spiking dynamics..."

"...wherein all feature extraction stages execute within a 
single GPU kernel without intermediate memory transfers..."
```

## Specification Requirements

### Enablement (35 U.S.C. 112(a))
Must describe invention so person of ordinary skill can make and use it.

**Required Details:**
- GPU kernel architecture (stages, shared memory)
- Q-learning hyperparameters (alpha, epsilon, gamma)
- Feature vector dimensions and contents
- Training procedure and convergence criteria
- Specific protein structures used (PDB IDs)

### Written Description
Must show inventor possessed the invention.

**Include:**
- Actual experimental results (accuracy numbers)
- Specific code implementations
- Hardware tested (RTX 3060 specifications)
- Dataset details (VASIL 12 countries)

### Best Mode
Must disclose best mode known to inventor.

**Disclose:**
- Optimal hyperparameters found
- Recommended GPU configuration
- Preferred feature combination weights

## Responding to 101 Rejections

### Common Examiner Arguments
```
"Claims recite mathematical concepts..."
"Claims can be performed mentally..."
"Claims recite abstract data processing..."
```

### Response Strategies

**Strategy 1: Technical Improvement**
```
"Claims recite a specific technical improvement to GPU 
computing by fusing multiple stages into a single kernel 
launch, reducing memory bandwidth by 85% compared to 
conventional staged approaches. See MPEP 2106.05(a)."
```

**Strategy 2: Particular Machine**
```
"Claims are tied to a particular machine - a GPU executing 
a mega-fused kernel with specific shared memory layout. 
The claims cannot be performed mentally or on general-purpose 
hardware without the claimed architecture."
```

**Strategy 3: Practical Application**
```
"Claims integrate the judicial exception into a practical 
application of predicting viral variant emergence, enabling 
public health responses. The claims do not merely recite 
the abstract idea but apply it to a specific real-world 
problem. See USPTO Example 49 (medical treatment)."
```

**Strategy 4: Unconventional Combination**
```
"Claims recite an unconventional ordered combination of 
neuromorphic computing and reinforcement learning that 
has not been previously applied to viral prediction. 
This combination produces results that could not be 
achieved by either approach alone."
```

## Abstract Guidelines

### Format
- Maximum 150 words
- Single paragraph
- No legal phrases ("comprising," "wherein")
- Technical summary of invention

### Example
```
A system and method for predicting viral variant emergence 
using a combination of neuromorphic computing and reinforcement 
learning. Structural features of viral proteins are extracted 
using a GPU-accelerated pipeline that computes contact networks, 
topological invariants, and spiking neural dynamics within a 
single kernel execution. A Q-learning agent processes the 
extracted features to predict whether a variant will increase 
or decrease in population frequency. The system achieves 
prediction accuracy exceeding 92% while processing over 300 
structures per second on consumer-grade hardware, enabling 
real-time surveillance of viral evolution.
```

## References
- See `references/patent_claim_structure.md` for detailed claim drafting
- See `references/prior_art_matrix.md` for comprehensive prior art analysis
