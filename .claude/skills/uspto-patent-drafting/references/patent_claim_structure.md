# Patent Claim Structure Reference

## Claim Anatomy

### Independent Claims
```
[Preamble] + [Transitional Phrase] + [Body]

Preamble: "A method for predicting viral evolution"
Transitional: "comprising" (open-ended) or "consisting of" (closed)
Body: Steps or elements that define the invention
```

### Transitional Phrases
| Phrase | Scope | Use When |
|--------|-------|----------|
| comprising | Open | Additional elements allowed |
| consisting of | Closed | ONLY recited elements |
| consisting essentially of | Partially open | Core elements fixed, non-material additions OK |
| including | Open | Same as comprising |
| having | Ambiguous | Avoid - courts interpret inconsistently |

### Antecedent Basis
```
First mention: "a processor" (indefinite article)
Subsequent: "the processor" (definite article)

Correct:
"receiving, by a processor, data..."
"processing, by the processor, the data..."

Incorrect:
"receiving, by the processor, data..." (no antecedent)
"processing, by a processor, the data..." (new element?)
```

## Claim Types for PRISM-4D

### Method Claims (35 U.S.C. 101 - Process)
```
Strongest for software/AI inventions
Focus on STEPS performed
Use active verbs: receiving, computing, determining, outputting

Structure:
1. A method for [goal], comprising:
   (a) [first step];
   (b) [second step]; and
   (c) [final step].
```

### System Claims (35 U.S.C. 101 - Machine)
```
Recite COMPONENTS and their configuration
Provides infringement options (making/using/selling)

Structure:
1. A system for [goal], comprising:
   a [first component] configured to [function];
   a [second component] coupled to the [first component] and
     configured to [function]; and
   a [third component] configured to [function].
```

### Apparatus Claims
```
Similar to system but focuses on structure
Good for hardware innovations

Structure:
1. An apparatus comprising:
   a graphics processing unit having [specific architecture];
   a memory coupled to the graphics processing unit; and
   a processor configured to [function].
```

### Computer-Readable Medium Claims
```
Catches software distribution
Must be "non-transitory" (excludes signals)

Structure:
1. A non-transitory computer-readable medium storing 
   instructions that, when executed by a processor, cause 
   the processor to perform operations comprising:
   [steps matching method claim]
```

## PRISM-4D Claim Set Strategy

### Recommended Claim Distribution
```
Claims 1-9:    Method claims (broadest + narrowing dependents)
Claims 10-18:  System claims (parallel structure)
Claims 19-20:  CRM claims (for software sales)
```

### Independent Claim Variations

**Claim 1: Broadest Method**
```
Focus: Overall pipeline
Key elements:
- GPU feature extraction
- Neuromorphic processing
- RL prediction

Avoid:
- Specific hyperparameters
- Exact feature dimensions
- Particular viral proteins
```

**Claim 10: System with Hardware**
```
Focus: Physical implementation
Key elements:
- GPU specification
- Memory architecture
- Processor configuration

Include:
- Means for receiving input
- Means for outputting prediction
```

**Claim 19: CRM for Software**
```
Focus: Distributed software
Mirror claim 1 steps
"Non-transitory" required post-2014
```

### Dependent Claim Strategy

**Fallback Position 1: Specific Hardware**
```
Claims 2-4: Narrow to GPU type, memory, architecture
If broad claims rejected, these survive
```

**Fallback Position 2: Specific Algorithm**
```
Claims 5-7: Narrow to neuromorphic specifics
Dendritic reservoir, spiking dynamics, LIF model
```

**Fallback Position 3: Specific Application**
```
Claims 8-9: Narrow to coronavirus, specific mutations
Application-specific claims harder to design around
```

## Means-Plus-Function Claims

### When to Use
```
When functional language is clearer than structural
35 U.S.C. 112(f) interpretation

Format: "means for [function]" or "[nonce word] for [function]"
Nonce words: module, unit, element, device
```

### Requirements
```
Specification MUST disclose:
1. Corresponding structure
2. Material
3. Acts

Example:
Claim: "a feature extraction module for computing..."
Spec must describe: GPU kernel with specific stages
```

### Risks
```
- Narrower scope (limited to disclosed structure + equivalents)
- Indefiniteness if structure not disclosed
- Generally avoid for software claims
```

## Functional Claim Language

### Acceptable (Definite)
```
"configured to [function]"
"adapted to [function]"
"operable to [function]"

These require structural capability, not just intended use
```

### Problematic (May Be Indefinite)
```
"for [function]" alone - may not require structure
"capable of [function]" - too broad
"wherein [function occurs]" - may be result, not limitation
```

### Best Practices
```
Combine functional and structural:
"a processor configured to execute instructions for 
computing a contact network, wherein the contact network 
comprises edges between residues within 12 Angstroms"

This adds:
1. Structural element (processor)
2. Functional requirement (configured to)
3. Specific limitation (12 Angstroms)
```

## Jepson Claims

### Format
```
"In a [known base], the improvement comprising:
[novel elements]"
```

### When to Use
- Clear improvement over prior art
- Base system well-known
- Want to highlight novelty

### Example for PRISM-4D
```
In a system for viral evolution prediction having a processor 
and memory, the improvement comprising:

a graphics processing unit configured to execute a mega-fused 
kernel combining contact network analysis, topological data 
analysis, and dendritic reservoir processing in a single 
kernel launch.
```

## Markush Groups

### Format
```
"selected from the group consisting of A, B, and C"
```

### Use Cases
```
Mutation positions:
"wherein the position is selected from the group consisting 
of 417, 484, 490, 493, 498, and 501"

Feature types:
"wherein the feature extraction comprises at least one of 
topological features, neuromorphic features, and contact 
network features"
```

### "At least one of" Language
```
"at least one of A, B, and C" - ambiguous (A alone? A+B?)

Clearer alternatives:
"at least one of A, B, or C" - any single item
"at least one of A, B, and C, or combinations thereof"
"one or more of A, B, and C"
```

## Ranges and Values

### Open-Ended Ranges
```
"at least 12 Angstroms" - 12 and above
"no more than 256 states" - 256 and below
"between 0.1 and 0.3" - includes endpoints unless specified
```

### Specific Values with Flexibility
```
"approximately 12 Angstroms" - some tolerance
"about 256 states" - reasonable variation
"substantially 92% accuracy" - avoid, too vague
```

### Best Practice: Dependent Narrowing
```
Claim 1: "a distance threshold"
Claim 2: "wherein the distance threshold is between 8 and 15 Angstroms"
Claim 3: "wherein the distance threshold is 12 Angstroms"
```

## Omnibus Claims (Avoid)

### What They Are
```
"A device substantially as described herein"
"A method as shown in Figure 3"
```

### Why Problematic
- Indefinite under U.S. law
- Not allowed by USPTO
- Only acceptable in some foreign jurisdictions

## Product-by-Process Claims

### Format
```
"A trained neural network produced by the process of:
training on dataset X using algorithm Y..."
```

### Limitations
- Claim scope = product, not process
- Infringer need not use same process
- Must be impossible to define product structurally

### Alternative for PRISM-4D
```
Instead of: "A Q-table trained by Q-learning..."
Use: "A Q-table comprising 256 states and values reflecting 
learned action preferences for viral evolution prediction"
```
