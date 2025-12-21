---
name: scientific-publication
description: "Scientific manuscript formatting for Nature, bioRxiv, and high-impact venues. Use when: (1) Drafting Nature-format articles with 200-word summary paragraphs, (2) Preparing bioRxiv preprints, (3) Creating publication-quality figures (300 DPI, RGB), (4) Formatting references with DOIs, (5) Writing abstracts for computational biology papers. Encodes Nature submission guidelines, bioRxiv requirements, and figure specifications."
---

# Scientific Publication Skill

## Purpose
Provide authoritative formatting guidance for submitting PRISM-4D manuscripts to high-impact venues (Nature, Nature Methods, bioRxiv) and conference proceedings (NeurIPS, ISMB).

## When to Use This Skill
- Drafting manuscript text with proper structure
- Formatting figures for print and digital publication
- Preparing bioRxiv preprints for rapid dissemination
- Writing abstracts and summary paragraphs
- Formatting references and citations

## Nature Journal Requirements

### Article Types

| Type | Words | Figures | References | Structure |
|------|-------|---------|------------|-----------|
| Article | 3,500 | 6 max | ~50 | Abstract + sections |
| Letter (legacy) | 1,500 | 4 max | ~30 | Summary paragraph |
| Brief Communication | 1,500 | 2 max | ~20 | Minimal sections |

### Summary Paragraph (Nature Style ~200 words)
```
Structure:
- Sentence 1-2: Background context (what is the problem?)
- Sentence 3-4: Gap in knowledge (what is unknown?)
- Sentence 5: "Here we show that..." (main finding)
- Sentence 6-7: Key results with quantitative data
- Sentence 8-9: Broader implications and significance
```

### Main Text Structure
```
Introduction (no heading)
  - Context and significance
  - Gap in current knowledge
  - Brief statement of approach
  
Results
  - Subheadings allowed
  - Present findings with figure references
  - Quantitative statements with statistics
  
Discussion
  - Interpretation of results
  - Comparison with prior work
  - Limitations
  - Future directions
  
Methods (appears online)
  - Sufficient detail for replication
  - Statistical methods
  - Data availability
  - Code availability
```

### Writing Style Guidelines
- Active voice preferred ("We developed..." not "A method was developed...")
- Avoid jargon; define specialized terms at first use
- No abbreviations in title
- Spell out numbers one through ten
- Use SI units with spaces (10 ms, not 10ms)
- Present tense for established facts, past tense for your results

## bioRxiv Preprint Requirements

### Submission Format
- Single PDF or Word document (figures embedded or separate)
- LaTeX must be converted to PDF before submission
- Figure formats: JPEG, TIFF, EPS, PowerPoint
- NOT accepted: PICT, Bitmap, Excel, PSD

### Content Requirements
- Complete manuscript (not partial results)
- All authors must consent to posting
- Cannot be already published in a journal
- No clinical trial results (use medRxiv)

### Subject Categories for PRISM-4D
- Bioinformatics
- Computational Biology
- Evolutionary Biology
- Microbiology
- Systems Biology

### Timeline
- Initial screening: 24-48 hours
- DOI assigned upon posting
- Revisions can be submitted anytime before journal acceptance
- Cannot be removed once posted

## Figure Specifications

### Resolution Requirements
| Content Type | Minimum DPI | Recommended |
|--------------|-------------|-------------|
| Line art | 1000 | 1200 |
| Halftones (photos) | 300 | 600 |
| Combinations | 500 | 600 |

### Dimensions
```
Single column:  89 mm (3.5 inches)
1.5 column:    120 mm (4.7 inches)  
Double column: 183 mm (7.2 inches)
Maximum height: 247 mm (9.7 inches)
```

### Color Mode
- RGB for digital submission
- CMYK for print (Nature will convert)
- Avoid red-green contrast (colorblind accessibility)

### Font Requirements
- Sans-serif (Arial, Helvetica) for labels
- Minimum 6 pt, recommended 8 pt
- Consistent font size across all panels
- Greek symbols from Symbol font

### File Formats
- TIFF: Best for final submission
- EPS: Good for vector graphics
- PDF: Acceptable
- PNG: Acceptable for web figures
- JPEG: Acceptable but lossy

## Figure Legends

### Structure
```
Figure 1 | Title in bold, single sentence.
a, Description of panel a with sample sizes.
b, Description of panel b. Statistical test used,
   P values, n = X biological replicates.
Error bars represent mean +/- s.d. (or s.e.m.).
Scale bars: 10 um (if applicable).
```

### Example for PRISM-4D
```
Figure 2 | FluxNet RL achieves superior accuracy on VASIL benchmark.
a, Per-country test accuracy comparing PRISM-4D (blue) to VASIL 
   baseline (gray). Error bars show 95% CI across 5 training runs.
b, Confusion matrix for aggregated predictions (n = 14,917 lineages).
c, Training convergence showing mean accuracy over 300 epochs.
   Shaded region indicates standard deviation (n = 5 runs).
d, Q-table visualization showing learned state-action preferences.
   States discretized into 256 bins from 6 continuous features.
```

## Reference Formatting

### Nature Style
```
1. Author, A. N., Author, B. T. & Author, C. D. Title of article
   in sentence case. Journal Name Vol, pages (Year).

Example:
1. Starr, T. N. et al. Deep mutational scanning of SARS-CoV-2 
   receptor binding domain reveals constraints on folding and 
   ACE2 binding. Cell 182, 1295-1310 (2020).
```

### Rules
- Include DOI when available
- "et al." after first author if more than 5 authors
- Journal names abbreviated (Nature, not Nature Publishing Group)
- No issue numbers for continuously paginated journals
- Preprints: include "Preprint at" and server name

### bioRxiv Citation Format
```
Author, A. N. & Author, B. T. Title of preprint. bioRxiv 
https://doi.org/10.1101/2024.01.01.123456 (2024).
```

## Data and Code Availability

### Required Statements
```
Data availability
The VASIL benchmark dataset is available at [repository]. 
Structural data from PDB accession 6M0J. GISAID sequences 
under accession EPI_SET_XXXXX (requires GISAID registration).

Code availability
PRISM-4D source code is available at https://github.com/[repo]
under MIT license. Documentation at https://prism4d.readthedocs.io.
```

### Repository Options
- GitHub (code): Zenodo DOI for archival
- Figshare (data and figures)
- Dryad (biological data)
- GISAID (viral sequences)
- PDB (structures)

## PRISM-4D Specific Guidance

### Key Claims to Highlight
1. **Speed**: "19,400x faster than EVEscape"
2. **Accuracy**: ">92% mean accuracy on VASIL benchmark"
3. **Novelty**: "First integration of neuromorphic computing with RL for viral prediction"
4. **Generalizability**: "Validated across 12 countries"

### Figures to Include
1. Architecture diagram (FluxNet RL + GPU pipeline)
2. Benchmark comparison (accuracy by country)
3. Training dynamics (convergence curves)
4. Case study (specific variant prediction)
5. Computational performance (structures/second)

### Methods Sections Needed
- GPU kernel implementation details
- Q-learning hyperparameters
- Feature extraction pipeline
- VASIL dataset preprocessing
- Statistical analysis approach

## References
- See `references/nature_format.md` for detailed Nature guidelines
- See `references/figure_guidelines.md` for figure preparation
- See `assets/latex_template.tex` for manuscript template
