# Nature Journal Formatting Reference

## Document Formatting

### General Requirements
- Double-spaced text
- 12-point Times New Roman or similar
- Left-aligned (not justified)
- Line numbers continuous throughout
- Page numbers in footer
- 2.5 cm (1 inch) margins

### File Format
- Microsoft Word (.docx) preferred
- PDF acceptable for initial submission
- LaTeX: convert to PDF, keep source for revision

### Title Page
```
Title (no abbreviations, max 90 characters)

Author names (no degrees)
Author1*, Author2, Author3†

Affiliations (numbered superscripts)
1 Department, Institution, City, Country.
2 Department, Institution, City, Country.

*Corresponding author. Email: author@institution.edu
†These authors contributed equally.

Keywords: keyword1, keyword2, keyword3 (4-6 keywords)
```

## Section-by-Section Guidelines

### Abstract (Nature Articles)
```
Length: ~150 words (unreferenced)
Structure:
- Background (1-2 sentences)
- Methods overview (1 sentence)  
- Key results (2-3 sentences)
- Conclusions (1-2 sentences)

Do NOT include:
- References
- Abbreviations (unless universally known)
- Specific statistical values
```

### Summary Paragraph (Nature Letters/older format)
```
Length: ~200 words (fully referenced)
Must include:
- References to key prior work
- "Here we show/demonstrate/report..."
- Specific quantitative findings
- Statement of significance
```

### Introduction
```
Length: ~500 words for Articles
No heading (flows directly after abstract)

Paragraph 1: Broad context, why this matters
Paragraph 2: Current state of knowledge, cite prior work
Paragraph 3: Gap/limitation in current approaches
Paragraph 4: Brief statement of your approach and main finding
```

### Results
```
Structure: Subheadings required
Each subsection: 
- Opening statement of aim
- Methods summary (brief)
- Findings with figure/table references
- Interpretation (brief)

Figure references: (Fig. 1a), (Extended Data Fig. 1)
Table references: (Table 1), (Supplementary Table 1)
Statistics: (P < 0.001, two-tailed t-test, n = 50)
```

### Discussion
```
Length: ~1000 words
No subheadings typically

Paragraph 1: Summary of main findings
Paragraph 2: Comparison with prior work
Paragraph 3: Mechanistic insights
Paragraph 4: Limitations and caveats
Paragraph 5: Broader implications
Paragraph 6: Future directions
```

### Methods
```
Location: After main text, appears online
Length: No strict limit (sufficient for replication)

Required subsections:
- Study design/overview
- Data sources
- [Specific methods - e.g., "GPU kernel implementation"]
- Statistical analysis
- Reporting summary

End statements (required):
- Data availability
- Code availability  
- Acknowledgements
- Author contributions
- Competing interests
- Additional information (correspondence, reprints)
```

## Extended Data and Supplementary Information

### Extended Data
- Up to 10 figures/tables
- Peer-reviewed
- Integral to main conclusions
- Cited as "Extended Data Fig. X"

### Supplementary Information
- Not peer-reviewed in detail
- Supporting material
- Methods details, additional data
- Cited as "Supplementary Table X"

## Statistical Reporting

### Required Information
```
For each statistical test:
- Test name (e.g., "two-tailed Mann-Whitney U test")
- Test statistic value
- Exact P value (or P < 0.0001 if very small)
- Sample sizes (n = X)
- Definition of center and dispersion
```

### Examples
```
Correct:
"PRISM-4D achieved 92.3% accuracy (95% CI: 91.1-93.5%, 
n = 14,917 lineages across 5 independent training runs)."

"Accuracy was significantly higher than baseline 
(P = 0.0023, two-tailed paired t-test, n = 12 countries)."

Incorrect:
"PRISM-4D was significantly better (P < 0.05)."
```

### Effect Sizes
Report effect sizes with confidence intervals:
```
"PRISM-4D improved accuracy by 8.3 percentage points 
(95% CI: 6.1-10.5 pp) compared to VASIL baseline."
```

## Computational Methods Checklist

### For Machine Learning Papers
- [ ] Training/validation/test split described
- [ ] Hyperparameter selection method
- [ ] Number of training runs for variance
- [ ] Hardware specifications
- [ ] Runtime/computational cost
- [ ] Random seeds for reproducibility

### For Bioinformatics Papers
- [ ] Sequence/structure database versions
- [ ] Alignment parameters
- [ ] Software versions
- [ ] Thresholds and cutoffs justified

## Nature Methods Specific

### Focus Areas
- New methods with broad applicability
- Significant improvements over existing methods
- Thorough benchmarking required

### Additional Requirements
- Protocol section (step-by-step)
- Anticipated results section
- Troubleshooting guide
- Timing information

## Review Process Timeline

```
Initial decision: 1-2 weeks
  |
  v (if sent for review)
Peer review: 4-8 weeks
  |
  v
Decision letter
  |
  v (if revision invited)
Revision submission: typically 3-6 months allowed
  |
  v
Re-review: 2-4 weeks
  |
  v
Final decision
  |
  v (if accepted)
Production: 4-6 weeks
```

## Cover Letter Template

```
Dear Editor,

We submit our manuscript entitled "[Title]" for consideration 
as a [Article/Letter] in Nature.

[Paragraph 1: Why this work is important - 2-3 sentences]
Predicting viral evolution is critical for pandemic preparedness...

[Paragraph 2: Key findings - 3-4 sentences]
We developed PRISM-4D, a neuromorphic computing platform that 
achieves >92% accuracy in predicting variant emergence...

[Paragraph 3: Why Nature - 2 sentences]
This work represents a significant advance in computational 
virology with immediate applications to pandemic surveillance...

[Paragraph 4: Practical matters]
We confirm this work has not been published elsewhere and is 
not under consideration at another journal. All authors have 
approved the manuscript. We declare no competing interests.

[Suggested reviewers - optional]
We suggest the following potential reviewers:
- Dr. X (expertise in viral evolution)
- Dr. Y (expertise in machine learning for biology)

We request exclusion of Dr. Z due to direct competition.

Sincerely,
[Corresponding author]
```
