# Figure Preparation Guidelines for Scientific Publication

## Universal Requirements

### Resolution Standards
```
Output Type          Minimum DPI    Recommended
-------------------------------------------------
Line art (graphs)    1000           1200
Halftone (photos)    300            600
Combination          500            600
Web only             150            300
```

### Color Specifications
```
Mode: RGB for submission (publisher converts to CMYK)
Color depth: 8-bit per channel minimum
Color space: sRGB (standard)

Accessible color palettes:
  - Blue (#0072B2) + Orange (#E69F00) + Green (#009E73)
  - Avoid red-green only distinctions
  - Include patterns/shapes for colorblind readers
```

## Dimension Guidelines

### Nature/Science Journals
```
Width Options:
  Single column:  89 mm  (3.50 in, ~1063 px at 300 DPI)
  1.5 column:    120 mm  (4.72 in, ~1417 px at 300 DPI)
  Double column: 183 mm  (7.20 in, ~2161 px at 300 DPI)

Maximum height: 247 mm (9.72 in)

Panel labels: 8 pt Arial, bold, lowercase (a, b, c, d)
Axis labels: 6-8 pt Arial/Helvetica
Tick labels: 6 pt minimum
```

### bioRxiv/Preprints
```
More flexible, but recommend:
  Width: 170 mm (6.7 in) for readability
  Height: Up to full page
  Resolution: 300 DPI minimum
```

### Conference Posters
```
Standard sizes:
  - 36 x 48 inches (landscape)
  - 42 x 36 inches (portrait)
Resolution: 150 DPI sufficient at viewing distance
```

## Figure Types for PRISM-4D

### Type 1: Architecture Diagram
```
Purpose: Show system overview
Format: Vector (SVG/PDF)
Elements:
  - GPU kernel stages (boxes)
  - Data flow (arrows)
  - Feature dimensions (labels)
  
Style recommendations:
  - Use consistent box sizes
  - Color-code by module type
  - Include dimension annotations
  - Show batch processing flow
```

### Type 2: Benchmark Bar Charts
```
Purpose: Compare accuracy across methods/countries
Format: Vector or 600 DPI raster

Required elements:
  - Clear axis labels with units
  - Error bars (specify type in legend)
  - Baseline reference line if applicable
  - Individual data points if n < 20
  - Statistical significance markers (*, **, ***)

Color scheme:
  - PRISM-4D: Primary color (e.g., blue)
  - Baselines: Neutral colors (gray shades)
  - Highlight: Accent color for key comparisons
```

### Type 3: Training Curves
```
Purpose: Show convergence dynamics
Format: Vector or 600 DPI raster

Required elements:
  - X-axis: Epoch or iteration
  - Y-axis: Metric (accuracy, loss)
  - Shaded region for variance (if multiple runs)
  - Horizontal line for target/baseline
  - Vertical line for early stopping if used

Recommendations:
  - Log scale for loss if spans orders of magnitude
  - Include both train and validation curves
  - Smooth noisy curves with moving average (state n-point)
```

### Type 4: Confusion Matrix / Heatmap
```
Purpose: Show classification performance
Format: Raster at 600 DPI

Required elements:
  - Row/column labels (clear, readable)
  - Color scale bar with values
  - Cell values (counts or percentages)
  - Title specifying what is shown

Color scheme:
  - Sequential: white to dark blue for counts
  - Diverging: blue-white-red for correlations
  - Ensure sufficient contrast for text overlay
```

### Type 5: Q-Table Visualization
```
Purpose: Show learned RL policy
Format: Heatmap or parallel coordinates

Required elements:
  - State dimensions labeled
  - Action preferences indicated
  - Colorbar for Q-values
  - Annotation of key states

Special considerations:
  - 256 states may need aggregation for clarity
  - Show action preference (Q[a1] - Q[a2])
  - Highlight states with strong preferences
```

## Panel Layout

### Multi-Panel Figures
```
Spacing:
  - Between panels: 5-10 mm
  - Panel labels: 2 mm from panel edge
  - Consistent alignment (left edges, baselines)

Label placement:
  a +---------+    b +---------+
    |         |      |         |
    |  Plot   |      |  Plot   |
    +---------+      +---------+

Not:
  +---------+ a    +---------+ b
  |         |      |         |
```

### Aspect Ratios
```
Standard plots: 4:3 or 3:2
Wide comparisons: 16:9
Square heatmaps: 1:1
Tall time series: 2:3 or 1:2
```

## Typography in Figures

### Font Hierarchy
```
Title: 10-12 pt, bold
Axis labels: 8-10 pt, regular
Tick labels: 6-8 pt, regular
Annotations: 6-8 pt, regular or italic
Panel labels: 8-10 pt, bold, lowercase
```

### Mathematical Notation
```
Variables: italic (n, x, P)
Functions: roman (sin, log, max)
Units: roman with space (10 ms, 5 mM)
Greek: Symbol font or Unicode
Subscripts: baseline offset (x_i)
```

## Export Settings

### Adobe Illustrator
```
Save As: PDF or EPS
Settings:
  - Compatibility: Acrobat 5 (PDF 1.4)
  - Preserve Illustrator Editing
  - Embed fonts
  - High resolution (300+ DPI for rasters)
```

### Python Matplotlib
```python
import matplotlib.pyplot as plt

# Set up figure with exact dimensions
fig, ax = plt.subplots(figsize=(3.5, 2.5))  # inches

# Use publication-quality settings
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.5,
    'axes.labelsize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

# Save as vector and raster
fig.savefig('figure1.pdf', format='pdf')
fig.savefig('figure1.tiff', format='tiff', dpi=600)
```

### R ggplot2
```r
library(ggplot2)

# Set theme for publication
theme_publication <- theme_classic() +
  theme(
    text = element_text(family = "Arial", size = 8),
    axis.text = element_text(size = 6),
    axis.title = element_text(size = 8),
    legend.text = element_text(size = 6),
    panel.border = element_rect(fill = NA, linewidth = 0.5)
  )

# Save with specific dimensions
ggsave("figure1.pdf", width = 89, height = 60, units = "mm", dpi = 600)
```

## Accessibility Checklist

- [ ] Colors distinguishable without hue (use brightness/patterns)
- [ ] Alt text prepared for each figure
- [ ] Sufficient contrast (4.5:1 minimum)
- [ ] No information conveyed by color alone
- [ ] Text readable at printed size
- [ ] Legend does not rely on shape alone

## Common Mistakes to Avoid

| Mistake | Problem | Fix |
|---------|---------|-----|
| Low resolution | Pixelated when printed | Export at 600 DPI |
| Inconsistent fonts | Unprofessional appearance | Use template |
| No error bars | Cannot assess uncertainty | Add appropriate bars |
| Red-green scheme | Inaccessible to colorblind | Use blue-orange |
| Tiny labels | Unreadable at column width | 6 pt minimum |
| Missing units | Ambiguous data | Always include units |
| Cluttered legends | Hard to parse | Simplify, use panels |
| JPEG artifacts | Degraded line art | Use TIFF or PDF |
