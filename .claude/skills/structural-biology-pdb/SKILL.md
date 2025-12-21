---
name: structural-biology-pdb
description: "PDB file format and SARS-CoV-2 spike protein structure guide. Use when: (1) Parsing PDB files with pdb_parser.rs or Biopython, (2) Applying mutations to spike RBD coordinates, (3) Extracting CA atoms for contact network analysis, (4) Working with 6M0J structure (ACE2-RBD complex), (5) Validating residue numbering and chain identifiers. Encodes PDB format specification, atom naming conventions, and RBD mutation hotspots (501, 484, 417)."
---

# Structural Biology/PDB Skill

## Purpose
Provide authoritative reference for PDB file parsing and SARS-CoV-2 spike RBD structure manipulation to prevent silent errors in coordinate transformations and mutation application.

## When to Use This Skill
- Implementing or debugging `pdb_parser.rs`
- Applying mutations to 6M0J or other spike structures
- Extracting alpha-carbon (CA) atoms for contact networks
- Validating residue numbering between GISAID lineages and PDB coordinates
- Understanding burial/surface exposure calculations

## PDB File Format Quick Reference

### ATOM Record (Columns 1-80)
```
Columns   Content                 Example
-------   ----------------------  ---------
1-6       Record name             "ATOM  "
7-11      Atom serial number      1234
13-16     Atom name               " CA " or " N  "
17        Alternate location      A, B, or blank
18-20     Residue name            "ALA", "GLY"
22        Chain identifier        A, B, E
23-26     Residue sequence number 501
27        Insertion code          blank or A
31-38     X coordinate (Angstrom) -12.345
39-46     Y coordinate (Angstrom)  23.456
47-54     Z coordinate (Angstrom)   7.890
55-60     Occupancy               1.00
61-66     Temperature factor (B)   25.50
77-78     Element symbol          "C ", "N "
79-80     Charge                  blank or 2+
```

### Critical Atom Naming Rules
```
Atom Name Format: Columns 13-16 (4 characters)
- Column 13-14: Element symbol (right-justified)
- Column 15-16: Remoteness + branch (left-justified)

Examples:
" CA "  - Alpha carbon (column 13 blank, C in 14, A in 15)
" N  "  - Backbone nitrogen
" C  "  - Backbone carbonyl carbon
" O  "  - Backbone oxygen
" CB "  - Beta carbon
" CG "  - Gamma carbon
" CD1"  - Delta 1 carbon
" NE "  - Epsilon nitrogen (Arg)
" OG1"  - Gamma oxygen 1 (Thr)
```

### Standard Amino Acid Codes
```
3-Letter  1-Letter  Full Name
ALA       A         Alanine
ARG       R         Arginine
ASN       N         Asparagine
ASP       D         Aspartic acid
CYS       C         Cysteine
GLN       Q         Glutamine
GLU       E         Glutamic acid
GLY       G         Glycine
HIS       H         Histidine
ILE       I         Isoleucine
LEU       L         Leucine
LYS       K         Lysine
MET       M         Methionine
PHE       F         Phenylalanine
PRO       P         Proline
SER       S         Serine
THR       T         Threonine
TRP       W         Tryptophan
TYR       Y         Tyrosine
VAL       V         Valine
```

## 6M0J Structure Reference

### Overview
- **PDB ID**: 6M0J
- **Title**: Crystal structure of SARS-CoV-2 spike RBD bound with ACE2
- **Resolution**: 2.45 Angstrom
- **Chains**: 
  - Chain A: ACE2 (residues 19-615)
  - Chain E: Spike RBD (residues 333-526)

### Chain E (RBD) Key Regions
```
Region                Residues    Function
-------------------   ----------  ----------------------------------
N-terminal           333-348      Connects to S1
Receptor Binding     417-505      Direct ACE2 contact
  Receptor Binding   438-506      Most critical for ACE2 binding
  Motif (RBM)
C-terminal           518-526      Connects to SD1

Key Interface Residues (within 4A of ACE2):
K417, G446, Y449, Y453, L455, F456, A475, F486, N487, Y489, Q493, G496, Q498, T500, N501, G502, Y505
```

### Coordinate System
- Origin: Crystallographic origin
- Units: Angstroms (1 A = 0.1 nm = 10^-10 m)
- Standard orientation: X, Y, Z orthogonal Cartesian

## Mutation Hotspots for PRISM-4D

### Critical RBD Mutations (VASIL Dataset)
```
Position  WT   Variants   Effect                    ΔΔG_binding
--------------------------------------------------------------
417       K    N, T       Immune escape             +0.4 kcal/mol
446       G    S          Minor                     -0.1 kcal/mol
452       L    R, Q       ACE2 affinity + escape    -0.3 kcal/mol
478       T    K          Modest effect             +0.1 kcal/mol
484       E    K, Q, A    Strong immune escape      -0.2 kcal/mol
490       F    S          ACE2 affinity             -0.5 kcal/mol
493       Q    R          ACE2 affinity             -0.4 kcal/mol
498       Q    R          ACE2 affinity (epistatic) -0.3 kcal/mol
501       N    Y, T       10x ACE2 affinity         -1.2 kcal/mol
505       Y    H          Modest effect             +0.2 kcal/mol
```

### Variant Mutation Profiles
```
Variant     RBD Mutations
---------   ------------------------------------------
Alpha       N501Y
Beta        K417N, E484K, N501Y
Gamma       K417T, E484K, N501Y
Delta       L452R, T478K
Omicron     G339D, S371L, S373P, S375F, K417N, N440K,
BA.1        G446S, S477N, T478K, E484A, Q493R, G496S,
            Q498R, N501Y, Y505H
```

## Parsing Patterns

### Rust PDB Parser (pdb_parser.rs)
```rust
/// Parse ATOM record line
pub fn parse_atom_line(line: &str) -> Option<Atom> {
    if !line.starts_with("ATOM  ") && !line.starts_with("HETATM") {
        return None;
    }
    
    // Fixed-width parsing (PDB is NOT whitespace-delimited!)
    let serial: u32 = line[6..11].trim().parse().ok()?;
    let name = line[12..16].to_string();  // Keep spaces!
    let alt_loc = line.chars().nth(16)?;
    let res_name = line[17..20].to_string();
    let chain = line.chars().nth(21)?;
    let res_seq: i32 = line[22..26].trim().parse().ok()?;
    let i_code = line.chars().nth(26)?;
    
    let x: f32 = line[30..38].trim().parse().ok()?;
    let y: f32 = line[38..46].trim().parse().ok()?;
    let z: f32 = line[46..54].trim().parse().ok()?;
    
    let occupancy: f32 = line.get(54..60)
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1.0);
    let b_factor: f32 = line.get(60..66)
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0.0);
    
    Some(Atom {
        serial, name, alt_loc, res_name, chain,
        res_seq, i_code, x, y, z, occupancy, b_factor,
    })
}

/// Extract CA atoms for contact network
pub fn extract_ca_atoms(atoms: &[Atom], chain: char) -> Vec<&Atom> {
    atoms.iter()
        .filter(|a| a.chain == chain && a.name.trim() == "CA")
        .collect()
}
```

### Common Parsing Errors
```
Error                          Cause                          Fix
---------------------------   ---------------------------    -----------------------
Missing atoms                 Whitespace-delimited parse     Use fixed-width columns
Wrong residue number          String vs int comparison       Parse as i32, not String
Chain mismatch               Case sensitivity               Use exact char match
Missing B-factors            Empty columns 60-66            Default to 0.0
Negative coordinates         Minus sign at column 31        Check for '-' in parsing
```

## Mutation Application

### Applying Point Mutation
```rust
/// Apply mutation to structure
pub fn apply_mutation(
    atoms: &mut [Atom],
    chain: char,
    res_seq: i32,
    new_res: &str,  // 3-letter code
) -> Result<()> {
    // Find all atoms of target residue
    let target_atoms: Vec<_> = atoms.iter_mut()
        .filter(|a| a.chain == chain && a.res_seq == res_seq)
        .collect();
    
    if target_atoms.is_empty() {
        return Err(anyhow!("Residue {} not found in chain {}", res_seq, chain));
    }
    
    // Update residue name for backbone atoms (N, CA, C, O)
    // Side chain atoms may need to be regenerated
    for atom in target_atoms {
        if ["N", "CA", "C", "O"].contains(&atom.name.trim()) {
            atom.res_name = new_res.to_string();
        }
    }
    
    Ok(())
}
```

### GISAID to PDB Residue Mapping
```
GISAID uses Spike numbering (1-1273 for full spike)
PDB 6M0J uses crystallographic numbering (333-526 for RBD)

To convert: PDB_resnum = GISAID_resnum (for RBD, they match!)

Exception: Some structures use different numbering schemes.
Always verify against SEQRES records.
```

## Distance Calculations

### Contact Distance (12 Angstrom cutoff for PRISM-4D)
```rust
/// Euclidean distance between two atoms
pub fn distance(a: &Atom, b: &Atom) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    (dx*dx + dy*dy + dz*dz).sqrt()
}

/// Build contact matrix for CA atoms
pub fn contact_matrix(ca_atoms: &[Atom], cutoff: f32) -> Vec<Vec<bool>> {
    let n = ca_atoms.len();
    let mut matrix = vec![vec![false; n]; n];
    
    for i in 0..n {
        for j in i+1..n {
            if distance(&ca_atoms[i], &ca_atoms[j]) < cutoff {
                matrix[i][j] = true;
                matrix[j][i] = true;
            }
        }
    }
    matrix
}
```

### Burial Calculation (SASA-based)
```rust
/// Simplified burial score based on neighbor count
/// More neighbors = more buried
pub fn burial_score(ca_atoms: &[Atom], target_idx: usize, radius: f32) -> f32 {
    let target = &ca_atoms[target_idx];
    let neighbors = ca_atoms.iter()
        .filter(|a| distance(a, target) < radius)
        .count() - 1;  // Exclude self
    
    // Normalize: 0 = surface (0-5 neighbors), 1 = buried (20+ neighbors)
    (neighbors as f32 / 20.0).min(1.0)
}
```

## References
- See `references/pdb_format.md` for complete format specification
- See `references/spike_rbd_mutations.md` for comprehensive mutation effects
- See `assets/6m0j_rbd_annotated.txt` for annotated RBD coordinates
