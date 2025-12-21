# PDB File Format Complete Reference

## Version: wwPDB Format 3.3 (Current Standard)

## File Structure Overview

```
HEADER    descriptive information
TITLE     title of the experiment
COMPND    compound information (multiple lines)
SOURCE    biological source
KEYWDS    keywords
EXPDTA    experimental technique
AUTHOR    author names
REVDAT    revision history
JRNL      journal citation
REMARK    various annotations (numbered)
DBREF     database cross-references
SEQRES    primary sequence
HELIX     helix definitions
SHEET     sheet definitions
SSBOND    disulfide bonds
CRYST1    unit cell parameters
ORIGXn    coordinate transformation
SCALEn    crystallographic scaling
ATOM      coordinate records
HETATM    heteroatom coordinates
TER       chain termination
CONECT    connectivity
MASTER    record count summary
END       end of file
```

## ATOM Record Detailed Specification

```
COLUMNS        DATA TYPE      CONTENTS
------------------------------------------------------------------------
1 -  6        Record name    "ATOM  "
7 - 11        Integer        Atom serial number (right-justified)
12            Character      Blank
13 - 16       Atom           Atom name (see naming rules below)
17            Character      Alternate location indicator
18 - 20       Residue name   Residue name
21            Character      Blank
22            Character      Chain identifier
23 - 26       Integer        Residue sequence number (right-justified)
27            AChar          Code for insertions of residues
28 - 30       Character      Blank
31 - 38       Real(8.3)      Orthogonal X coordinate (Angstroms)
39 - 46       Real(8.3)      Orthogonal Y coordinate (Angstroms)
47 - 54       Real(8.3)      Orthogonal Z coordinate (Angstroms)
55 - 60       Real(6.2)      Occupancy
61 - 66       Real(6.2)      Temperature factor (B-factor)
67 - 76       Character      Blank (segment identifier in older formats)
77 - 78       LString(2)     Element symbol (right-justified)
79 - 80       LString(2)     Charge on the atom
```

## Atom Naming Convention

### Standard Backbone Atoms
```
Atom   Columns 13-16   Element   Description
-----  --------------  --------  ---------------------------
N      " N  "          N         Backbone nitrogen
CA     " CA "          C         Alpha carbon
C      " C  "          C         Backbone carbonyl carbon
O      " O  "          O         Backbone oxygen
OXT    " OXT"          O         Terminal oxygen (C-terminus only)
```

### Greek Letter Naming (Remoteness Indicator)
```
Position   Greek   Atoms      Example Residue
--------   -----   --------   ---------------
Alpha      A       CA         All residues
Beta       B       CB         All except Gly
Gamma      G       CG, OG     Val, Thr, Ser
Delta      D       CD, OD     Pro, Arg, Lys
Epsilon    E       CE, NE     Lys, Arg, Met
Zeta       Z       CZ, NZ     Lys, Arg, Phe
Eta        H       OH, NH     Tyr, Arg
```

### Branch Numbering
```
When atoms branch, add number 1, 2:
CB           Single beta carbon (Ala)
CG           Single gamma carbon (Pro)
CG1, CG2     Branched gamma carbons (Val, Ile)
CD1, CD2     Branched delta carbons (Leu, Phe, Tyr)
OD1, OD2     Carboxyl oxygens (Asp)
OE1, OE2     Carboxyl oxygens (Glu)
```

### Hydrogen Naming (v3.0+)
```
Standard pattern: H + remoteness + branch + number
Examples:
H            Backbone amide hydrogen
HA           Alpha hydrogen
HB2, HB3     Beta hydrogens (numbered)
HG           Gamma hydrogen
HD21, HD22   Delta 2 branch hydrogens
```

## HETATM Records

Same format as ATOM, used for:
- Water molecules (HOH, WAT)
- Ligands and cofactors
- Metal ions
- Modified amino acids
- Non-standard residues

```
Example:
HETATM 1071  FE  HEM A   1       8.128   7.371 -15.022  1.00 16.74          FE
HETATM 1107  O   HOH A 201      12.345   6.789  -8.901  1.00 35.67           O
```

## TER Record

Marks end of a polymer chain:
```
COLUMNS        DATA TYPE      CONTENTS
1 -  6         Record name    "TER   "
7 - 11         Integer        Serial number (one greater than last ATOM)
18 - 20        Residue name   Residue name of last residue
22             Character      Chain identifier
23 - 26        Integer        Residue sequence number
27             AChar          Insertion code
```

## SEQRES Records

Primary sequence information:
```
COLUMNS        DATA TYPE      CONTENTS
1 -  6         Record name    "SEQRES"
8 - 10         Integer        Serial number of the SEQRES record
12             Character      Chain identifier
14 - 17        Integer        Number of residues in the chain
20 - 22        Residue name   Residue 1
24 - 26        Residue name   Residue 2
... (continues with 13 residues per record)
```

Example:
```
SEQRES   1 E  194  SER THR ILE ALA ASN ALA SER ASN PRO TRP ASN ALA THR
SEQRES   2 E  194  GLU TYR LEU VAL LYS TYR VAL GLU ASN VAL LYS PHE LYS
```

## REMARK Records

Common remark types for PRISM-4D:
```
REMARK   2   RESOLUTION    2.45 ANGSTROMS
REMARK   3   REFINEMENT (R-factors, geometry)
REMARK 200   EXPERIMENTAL DETAILS
REMARK 350   BIOMOLECULE (biological assembly)
REMARK 465   MISSING RESIDUES
REMARK 470   MISSING ATOMS
REMARK 500   GEOMETRY WARNINGS
REMARK 800   SITE DESCRIPTION
```

## Alternate Conformations

When atoms have multiple conformations:
```
ATOM    102  CA AALA A  14      10.123  20.456  30.789  0.60 15.00           C
ATOM    103  CA BALA A  14      10.234  20.567  30.890  0.40 18.00           C
```
- Column 17: A, B, C... for conformers
- Occupancies should sum to 1.0
- For PRISM-4D: typically use conformation A only

## Insertion Codes

Handle sequence insertions without renumbering:
```
ATOM    500  CA  ALA A  50      ...
ATOM    507  CA  GLY A  50A     ...   <- Inserted residue
ATOM    514  CA  VAL A  51      ...
```
- Column 27: A, B, C... for insertions
- Preserves original numbering scheme

## Coordinate Transformation Records

### CRYST1 (Crystal Parameters)
```
COLUMNS        DATA TYPE      CONTENTS
1 -  6         Record name    "CRYST1"
7 - 15         Real(9.3)      a (Angstroms)
16 - 24        Real(9.3)      b (Angstroms)
25 - 33        Real(9.3)      c (Angstroms)
34 - 40        Real(7.2)      alpha (degrees)
41 - 47        Real(7.2)      beta (degrees)
48 - 54        Real(7.2)      gamma (degrees)
56 - 66        LString        Space group
67 - 70        Integer        Z value
```

### SCALEn (Fractional Transformation)
```
SCALE1    0.018293  0.000000  0.000000        0.00000
SCALE2    0.000000  0.020576  0.000000        0.00000
SCALE3    0.000000  0.000000  0.009174        0.00000
```
Converts orthogonal Angstrom coordinates to fractional crystallographic coordinates.

## Common File Issues and Solutions

### Issue 1: Non-standard Line Endings
```python
# Fix: Normalize line endings
content = content.replace('\r\n', '\n').replace('\r', '\n')
```

### Issue 2: Missing Columns
```python
# Fix: Pad short lines
if len(line) < 80:
    line = line.ljust(80)
```

### Issue 3: Multiple Models (NMR structures)
```
MODEL        1
ATOM    ...
ENDMDL
MODEL        2
ATOM    ...
ENDMDL
```
For PRISM-4D: Use MODEL 1 only (lowest energy conformer)

### Issue 4: Hybrid-36 Encoding (Large Structures)
For serial numbers > 99999 or residue numbers > 9999:
```
Standard:  1 -  99999  (digits)
Hybrid36:  A0000 - ZZZZZ (alphanumeric)

Decoding: 
  A0000 = 100000
  A0001 = 100001
  ...
```

## Validation Checklist

Before using PDB file in PRISM-4D:
- [ ] All ATOM records have valid coordinates (not 9999.999)
- [ ] Chain E (RBD) residues 333-526 present
- [ ] No gaps in CA atoms for RBD
- [ ] Occupancy > 0.5 for all atoms used
- [ ] B-factors not all identical (indicates refinement issue)
- [ ] TER records properly placed between chains
- [ ] No HETATM water molecules near active site residues
