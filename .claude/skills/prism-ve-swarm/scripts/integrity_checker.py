#!/usr/bin/env python3
"""
PRISM-4D Integrity Checker
Scans codebase for scientific integrity violations.
"""

import sys
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class Violation:
    code: str
    severity: Severity
    file: str
    line: int
    description: str
    context: str

# ============================================================
# FORBIDDEN PATTERNS
# ============================================================

# VASIL coefficients that must NEVER appear in prediction code
FORBIDDEN_COEFFICIENTS = [
    (0.65, "VASIL alpha coefficient"),
    (0.35, "VASIL beta coefficient"),
    (0.92, "VASIL target accuracy (if used as threshold)"),
]

# Patterns indicating look-ahead bias
LOOK_AHEAD_PATTERNS = [
    (r'next_frequency', "Direct access to future frequency"),
    (r'next_week', "Reference to future time"),
    (r'\.shift\s*\(\s*-', "Pandas shift with negative (future) direction"),
    (r't\s*\+\s*1', "Future time index"),
    (r'future_', "Variable with 'future' prefix"),
    (r'observed_direction.*feature', "Observed direction used as feature"),
]

# Patterns indicating train/test leakage
LEAKAGE_PATTERNS = [
    (r'test.*\.mean\(\)', "Computing statistics on test data"),
    (r'test.*\.std\(\)', "Computing statistics on test data"),
    (r'normalize.*test', "Normalizing using test data"),
    (r'fit\s*\(.*test', "Fitting model on test data"),
]

# Patterns indicating data snooping
SNOOPING_PATTERNS = [
    (r'best_params.*test_acc', "Selecting parameters based on test accuracy"),
    (r'if.*test_acc.*>.*threshold', "Using test accuracy in decisions"),
]

# ============================================================
# SCANNING FUNCTIONS
# ============================================================

def scan_for_forbidden_coefficients(content: str, filepath: str) -> List[Violation]:
    """Scan for VASIL paper coefficients."""
    violations = []
    lines = content.split('\n')
    
    for coef, description in FORBIDDEN_COEFFICIENTS:
        # Look for exact coefficient values
        pattern = rf'\b{coef}\b'
        
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                # Check if it's in a suspicious context
                context_words = ['escape', 'transmit', 'fitness', 'weight', 'alpha', 'beta']
                line_lower = line.lower()
                
                if any(w in line_lower for w in context_words):
                    violations.append(Violation(
                        code="IG-003",
                        severity=Severity.CRITICAL,
                        file=filepath,
                        line=i,
                        description=f"Forbidden coefficient: {description}",
                        context=line.strip()
                    ))
    
    return violations

def scan_for_look_ahead(content: str, filepath: str) -> List[Violation]:
    """Scan for look-ahead bias patterns."""
    violations = []
    lines = content.split('\n')
    
    for pattern, description in LOOK_AHEAD_PATTERNS:
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line, re.IGNORECASE):
                # Check if it's in a comment
                stripped = line.strip()
                if stripped.startswith('//') or stripped.startswith('#'):
                    continue
                
                violations.append(Violation(
                    code="IG-001",
                    severity=Severity.CRITICAL,
                    file=filepath,
                    line=i,
                    description=f"Look-ahead bias: {description}",
                    context=line.strip()
                ))
    
    return violations

def scan_for_leakage(content: str, filepath: str) -> List[Violation]:
    """Scan for train/test leakage patterns."""
    violations = []
    lines = content.split('\n')
    
    for pattern, description in LEAKAGE_PATTERNS:
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line, re.IGNORECASE):
                violations.append(Violation(
                    code="IG-002",
                    severity=Severity.CRITICAL,
                    file=filepath,
                    line=i,
                    description=f"Train/test leakage: {description}",
                    context=line.strip()
                ))
    
    return violations

def scan_for_snooping(content: str, filepath: str) -> List[Violation]:
    """Scan for data snooping patterns."""
    violations = []
    lines = content.split('\n')
    
    for pattern, description in SNOOPING_PATTERNS:
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line, re.IGNORECASE):
                violations.append(Violation(
                    code="IG-004",
                    severity=Severity.HIGH,
                    file=filepath,
                    line=i,
                    description=f"Data snooping: {description}",
                    context=line.strip()
                ))
    
    return violations

def check_reproducibility_requirements(content: str, filepath: str) -> List[Violation]:
    """Check for reproducibility requirements."""
    violations = []
    
    # Check for random seed setting
    has_random = 'random' in content.lower() or 'rand' in content.lower()
    has_seed = 'seed' in content.lower() or 'SEED' in content
    
    if has_random and not has_seed:
        violations.append(Violation(
            code="IG-005",
            severity=Severity.HIGH,
            file=filepath,
            line=0,
            description="Randomness used without apparent seed setting",
            context="File uses random operations but no seed found"
        ))
    
    return violations

# ============================================================
# MAIN SCANNER
# ============================================================

def scan_file(filepath: Path) -> List[Violation]:
    """Scan a single file for all integrity violations."""
    try:
        content = filepath.read_text()
    except Exception as e:
        return [Violation(
            code="IG-ERR",
            severity=Severity.LOW,
            file=str(filepath),
            line=0,
            description=f"Could not read file: {e}",
            context=""
        )]
    
    violations = []
    str_path = str(filepath)
    
    violations.extend(scan_for_forbidden_coefficients(content, str_path))
    violations.extend(scan_for_look_ahead(content, str_path))
    violations.extend(scan_for_leakage(content, str_path))
    violations.extend(scan_for_snooping(content, str_path))
    violations.extend(check_reproducibility_requirements(content, str_path))
    
    return violations

def scan_directory(root: Path, extensions: List[str] = None) -> List[Violation]:
    """Scan all relevant files in a directory."""
    if extensions is None:
        extensions = ['.rs', '.cu', '.cuh', '.py']
    
    all_violations = []
    
    for ext in extensions:
        for filepath in root.rglob(f'*{ext}'):
            # Skip target directories
            if 'target' in filepath.parts:
                continue
            
            violations = scan_file(filepath)
            all_violations.extend(violations)
    
    return all_violations

def generate_report(violations: List[Violation]) -> str:
    """Generate a formatted report of violations."""
    if not violations:
        return """
╔══════════════════════════════════════════════════════════════╗
║            PRISM-4D INTEGRITY CHECK: PASSED ✓                ║
║                                                              ║
║  No integrity violations detected.                           ║
║  Codebase is clear for experimentation.                      ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    # Sort by severity
    severity_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
    violations.sort(key=lambda v: (severity_order[v.severity], v.file, v.line))
    
    # Count by severity
    critical = sum(1 for v in violations if v.severity == Severity.CRITICAL)
    high = sum(1 for v in violations if v.severity == Severity.HIGH)
    medium = sum(1 for v in violations if v.severity == Severity.MEDIUM)
    low = sum(1 for v in violations if v.severity == Severity.LOW)
    
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║            PRISM-4D INTEGRITY CHECK: {'FAILED ✗' if critical else 'WARNINGS ⚠'}              ║
╚══════════════════════════════════════════════════════════════╝

SUMMARY
-------
  CRITICAL: {critical}
  HIGH:     {high}
  MEDIUM:   {medium}
  LOW:      {low}
  TOTAL:    {len(violations)}

VIOLATIONS
----------
"""
    
    for v in violations:
        icon = "🛑" if v.severity == Severity.CRITICAL else "⚠️" if v.severity == Severity.HIGH else "ℹ️"
        report += f"""
{icon} [{v.code}] {v.severity.value}
   File: {v.file}:{v.line}
   Issue: {v.description}
   Context: {v.context[:80]}{'...' if len(v.context) > 80 else ''}
"""
    
    if critical > 0:
        report += """
═══════════════════════════════════════════════════════════════
🛑 CRITICAL VIOLATIONS DETECTED - DO NOT PROCEED WITH EXPERIMENTS
   Fix all critical issues before running any benchmarks.
═══════════════════════════════════════════════════════════════
"""
    
    return report

# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PRISM-4D Integrity Checker")
    parser.add_argument("path", type=Path, nargs="?", 
                       default=Path("/mnt/c/Users/Predator/Desktop/PRISM"),
                       help="Path to scan (file or directory)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--strict", action="store_true", 
                       help="Exit with error code if any violations found")
    
    args = parser.parse_args()
    
    if args.path.is_file():
        violations = scan_file(args.path)
    elif args.path.is_dir():
        violations = scan_directory(args.path)
    else:
        print(f"Error: Path not found: {args.path}")
        sys.exit(1)
    
    if args.json:
        import json
        output = [
            {
                "code": v.code,
                "severity": v.severity.value,
                "file": v.file,
                "line": v.line,
                "description": v.description,
                "context": v.context
            }
            for v in violations
        ]
        print(json.dumps(output, indent=2))
    else:
        print(generate_report(violations))
    
    # Exit codes
    if args.strict:
        critical = sum(1 for v in violations if v.severity == Severity.CRITICAL)
        if critical > 0:
            sys.exit(2)  # Critical violations
        elif violations:
            sys.exit(1)  # Other violations
    
    sys.exit(0)

if __name__ == "__main__":
    main()
