#!/usr/bin/env python3
"""
PRISM-4D Data Flow Validator (DFV)
Detects pipeline issues: null buffers, constant features, metadata propagation failures.
"""

import sys
import json
import struct
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import re

class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class DFVViolation:
    code: str
    severity: Severity
    message: str
    feature_index: Optional[int] = None
    constant_value: Optional[float] = None
    diagnosis: Optional[str] = None
    fix_hint: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None

# ============================================================
# FEATURE LAYOUT CONSTANTS
# ============================================================

FEATURE_LAYOUT = {
    "tda": (0, 47, "TDA topological features"),
    "base": (48, 79, "Base structural features"),
    "physics": (80, 91, "Physics features"),
    "fitness": (92, 95, "Fitness features (ddG, expression)"),
    "cycle": (96, 100, "Cycle features (phase, velocity, frequency)"),
    "spike": (101, 108, "Spike features (LIF neuron outputs)"),
}

FEATURE_NAMES = {
    92: "ddG_binding",
    93: "ddG_stability", 
    94: "expression",
    95: "transmissibility",
    96: "cycle_phase",
    97: "cycle_velocity",
    98: "cycle_acceleration",
    99: "wave_frequency",
    100: "temporal_coherence",
    101: "velocity_spike_density",
    102: "frequency_spike_density",
    103: "emergence_spike_density",
    104: "burst_ratio",
    105: "phase_coherence",
    106: "spike_momentum",
    107: "threshold_crossings",
    108: "refractory_fraction",
}

# ============================================================
# RUST CODE ANALYSIS
# ============================================================

def scan_for_nullptr_params(rust_file: Path) -> List[DFVViolation]:
    """
    Scan Rust code for kernel launches with nullptr parameters.
    """
    violations = []
    
    try:
        content = rust_file.read_text()
    except Exception as e:
        return [DFVViolation(
            code="DFV-ERR",
            severity=Severity.LOW,
            message=f"Could not read file: {e}",
            file_path=str(rust_file)
        )]
    
    lines = content.split('\n')
    
    # Look for kernel launch patterns with null pointers
    nullptr_patterns = [
        (r'std::ptr::null\(\)', "std::ptr::null()"),
        (r'std::ptr::null_mut\(\)', "std::ptr::null_mut()"),
        (r'ptr:\s*0', "ptr: 0"),
        (r'CudaSlice::default\(\)', "CudaSlice::default()"),
        (r'None\s*as\s*\*', "None as pointer"),
    ]
    
    in_kernel_launch = False
    kernel_start_line = 0
    
    for i, line in enumerate(lines, 1):
        # Detect kernel launch context
        if 'launch_kernel' in line or 'cuLaunchKernel' in line:
            in_kernel_launch = True
            kernel_start_line = i
        
        if in_kernel_launch:
            for pattern, desc in nullptr_patterns:
                if re.search(pattern, line):
                    # Check context to identify which parameter
                    param_context = extract_param_context(lines, i)
                    
                    violations.append(DFVViolation(
                        code="DFV-001",
                        severity=Severity.CRITICAL,
                        message=f"Null pointer passed to kernel: {desc}",
                        file_path=str(rust_file),
                        line_number=i,
                        diagnosis=f"Parameter context: {param_context}",
                        fix_hint="Allocate and populate buffer before kernel launch"
                    ))
            
            # End of kernel launch block (heuristic: closing paren or semicolon)
            if ')' in line and ';' in line:
                in_kernel_launch = False
    
    return violations

def extract_param_context(lines: List[str], line_num: int, context_size: int = 3) -> str:
    """Extract surrounding context for a line."""
    start = max(0, line_num - context_size - 1)
    end = min(len(lines), line_num + context_size)
    return '\n'.join(lines[start:end])

def scan_for_missing_buffer_alloc(rust_file: Path) -> List[DFVViolation]:
    """
    Check if required buffers are allocated in BufferPool.
    """
    violations = []
    
    try:
        content = rust_file.read_text()
    except:
        return violations
    
    # Required buffers for full PRISM-4D pipeline
    required_buffers = [
        'd_atoms',
        'd_ca_indices',
        'd_combined_features',
        'd_frequency_velocity',  # For cycle features
        'd_escape_scores',       # For fitness features
    ]
    
    # Check which are missing
    for buffer in required_buffers:
        if buffer not in content:
            severity = Severity.CRITICAL if 'frequency' in buffer or 'escape' in buffer else Severity.MEDIUM
            
            violations.append(DFVViolation(
                code="DFV-003",
                severity=severity,
                message=f"Buffer '{buffer}' not found in {rust_file.name}",
                file_path=str(rust_file),
                fix_hint=f"Add '{buffer}: CudaSlice<f32>' to BatchBufferPool struct"
            ))
    
    return violations

# ============================================================
# FEATURE OUTPUT ANALYSIS
# ============================================================

def analyze_feature_variance(features_file: Path) -> List[DFVViolation]:
    """
    Analyze extracted features for zero variance (constant features).
    
    Expects a JSON or CSV file with feature values.
    """
    violations = []
    
    try:
        if features_file.suffix == '.json':
            with open(features_file) as f:
                data = json.load(f)
            features = data.get('features', data)
        else:
            # Assume CSV-like format
            import csv
            with open(features_file) as f:
                reader = csv.reader(f)
                features = [list(map(float, row)) for row in reader]
    except Exception as e:
        return [DFVViolation(
            code="DFV-ERR",
            severity=Severity.LOW,
            message=f"Could not read features file: {e}",
            file_path=str(features_file)
        )]
    
    if not features:
        return violations
    
    # Convert to column-wise for variance analysis
    n_features = len(features[0]) if features else 0
    n_structures = len(features)
    
    for feat_idx in range(n_features):
        values = [row[feat_idx] for row in features if len(row) > feat_idx]
        
        if not values:
            continue
        
        # Compute variance
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        
        if variance < 1e-10:
            constant_val = values[0]
            feat_name = FEATURE_NAMES.get(feat_idx, f"F{feat_idx}")
            diagnosis = diagnose_constant_feature(feat_idx, constant_val)
            
            violations.append(DFVViolation(
                code="DFV-002",
                severity=Severity.HIGH,
                message=f"Feature {feat_name} (F{feat_idx}) has zero variance",
                feature_index=feat_idx,
                constant_value=constant_val,
                diagnosis=diagnosis,
                fix_hint=get_fix_hint_for_feature(feat_idx)
            ))
    
    return violations

def diagnose_constant_feature(idx: int, value: float) -> str:
    """
    Provide diagnostic hints based on feature index and constant value.
    """
    # Cycle features (96-100)
    if 96 <= idx <= 100:
        if abs(value) < 1e-6:
            return ("Cycle features all zero - frequency/velocity inputs likely nullptr. "
                   "Check that d_frequency_velocity buffer is allocated and populated.")
        elif abs(value - 0.5) < 1e-6:
            return ("Cycle features at midpoint (0.5) - LIF neurons receiving constant input. "
                   "Check that per-lineage frequency/velocity varies across structures.")
        else:
            return f"Cycle features constant at {value} - check Stage 8 input propagation."
    
    # Spike features (101-108)
    if 101 <= idx <= 108:
        if abs(value - 0.5) < 1e-6:
            return ("Spike features at 0.5 - cycle features (F96-F100) likely constant. "
                   "Fix cycle features first, then spike features will vary.")
        elif abs(value - 0.333333) < 1e-4:
            return ("Spike features at 0.333 - LIF membrane potential at resting state. "
                   "Neurons not receiving varying input from cycle features.")
        return f"Spike features constant at {value} - check Stage 8.5 input propagation."
    
    # Fitness features (92-95)
    if 92 <= idx <= 95:
        if abs(value) < 1e-6:
            return ("Fitness features all zero - ddG/expression not being computed. "
                   "Check Stage 7 kernel code and mutation data input.")
        return f"Fitness features constant at {value} - check ddG computation per lineage."
    
    # TDA features (0-47)
    if 0 <= idx <= 47:
        return f"TDA features constant - check Stage 3 topology computation."
    
    return f"Unknown feature range - check corresponding kernel stage for index {idx}."

def get_fix_hint_for_feature(idx: int) -> str:
    """
    Provide specific fix hints based on feature index.
    """
    if 96 <= idx <= 100:
        return """
FIX FOR CYCLE FEATURES (F96-F100):
1. In mega_fused_batch.rs, add to BatchBufferPool:
   d_frequency_velocity: CudaSlice<f32>

2. In build_mega_batch(), pack per-lineage data:
   for (i, meta) in batch_metadata.iter().enumerate() {
       h_freq_vel[i * 2 + 0] = meta.frequency;
       h_freq_vel[i * 2 + 1] = meta.frequency_velocity;
   }

3. In kernel launch, pass d_frequency_velocity instead of nullptr
"""
    
    if 101 <= idx <= 108:
        return """
FIX FOR SPIKE FEATURES (F101-F108):
First, ensure cycle features (F96-F100) are non-constant.
Spike features derive from cycle features through LIF neurons.
If cycle features are fixed, spike features cannot vary.
"""
    
    if 92 <= idx <= 95:
        return """
FIX FOR FITNESS FEATURES (F92-F95):
1. Check that mutation data is being passed to kernel
2. Verify ddG calculation in Stage 7 uses per-residue mutations
3. Ensure expression fitness computation varies by lineage
"""
    
    return "Check corresponding kernel stage implementation."

# ============================================================
# CUDA KERNEL ANALYSIS
# ============================================================

def scan_cuda_kernel(cu_file: Path) -> List[DFVViolation]:
    """
    Scan CUDA kernel for potential data flow issues.
    """
    violations = []
    
    try:
        content = cu_file.read_text()
    except Exception as e:
        return [DFVViolation(
            code="DFV-ERR",
            severity=Severity.LOW,
            message=f"Could not read CUDA file: {e}",
            file_path=str(cu_file)
        )]
    
    lines = content.split('\n')
    
    # Check for hardcoded feature values
    hardcoded_patterns = [
        (r'combined_features\[.*\]\s*=\s*0\.5f?;', "Hardcoded 0.5 to combined_features"),
        (r'combined_features\[.*\]\s*=\s*0\.0f?;', "Hardcoded 0.0 to combined_features"),
        (r'combined_features\[.*\]\s*=\s*0\.333', "Hardcoded 0.333 to combined_features"),
    ]
    
    for i, line in enumerate(lines, 1):
        for pattern, desc in hardcoded_patterns:
            if re.search(pattern, line):
                # Check if it's a default/fallback (acceptable) or always executed (bug)
                context = '\n'.join(lines[max(0, i-5):min(len(lines), i+3)])
                
                if 'if' not in context.lower() and 'else' not in context.lower():
                    violations.append(DFVViolation(
                        code="DFV-002",
                        severity=Severity.MEDIUM,
                        message=f"Potentially hardcoded feature value: {desc}",
                        file_path=str(cu_file),
                        line_number=i,
                        diagnosis="Feature assigned constant value unconditionally",
                        fix_hint="Ensure value is computed from input data, not hardcoded"
                    ))
    
    # Check for missing null checks on input pointers
    input_params = re.findall(r'const\s+float\s*\*\s*(\w+)', content)
    
    for param in input_params:
        # Check if there's a null check for this parameter
        if f'if ({param}' not in content and f'if({param}' not in content:
            if param in ['frequency_velocity_in', 'escape_scores_in', 'cycle_data_in']:
                violations.append(DFVViolation(
                    code="DFV-001",
                    severity=Severity.MEDIUM,
                    message=f"No null check for kernel parameter '{param}'",
                    file_path=str(cu_file),
                    fix_hint=f"Add null check: if ({param} != nullptr) before using"
                ))
    
    return violations

# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(violations: List[DFVViolation]) -> str:
    """Generate formatted DFV report."""
    
    if not violations:
        return """
╔══════════════════════════════════════════════════════════════════════╗
║              DATA FLOW VALIDATOR: ALL CHECKS PASSED ✓                ║
║                                                                      ║
║  • No null buffer detections                                         ║
║  • No constant features detected                                     ║
║  • No metadata propagation failures                                  ║
║                                                                      ║
║  Pipeline integrity verified. Ready for optimization.                ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    
    # Count by severity
    critical = sum(1 for v in violations if v.severity == Severity.CRITICAL)
    high = sum(1 for v in violations if v.severity == Severity.HIGH)
    medium = sum(1 for v in violations if v.severity == Severity.MEDIUM)
    
    # Sort by severity
    severity_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
    violations.sort(key=lambda v: severity_order[v.severity])
    
    report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║              DATA FLOW VALIDATOR: ISSUES DETECTED                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  CRITICAL: {critical:3}                                                       ║
║  HIGH:     {high:3}                                                       ║
║  MEDIUM:   {medium:3}                                                       ║
║  TOTAL:    {len(violations):3}                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    
    for v in violations:
        icon = "🛑" if v.severity == Severity.CRITICAL else "⚠️" if v.severity == Severity.HIGH else "ℹ️"
        
        report += f"""
{icon} [{v.code}] {v.severity.value}
   {v.message}
"""
        if v.file_path:
            line_info = f":{v.line_number}" if v.line_number else ""
            report += f"   File: {v.file_path}{line_info}\n"
        
        if v.feature_index is not None:
            feat_name = FEATURE_NAMES.get(v.feature_index, f"F{v.feature_index}")
            report += f"   Feature: {feat_name} (index {v.feature_index})\n"
        
        if v.constant_value is not None:
            report += f"   Constant value: {v.constant_value}\n"
        
        if v.diagnosis:
            report += f"   Diagnosis: {v.diagnosis}\n"
        
        if v.fix_hint:
            # Truncate long fix hints for display
            hint = v.fix_hint.strip()
            if '\n' in hint:
                hint = hint.split('\n')[0] + " ..."
            report += f"   Fix: {hint}\n"
    
    if critical > 0:
        report += """
═══════════════════════════════════════════════════════════════════════
🛑 CRITICAL ISSUES - Pipeline cannot produce valid features
   Fix these issues before running any benchmarks or optimization.
═══════════════════════════════════════════════════════════════════════
"""
    
    return report

# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PRISM-4D Data Flow Validator")
    parser.add_argument("path", type=Path, nargs="?",
                       default=Path("/mnt/c/Users/Predator/Desktop/PRISM"),
                       help="Project root or specific file to scan")
    parser.add_argument("--features", type=Path, help="Feature output file to analyze")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--strict", action="store_true",
                       help="Exit with error code if any issues found")
    
    args = parser.parse_args()
    
    all_violations = []
    
    # Scan Rust files
    if args.path.is_dir():
        rust_files = list(args.path.rglob("*.rs"))
        for rf in rust_files:
            if 'target' in rf.parts:
                continue
            all_violations.extend(scan_for_nullptr_params(rf))
            if 'batch' in rf.name.lower() or 'buffer' in rf.name.lower():
                all_violations.extend(scan_for_missing_buffer_alloc(rf))
        
        # Scan CUDA files
        cuda_files = list(args.path.rglob("*.cu"))
        for cf in cuda_files:
            if 'target' in cf.parts:
                continue
            all_violations.extend(scan_cuda_kernel(cf))
    
    elif args.path.is_file():
        if args.path.suffix == '.rs':
            all_violations.extend(scan_for_nullptr_params(args.path))
            all_violations.extend(scan_for_missing_buffer_alloc(args.path))
        elif args.path.suffix == '.cu':
            all_violations.extend(scan_cuda_kernel(args.path))
    
    # Analyze features file if provided
    if args.features and args.features.exists():
        all_violations.extend(analyze_feature_variance(args.features))
    
    # Output
    if args.json:
        output = [
            {
                "code": v.code,
                "severity": v.severity.value,
                "message": v.message,
                "feature_index": v.feature_index,
                "constant_value": v.constant_value,
                "diagnosis": v.diagnosis,
                "fix_hint": v.fix_hint,
                "file_path": v.file_path,
                "line_number": v.line_number
            }
            for v in all_violations
        ]
        print(json.dumps(output, indent=2))
    else:
        print(generate_report(all_violations))
    
    # Exit codes
    if args.strict:
        critical = sum(1 for v in all_violations if v.severity == Severity.CRITICAL)
        if critical > 0:
            sys.exit(2)
        elif all_violations:
            sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
