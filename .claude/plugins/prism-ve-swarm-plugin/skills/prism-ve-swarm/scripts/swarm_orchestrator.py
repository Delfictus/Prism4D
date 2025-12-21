#!/usr/bin/env python3
"""
PRISM-4D Multi-Agent Swarm Orchestrator
Coordinates hypothesis testing with scientific integrity guarantees.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from enum import Enum
import subprocess
import sys

# ============================================================
# SWARM STATE MANAGEMENT
# ============================================================

class SwarmPhase(Enum):
    INIT = "init"
    HYPOTHESIS = "hypothesis"
    VALIDATION = "validation"
    FINALIZATION = "finalization"
    HALTED = "halted"

class IntegrityStatus(Enum):
    CLEAN = "clean"
    WARNING = "warning"
    VIOLATION = "violation"

@dataclass
class HypothesisResult:
    id: str
    status: str  # proposed, testing, accepted, rejected
    baseline_accuracy: Optional[float] = None
    treatment_accuracy: Optional[float] = None
    effect_size: Optional[float] = None
    p_value: Optional[float] = None
    integrity_status: str = "pending"

@dataclass
class SwarmState:
    phase: SwarmPhase = SwarmPhase.INIT
    cycle_number: int = 0
    baseline_accuracy: float = 0.423  # Current known baseline
    best_accuracy: float = 0.423
    best_configuration: Dict[str, Any] = field(default_factory=dict)
    active_hypothesis: Optional[str] = None
    hypothesis_queue: List[str] = field(default_factory=list)
    hypothesis_results: Dict[str, HypothesisResult] = field(default_factory=dict)
    integrity_violations: List[str] = field(default_factory=list)
    integrity_warnings: List[str] = field(default_factory=list)
    experiment_log: List[Dict] = field(default_factory=list)
    
    def to_dict(self):
        d = asdict(self)
        d['phase'] = self.phase.value
        return d
    
    @classmethod
    def from_dict(cls, d):
        d['phase'] = SwarmPhase(d['phase'])
        d['hypothesis_results'] = {
            k: HypothesisResult(**v) for k, v in d.get('hypothesis_results', {}).items()
        }
        return cls(**d)

# ============================================================
# WIN CONDITIONS (IMMUTABLE)
# ============================================================

WIN_CONDITIONS = {
    "target_accuracy": 0.92,
    "max_cycles": 50,
    "min_improvement_threshold": 0.005,
    "consecutive_low_improvement_limit": 3,
    "max_train_test_gap": 0.10
}

VASIL_COUNTRIES = [
    "Germany", "USA", "UK", "Japan", "Brazil", "France",
    "Canada", "Denmark", "Australia", "Sweden", "Mexico", "SouthAfrica"
]

# ============================================================
# INTEGRITY GUARDIAN
# ============================================================

class IntegrityGuardian:
    """Agent responsible for scientific integrity enforcement."""
    
    FORBIDDEN_COEFFICIENTS = [0.65, 0.35, 0.92]
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.audit_log = []
    
    def pre_execution_audit(self, hypothesis_id: str, files_modified: List[str]) -> IntegrityStatus:
        """Audit before running an experiment."""
        violations = []
        warnings = []
        
        # Check each modified file
        for filepath in files_modified:
            full_path = self.project_root / filepath
            if full_path.exists():
                content = full_path.read_text()
                
                # Check for forbidden coefficients
                for coef in self.FORBIDDEN_COEFFICIENTS:
                    if str(coef) in content:
                        # Context check
                        if self._is_suspicious_coefficient_use(content, coef):
                            violations.append(f"IG-003: Forbidden coefficient {coef} found in {filepath}")
                
                # Check for look-ahead patterns
                look_ahead = self._detect_look_ahead_patterns(content)
                if look_ahead:
                    violations.append(f"IG-001: Look-ahead pattern detected in {filepath}: {look_ahead}")
        
        self.audit_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "hypothesis": hypothesis_id,
            "audit_type": "pre_execution",
            "violations": violations,
            "warnings": warnings
        })
        
        if violations:
            return IntegrityStatus.VIOLATION
        elif warnings:
            return IntegrityStatus.WARNING
        return IntegrityStatus.CLEAN
    
    def post_execution_audit(self, hypothesis_id: str, results: Dict) -> IntegrityStatus:
        """Audit after experiment completion."""
        violations = []
        warnings = []
        
        train_acc = results.get('train_accuracy', 0)
        test_acc = results.get('test_accuracy', 0)
        
        # Check for overfitting
        gap = train_acc - test_acc
        if gap > WIN_CONDITIONS['max_train_test_gap']:
            warnings.append(f"IG-WARNING: Train-test gap {gap:.2%} exceeds threshold")
        
        # Check for suspiciously perfect results
        if test_acc > 0.98:
            violations.append("IG-002: Suspiciously high accuracy - check for data leakage")
        
        # Check reproducibility (would need actual implementation)
        # violations.extend(self._check_reproducibility(hypothesis_id))
        
        self.audit_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "hypothesis": hypothesis_id,
            "audit_type": "post_execution",
            "violations": violations,
            "warnings": warnings
        })
        
        if violations:
            return IntegrityStatus.VIOLATION
        elif warnings:
            return IntegrityStatus.WARNING
        return IntegrityStatus.CLEAN
    
    def _is_suspicious_coefficient_use(self, content: str, coef: float) -> bool:
        """Check if coefficient appears in suspicious context."""
        # Look for patterns like "0.65 * escape" or "escape * 0.65"
        suspicious_patterns = [
            f"{coef} * escape",
            f"escape * {coef}",
            f"{coef} * transmit",
            f"transmit * {coef}",
            f"alpha = {coef}",
            f"beta = {coef}"
        ]
        return any(p in content.lower() for p in suspicious_patterns)
    
    def _detect_look_ahead_patterns(self, content: str) -> Optional[str]:
        """Detect potential look-ahead bias patterns."""
        look_ahead_patterns = [
            ("next_frequency", "Using next frequency in computation"),
            ("future_date", "Reference to future date"),
            ("t + 1", "Potential future time reference"),
            (".shift(-", "Pandas shift with negative (future) values"),
        ]
        
        for pattern, description in look_ahead_patterns:
            if pattern in content:
                return description
        return None
    
    def generate_certificate(self, hypothesis_id: str) -> str:
        """Generate integrity certificate for a hypothesis."""
        relevant_audits = [a for a in self.audit_log if a['hypothesis'] == hypothesis_id]
        
        all_violations = []
        all_warnings = []
        for audit in relevant_audits:
            all_violations.extend(audit.get('violations', []))
            all_warnings.extend(audit.get('warnings', []))
        
        status = "CERTIFIED" if not all_violations else "FAILED"
        
        return f"""
PRISM-4D INTEGRITY CERTIFICATE
==============================
Hypothesis: {hypothesis_id}
Date: {datetime.utcnow().isoformat()}
Status: {status}

Audits Performed: {len(relevant_audits)}
Violations: {len(all_violations)}
Warnings: {len(all_warnings)}

{chr(10).join('- ' + v for v in all_violations) if all_violations else 'No violations detected.'}

{chr(10).join('- ' + w for w in all_warnings) if all_warnings else 'No warnings.'}
"""

# ============================================================
# STATISTICAL VALIDATOR
# ============================================================

class StatisticalValidator:
    """Agent responsible for statistical rigor."""
    
    @staticmethod
    def compute_wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple:
        """Wilson score confidence interval."""
        import math
        
        if n == 0:
            return (0.0, 1.0)
        
        z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        
        p_hat = successes / n
        
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denominator
        margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator
        
        return (max(0, center - margin), min(1, center + margin))
    
    @staticmethod
    def test_significance(baseline_acc: float, new_acc: float, n: int) -> Dict:
        """Test if improvement is statistically significant."""
        # Simple z-test for proportions
        import math
        
        p1 = baseline_acc
        p2 = new_acc
        p_pooled = (p1 + p2) / 2
        
        if p_pooled == 0 or p_pooled == 1:
            return {"p_value": 1.0, "significant": False}
        
        se = math.sqrt(2 * p_pooled * (1 - p_pooled) / n)
        
        if se == 0:
            return {"p_value": 1.0, "significant": False}
        
        z = (p2 - p1) / se
        
        # Two-tailed p-value (approximate)
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        
        return {
            "z_score": z,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "effect_size": p2 - p1
        }
    
    def validate_results(self, results: Dict) -> Dict:
        """Comprehensive validation of experimental results."""
        validation = {
            "passed": True,
            "issues": []
        }
        
        # Check sample size
        if results.get('n_test', 0) < 100:
            validation['issues'].append("Sample size too small for reliable inference")
            validation['passed'] = False
        
        # Compute CI
        if 'test_accuracy' in results and 'n_test' in results:
            ci = self.compute_wilson_ci(
                int(results['test_accuracy'] * results['n_test']),
                results['n_test']
            )
            results['ci_95'] = ci
        
        # Check for significance
        if 'baseline_accuracy' in results and 'test_accuracy' in results:
            sig = self.test_significance(
                results['baseline_accuracy'],
                results['test_accuracy'],
                results.get('n_test', 9340)
            )
            results['significance'] = sig
            
            if not sig['significant']:
                validation['issues'].append("Improvement not statistically significant")
        
        return validation

# ============================================================
# SWARM ORCHESTRATOR
# ============================================================

class SwarmOrchestrator:
    """Central coordinator for the multi-agent swarm."""
    
    def __init__(self, project_root: Path, state_file: Path = None):
        self.project_root = project_root
        self.state_file = state_file or project_root / "swarm_state.json"
        
        # Initialize agents
        self.integrity_guardian = IntegrityGuardian(project_root)
        self.statistical_validator = StatisticalValidator()
        
        # Load or initialize state
        if self.state_file.exists():
            self.state = self._load_state()
        else:
            self.state = SwarmState()
    
    def _load_state(self) -> SwarmState:
        with open(self.state_file) as f:
            return SwarmState.from_dict(json.load(f))
    
    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2, default=str)
    
    def should_continue(self) -> bool:
        """Check if swarm should continue running."""
        # Check for integrity violations
        if self.state.integrity_violations:
            print("🛑 HALT: Integrity violations detected")
            return False
        
        # Check if target reached
        if self.state.best_accuracy >= WIN_CONDITIONS['target_accuracy']:
            print(f"🏆 TARGET REACHED: {self.state.best_accuracy:.1%}")
            return False
        
        # Check max cycles
        if self.state.cycle_number >= WIN_CONDITIONS['max_cycles']:
            print(f"⏰ MAX CYCLES REACHED: {self.state.cycle_number}")
            return False
        
        # Check for diminishing returns
        if self._check_diminishing_returns():
            print("📉 DIMINISHING RETURNS: Stopping early")
            return False
        
        return True
    
    def _check_diminishing_returns(self) -> bool:
        """Check if recent cycles show minimal improvement."""
        if len(self.state.experiment_log) < WIN_CONDITIONS['consecutive_low_improvement_limit']:
            return False
        
        recent = self.state.experiment_log[-WIN_CONDITIONS['consecutive_low_improvement_limit']:]
        
        for exp in recent:
            if exp.get('effect_size', 0) > WIN_CONDITIONS['min_improvement_threshold']:
                return False
        
        return True
    
    def run_hypothesis_cycle(self, hypothesis_id: str, files_modified: List[str]) -> Dict:
        """Execute a single hypothesis testing cycle."""
        self.state.cycle_number += 1
        self.state.active_hypothesis = hypothesis_id
        self.state.phase = SwarmPhase.HYPOTHESIS
        self._save_state()
        
        print(f"\n{'='*60}")
        print(f"CYCLE {self.state.cycle_number}: Testing {hypothesis_id}")
        print(f"{'='*60}")
        
        # Phase 1: Pre-execution integrity check
        print("\n[IG] Pre-execution integrity audit...")
        integrity_status = self.integrity_guardian.pre_execution_audit(hypothesis_id, files_modified)
        
        if integrity_status == IntegrityStatus.VIOLATION:
            self.state.integrity_violations.append(f"Cycle {self.state.cycle_number}: {hypothesis_id}")
            self.state.phase = SwarmPhase.HALTED
            self._save_state()
            return {"status": "HALTED", "reason": "Integrity violation"}
        
        # Phase 2: Run experiment (placeholder - actual implementation needed)
        print("\n[FE] Running experiment...")
        results = self._run_experiment(hypothesis_id)
        
        # Phase 3: Statistical validation
        print("\n[SV] Statistical validation...")
        self.state.phase = SwarmPhase.VALIDATION
        validation = self.statistical_validator.validate_results(results)
        
        # Phase 4: Post-execution integrity check
        print("\n[IG] Post-execution integrity audit...")
        post_integrity = self.integrity_guardian.post_execution_audit(hypothesis_id, results)
        
        if post_integrity == IntegrityStatus.VIOLATION:
            self.state.integrity_violations.append(f"Post-execution: {hypothesis_id}")
            self.state.phase = SwarmPhase.HALTED
            self._save_state()
            return {"status": "HALTED", "reason": "Post-execution integrity violation"}
        
        # Phase 5: Record results
        hypothesis_result = HypothesisResult(
            id=hypothesis_id,
            status="accepted" if self._is_improvement(results) else "rejected",
            baseline_accuracy=self.state.best_accuracy,
            treatment_accuracy=results.get('test_accuracy'),
            effect_size=results.get('effect_size'),
            p_value=results.get('significance', {}).get('p_value'),
            integrity_status=post_integrity.value
        )
        
        self.state.hypothesis_results[hypothesis_id] = hypothesis_result
        
        # Update best if improved
        if self._is_improvement(results):
            self.state.best_accuracy = results['test_accuracy']
            self.state.best_configuration[hypothesis_id] = True
            print(f"\n✓ ACCEPTED: New best accuracy: {self.state.best_accuracy:.1%}")
        else:
            print(f"\n✗ REJECTED: No significant improvement")
        
        # Log experiment
        self.state.experiment_log.append({
            "cycle": self.state.cycle_number,
            "hypothesis": hypothesis_id,
            "results": results,
            "validation": validation,
            "outcome": hypothesis_result.status
        })
        
        self.state.active_hypothesis = None
        self._save_state()
        
        return {
            "status": "COMPLETED",
            "hypothesis_result": asdict(hypothesis_result),
            "validation": validation
        }
    
    def _run_experiment(self, hypothesis_id: str) -> Dict:
        """Placeholder for actual experiment execution."""
        # This would actually run the PRISM benchmark
        # For now, return placeholder results
        return {
            "test_accuracy": self.state.best_accuracy,  # Placeholder
            "train_accuracy": self.state.best_accuracy + 0.05,
            "n_test": 9340,
            "n_train": 1745,
            "effect_size": 0.0
        }
    
    def _is_improvement(self, results: Dict) -> bool:
        """Check if results represent meaningful improvement."""
        effect = results.get('effect_size', 0)
        significant = results.get('significance', {}).get('significant', False)
        
        return effect > WIN_CONDITIONS['min_improvement_threshold'] and significant
    
    def generate_report(self) -> str:
        """Generate comprehensive swarm execution report."""
        report = f"""
PRISM-4D SWARM EXECUTION REPORT
===============================
Generated: {datetime.utcnow().isoformat()}

SUMMARY
-------
Cycles Completed: {self.state.cycle_number}
Baseline Accuracy: {self.state.baseline_accuracy:.1%}
Best Accuracy: {self.state.best_accuracy:.1%}
Total Improvement: {(self.state.best_accuracy - self.state.baseline_accuracy):.1%}
Target: {WIN_CONDITIONS['target_accuracy']:.1%}
Gap Remaining: {(WIN_CONDITIONS['target_accuracy'] - self.state.best_accuracy):.1%}

INTEGRITY STATUS
----------------
Violations: {len(self.state.integrity_violations)}
Warnings: {len(self.state.integrity_warnings)}

HYPOTHESIS OUTCOMES
-------------------
"""
        for hyp_id, result in self.state.hypothesis_results.items():
            effect = f"+{result.effect_size:.1%}" if result.effect_size else "N/A"
            report += f"- {hyp_id}: {result.status.upper()} ({effect})\n"
        
        return report

# ============================================================
# CLI INTERFACE
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PRISM-4D Multi-Agent Swarm Orchestrator")
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/c/Users/Predator/Desktop/PRISM"))
    parser.add_argument("--init", action="store_true", help="Initialize new swarm state")
    parser.add_argument("--status", action="store_true", help="Show current swarm status")
    parser.add_argument("--run-cycle", type=str, help="Run hypothesis cycle with given ID")
    parser.add_argument("--files", nargs="+", help="Files modified for hypothesis")
    parser.add_argument("--report", action="store_true", help="Generate execution report")
    
    args = parser.parse_args()
    
    orchestrator = SwarmOrchestrator(args.project_root)
    
    if args.init:
        orchestrator.state = SwarmState()
        orchestrator._save_state()
        print("✓ Swarm state initialized")
    
    elif args.status:
        print(f"Phase: {orchestrator.state.phase.value}")
        print(f"Cycle: {orchestrator.state.cycle_number}")
        print(f"Best Accuracy: {orchestrator.state.best_accuracy:.1%}")
        print(f"Integrity Violations: {len(orchestrator.state.integrity_violations)}")
    
    elif args.run_cycle:
        if not args.files:
            print("Error: --files required with --run-cycle")
            sys.exit(1)
        result = orchestrator.run_hypothesis_cycle(args.run_cycle, args.files)
        print(json.dumps(result, indent=2))
    
    elif args.report:
        print(orchestrator.generate_report())
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
