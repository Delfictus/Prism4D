#!/usr/bin/env python3
"""
PRISM-LBS CryptoBench Validation Script
Compares detected pockets against ground truth binding sites.
"""

import json
import os
import csv
import sys
from pathlib import Path
from collections import defaultdict

def load_ground_truth(dataset_path):
    """Load CryptoBench ground truth dataset."""
    with open(dataset_path, 'r') as f:
        return json.load(f)

def extract_residue_numbers(pocket_selection):
    """Extract residue numbers from pocket selection like 'B_12' -> 12."""
    residues = set()
    for item in pocket_selection:
        parts = item.split('_')
        if len(parts) >= 2:
            try:
                residues.add(int(parts[1]))
            except ValueError:
                pass
    return residues

def load_prism_results(results_dir):
    """Load all PRISM pocket detection results."""
    results = {}
    for json_file in Path(results_dir).glob("*.json"):
        pdb_id = json_file.stem.lower()
        with open(json_file, 'r') as f:
            data = json.load(f)
            results[pdb_id] = data
    return results

def compute_overlap(detected_residues, ground_truth_residues):
    """Compute overlap metrics between detected and ground truth residues."""
    if not detected_residues or not ground_truth_residues:
        return 0.0, 0.0, 0.0

    intersection = detected_residues & ground_truth_residues

    precision = len(intersection) / len(detected_residues) if detected_residues else 0.0
    recall = len(intersection) / len(ground_truth_residues) if ground_truth_residues else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def main():
    # Paths
    results_dir = "/mnt/c/Users/Predator/Desktop/PRISM/lbs_results/cryptobench_100_batch"
    dataset_path = "/mnt/c/Users/Predator/Desktop/PRISM/benchmarks/datasets/cryptobench/dataset.json"
    output_csv = "/mnt/c/Users/Predator/Desktop/PRISM/lbs_results/cryptobench_100_batch/validation_summary.csv"
    output_json = "/mnt/c/Users/Predator/Desktop/PRISM/lbs_results/cryptobench_100_batch/validation_detailed.json"

    print("=" * 70)
    print("PRISM-LBS CryptoBench Validation Report")
    print("=" * 70)

    # Load data
    print("\nLoading ground truth dataset...")
    ground_truth = load_ground_truth(dataset_path)
    print(f"  Ground truth entries: {len(ground_truth)}")

    print("\nLoading PRISM results...")
    prism_results = load_prism_results(results_dir)
    print(f"  PRISM results: {len(prism_results)}")

    # Validation results
    validation_results = []
    stats = {
        'total_structures': len(prism_results),
        'structures_with_pockets': 0,
        'structures_without_pockets': 0,
        'structures_with_gt_match': 0,
        'structures_passing_druggability': 0,
        'avg_druggability': 0.0,
        'avg_precision': 0.0,
        'avg_recall': 0.0,
        'avg_f1': 0.0,
    }

    druggability_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    print("\nAnalyzing each structure...")
    print("-" * 70)
    print(f"{'PDB':<8} {'Pockets':<8} {'Druggability':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Status'}")
    print("-" * 70)

    for pdb_id, result in sorted(prism_results.items()):
        pockets = result.get('pockets', [])
        num_atoms = result.get('num_atoms', 0)
        structure_name = result.get('structure', '')

        # Get detected residues from all pockets
        detected_residues = set()
        max_druggability = 0.0
        total_volume = 0.0

        for pocket in pockets:
            detected_residues.update(pocket.get('residue_indices', []))
            druggability = pocket.get('druggability', 0.0)
            max_druggability = max(max_druggability, druggability)
            total_volume += pocket.get('volume', 0.0)

        # Get ground truth for this PDB
        gt_entries = ground_truth.get(pdb_id, [])
        gt_residues = set()
        for entry in gt_entries:
            gt_residues.update(extract_residue_numbers(entry.get('apo_pocket_selection', [])))

        # Compute overlap if we have ground truth
        precision, recall, f1 = 0.0, 0.0, 0.0
        has_gt = len(gt_residues) > 0

        if has_gt and detected_residues:
            precision, recall, f1 = compute_overlap(detected_residues, gt_residues)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            stats['structures_with_gt_match'] += 1

        # Track statistics
        if pockets:
            stats['structures_with_pockets'] += 1
            druggability_scores.append(max_druggability)
        else:
            stats['structures_without_pockets'] += 1

        if max_druggability >= 0.60:
            stats['structures_passing_druggability'] += 1

        # Determine status
        if not pockets:
            status = "NO_POCKETS"
        elif max_druggability >= 0.60:
            if f1 >= 0.3:
                status = "PASS"
            elif has_gt:
                status = "LOW_OVERLAP"
            else:
                status = "NO_GT"
        else:
            status = "LOW_DRUG"

        # Print row
        print(f"{pdb_id:<8} {len(pockets):<8} {max_druggability:<12.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {status}")

        # Store detailed result
        validation_results.append({
            'pdb_id': pdb_id,
            'structure_name': structure_name,
            'num_atoms': num_atoms,
            'num_pockets': len(pockets),
            'max_druggability': max_druggability,
            'total_volume': total_volume,
            'num_detected_residues': len(detected_residues),
            'num_gt_residues': len(gt_residues),
            'has_ground_truth': has_gt,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'status': status,
            'detected_residues': sorted(list(detected_residues)),
            'gt_residues': sorted(list(gt_residues))
        })

    # Compute averages
    if druggability_scores:
        stats['avg_druggability'] = sum(druggability_scores) / len(druggability_scores)
    if precision_scores:
        stats['avg_precision'] = sum(precision_scores) / len(precision_scores)
    if recall_scores:
        stats['avg_recall'] = sum(recall_scores) / len(recall_scores)
    if f1_scores:
        stats['avg_f1'] = sum(f1_scores) / len(f1_scores)

    # Summary
    print("-" * 70)
    print("\nSUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total structures analyzed:        {stats['total_structures']}")
    print(f"Structures with pockets:          {stats['structures_with_pockets']}")
    print(f"Structures without pockets:       {stats['structures_without_pockets']}")
    print(f"Passing druggability (>=0.60):    {stats['structures_passing_druggability']}")
    print(f"Structures with ground truth:     {stats['structures_with_gt_match']}")
    print()
    print(f"Average max druggability:         {stats['avg_druggability']:.4f}")
    print(f"Average precision (GT matches):   {stats['avg_precision']:.4f}")
    print(f"Average recall (GT matches):      {stats['avg_recall']:.4f}")
    print(f"Average F1 score (GT matches):    {stats['avg_f1']:.4f}")

    # Druggability distribution
    print("\nDRUGGABILITY DISTRIBUTION")
    print("-" * 40)
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
    for low, high in bins:
        count = sum(1 for d in druggability_scores if low <= d < high)
        bar = '#' * count
        print(f"  {low:.1f}-{high:.1f}: {count:3d} {bar}")

    # Write CSV
    print(f"\nWriting CSV summary to: {output_csv}")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'pdb_id', 'num_pockets', 'max_druggability', 'total_volume',
            'num_detected_residues', 'num_gt_residues', 'precision', 'recall', 'f1', 'status'
        ])
        writer.writeheader()
        for r in validation_results:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    # Write detailed JSON
    print(f"Writing detailed JSON to: {output_json}")
    with open(output_json, 'w') as f:
        json.dump({
            'summary_stats': stats,
            'results': validation_results
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70)

    return stats

if __name__ == "__main__":
    main()
