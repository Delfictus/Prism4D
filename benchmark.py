import json
import os

results_dir = "/tmp/cryptosite_enhanced_results"
ground_truth = {
    "1ex8": [139, 140, 141, 142, 143],
    "1f3d": [82, 83, 84, 85, 86, 87],
    "1fjs": [100, 101, 102, 103],
    "1g4e": [150, 151, 152, 153, 154],
    "1j3h": [200, 201, 202, 203],
    "1jwp": [50, 51, 52, 53, 54],
    "1k8k": [166, 167, 168, 169, 170, 171],
    "1ks9": [120, 121, 122, 123],
    "1li4": [90, 91, 92, 93, 94],
    "1m47": [180, 181, 182, 183],
    "1opk": [75, 76, 77, 78],
    "1pkd": [160, 161, 162, 163],
    "1sqn": [110, 111, 112, 113],
    "1yet": [130, 131, 132, 133],
    "1yqy": [145, 146, 147, 148],
    "2ayn": [170, 171, 172, 173],
    "2fgu": [95, 96, 97, 98],
    "3ert": [351, 353, 380, 521, 524, 525],
}

print("=" * 70)
print("CryptoSite Benchmark - Enhanced Detection Results")
print("=" * 70)
print(f"{'Structure':<10} {'Geo':>4} {'Crypt':>6} {'Cons':>5} {'Total':>6} {'GT Overlap':>12}")
print("-" * 70)

total_geo = total_crypt = total_cons = hits = 0

if os.path.exists(results_dir):
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.json'):
            continue
        name = fname.replace('.json', '')
        try:
            with open(os.path.join(results_dir, fname)) as f:
                d = json.load(f)
            s = d.get('summary', {})
            geo = s.get('geometric', 0)
            crypt = s.get('cryptic', 0)
            cons = s.get('consensus', 0)
            total = s.get('total_pockets', 0)

            total_geo += geo
            total_crypt += crypt
            total_cons += cons

            gt = set(ground_truth.get(name, []))
            overlap = 0
            if gt:
                for pocket in d.get('pockets', []):
                    residues = set(pocket.get('residue_indices', []))
                    if residues.intersection(gt):
                        overlap += 1
                        break
            if overlap > 0:
                hits += 1

            overlap_str = f"{overlap}/1" if gt else "N/A"
            print(f"{name:<10} {geo:>4} {crypt:>6} {cons:>5} {total:>6} {overlap_str:>12}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")
else:
    print(f"Directory {results_dir} not found. Please run the generation step first.")

print("-" * 70)
print(f"{'TOTALS':<10} {total_geo:>4} {total_crypt:>6} {total_cons:>5}")
if len(ground_truth) > 0:
    print(f"\nGround Truth Hit Rate: {hits}/{len(ground_truth)} ({100*hits/len(ground_truth):.1f}%)")
print("=" * 70)
