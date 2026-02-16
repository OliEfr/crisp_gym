import json
import os

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_PATH = "/home/maxchr/datasets/stack_v6"
# Only extract these keys (add others if you specifically need them)
KEYS_TO_KEEP = ["action", "observation.state"] 
# ==========================================

def load_condensed_stats():
    stats_path = os.path.join(DATASET_PATH, "meta", "stats.json")
    
    if not os.path.exists(stats_path):
        print(f"ERROR: File not found at {stats_path}")
        return

    with open(stats_path, 'r') as f:
        data = json.load(f)

    condensed = {}
    
    for key in KEYS_TO_KEEP:
        if key in data:
            # We only want mean and std
            condensed[key] = {
                "mean": data[key]["mean"],
                "std": data[key]["std"]
            }
        else:
            print(f"Warning: Key '{key}' not found in stats.")

    print("\n" + "="*30)
    print("CONDENSED STATS (Mean & Std Only)")
    print("="*30 + "\n")
    
    # Print in a format easy to copy-paste (JSON)
    print(json.dumps(condensed, indent=4))
    
    # Optional: Print a YAML-like format if you prefer that
    # print("\n--- YAML STYLE ---")
    # for key, val in condensed.items():
    #     print(f"{key}:")
    #     print(f"  mean: {val['mean']}")
    #     print(f"  std: {val['std']}")

if __name__ == "__main__":
    load_condensed_stats()