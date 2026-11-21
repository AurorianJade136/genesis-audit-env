import json
import csv
import os
import random
import shutil

def generate_episode(source_dir, target_dir, seed):
    random.seed(seed)
    os.makedirs(target_dir, exist_ok=True)
    
    # 1. Randomize IDs mapping
    id_map = {
        "INV-1001": f"INV-{random.randint(2000, 2999)}",
        "INV-1002": f"INV-{random.randint(2000, 2999)}",
        "INV-1003": f"INV-{random.randint(2000, 2999)}",
        "INV-1004": f"INV-{random.randint(2000, 2999)}",
        "INV-1005": f"INV-{random.randint(2000, 2999)}",
        "INV-1006": f"INV-{random.randint(2000, 2999)}",
        "pay_001": f"pay_{random.randint(100, 999)}",
        "pay_002": f"pay_{random.randint(100, 999)}",
        "pay_003": f"pay_{random.randint(100, 999)}",
        "pay_004": f"pay_{random.randint(100, 999)}",
        "pay_005": f"pay_{random.randint(100, 999)}",
        "doc_inv_01": f"doc_inv_{random.randint(10, 99)}",
        "doc_inv_02": f"doc_inv_{random.randint(10, 99)}",
        "doc_inv_03": f"doc_inv_{random.randint(10, 99)}",
        "doc_inv_04": f"doc_inv_{random.randint(10, 99)}",
        "doc_inv_05": f"doc_inv_{random.randint(10, 99)}",
        "doc_inv_06": f"doc_inv_{random.randint(10, 99)}",
        "doc_pay_01": f"doc_pay_{random.randint(10, 99)}",
        "doc_pay_02": f"doc_pay_{random.randint(10, 99)}",
        "doc_pay_03": f"doc_pay_{random.randint(10, 99)}",
        "doc_pay_04": f"doc_pay_{random.randint(10, 99)}",
        "doc_pay_05": f"doc_pay_{random.randint(10, 99)}",
    }
    
    def replace_ids(text):
        for old, new in id_map.items():
            text = text.replace(old, new)
        return text

    # 2. Copy and modify manifest.json
    with open(os.path.join(source_dir, "manifest.json"), "r") as f:
        manifest = json.load(f)
    manifest["episode_id"] = os.path.basename(target_dir)
    with open(os.path.join(target_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # 3. Copy and modify truth.json
    with open(os.path.join(source_dir, "truth.json"), "r") as f:
        truth_text = f.read()
    with open(os.path.join(target_dir, "truth.json"), "w") as f:
        f.write(replace_ids(truth_text))

    # 4. Copy and modify invoices.jsonl & payments.jsonl
    for filename in ["invoices.jsonl", "payments.jsonl"]:
        with open(os.path.join(source_dir, filename), "r") as f:
            text = f.read()
        with open(os.path.join(target_dir, filename), "w") as f:
            f.write(replace_ids(text))

    # 5. Copy and modify ledger.csv and fx.csv
    for filename in ["ledger.csv", "fx.csv", "policy.json"]:
        with open(os.path.join(source_dir, filename), "r") as f:
            text = f.read()
        with open(os.path.join(target_dir, filename), "w") as f:
            f.write(replace_ids(text))

if __name__ == "__main__":
    source = "data/episode_001"
    print("Generating real train episodes...")
    for i in range(1, 11):
        generate_episode(source, f"data/train_{i:02d}", seed=i*10)
    
    print("Generating real dev episodes...")
    for i in range(1, 11):
        generate_episode(source, f"data/dev_{i:02d}", seed=i*100)
    
    print("Successfully generated 10 train and 10 dev episodes.")