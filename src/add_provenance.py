import json, os, datetime

TRUTH = "truth"
OUT = "truth"

def main():
    for name in sorted(os.listdir(TRUTH)):
        tdir = os.path.join(TRUTH, name)
        tpath = os.path.join(tdir, "truth.json")
        if not os.path.isfile(tpath):
            continue
        with open(tpath) as f:
            truth = json.load(f)
        issues = truth.get("issues", [])
        prov = {
            "episode_id": name,
            "generator_version": "1.0",
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "template": "episode_001",
            "issues_injected": [
                {
                    "issue_id": i.get("issue_id"),
                    "type": i.get("type"),
                    "affected_ids": i.get("affected_ids"),
                    "expected_amount_minor": i.get("expected_amount_minor"),
                    "currency": i.get("currency"),
                }
                for i in issues
            ],
            "generation_note": "Synthetic episode derived from template; see truth.json for evidence sets.",
        }
        with open(os.path.join(OUT, name, "provenance.json"), "w") as f:
            json.dump(prov, f, indent=2)
        print(f"wrote truth/{name}/provenance.json")

if __name__ == "__main__":
    main()