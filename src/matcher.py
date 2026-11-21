def match(truth_issues, findings, inspection_steps):
    used_truth = set()
    used_findings = set()
    tp = 0

    for i, f in enumerate(findings):
        for j, t in enumerate(truth_issues):
            if j in used_truth:
                continue
            
            # Check if finding matches truth
            same_type = (f.get("type") == t.get("type"))
            same_ids = (set(f.get("affected_ids", [])) == set(t.get("affected_ids", [])))
            same_amount = (f.get("claimed_amount_minor") == t.get("expected_amount_minor"))
            same_currency = (f.get("currency") == t.get("currency"))
            
            # Check evidence
            valid_evidence = False
            for valid_set in t.get("valid_evidence_sets", []):
                if set(valid_set).issubset(set(f.get("evidence_ids", []))):
                    valid_evidence = True
                    break
            
            if same_type and same_ids and same_amount and same_currency and valid_evidence:
                tp += 1
                used_truth.add(j)
                used_findings.add(i)
                break
                
    fp = len(findings) - tp
    fn = len(truth_issues) - tp

    if tp + fp + fn == 0:
        f1 = 1.0 if len(truth_issues) == 0 else 0.0
    elif tp == 0:
        f1 = 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

    reward = max(-1.0, min(1.0, (2 * f1) - 1 - (0.002 * inspection_steps)))
    
    return {
        "tp": tp, "fp": fp, "fn": fn, "f1": f1, "reward": reward
    }