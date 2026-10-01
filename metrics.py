"""
Simulates the human-annotation step and computes all evaluation metrics.
Annotator 1: Head of Law Department (formal/policy-oriented)
Annotator 2: VP of the University (strategic/holistic)
IMPORTANT: These annotations are SIMULATED for paper-development purposes.
           They must be replaced with real expert judgments before final submission.
"""

import json
import numpy as np
import math
from collections import defaultdict

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ── Gold relevance: logical relevance sets per indicator ─────────────────────
# Based on semantic analysis of commitment text vs. indicator construct.
# Each set = commitment IDs that genuinely address the indicator's construct.
LOGICAL_RELEVANCE = {
    "I01": {"C27", "C32", "C14", "C09", "C17"},          # salary fairness
    "I02": {"C27", "C16", "C32", "C13", "C09"},           # salary competitiveness
    "I03": {"C27", "C06", "C03", "C32", "C08"},           # pay transparency
    "I04": {"C28", "C27", "C29", "C21"},                  # health coverage
    "I05": {"C30", "C31", "C29", "C28", "C23"},           # non-monetary benefits
    "I06": {"C29", "C32", "C27", "C31"},                  # end of service
    "I07": {"C02", "C01", "C03", "C05", "C07"},           # strategic clarity
    "I08": {"C05", "C08", "C06", "C07", "C01"},           # decision transparency
    "I09": {"C01", "C02", "C03"},                         # participation in decisions
    "I10": {"C01", "C10", "C17", "C13"},                  # empowerment
    "I11": {"C08", "C11", "C05"},                         # responsiveness
    "I12": {"C06", "C08", "C05", "C20"},                  # workflow efficiency
    "I13": {"C01", "C08", "C02"},                         # conflict resolution
    "I14": {"C06", "C05", "C08", "C07"},                  # policy communication
    "I15": {"C05", "C07", "C08", "C06"},                  # clarity of communication
    "I16": {"C21", "C13", "C11"},                         # workspace comfort
    "I17": {"C24", "C20", "C33", "C12"},                  # access to equipment/tech
    "I18": {"C17", "C14", "C32", "C19"},                  # promotion clarity
    "I19": {"C20", "C33", "C11", "C34"},                  # training effectiveness
    "I20": {"C23", "C34", "C26", "C12"},                  # extra-role involvement
    "I21": {"C32", "C19", "C17", "C27"},                  # recognition
    "I22": {"C25", "C22", "C31"},                         # flexibility
    "I23": {"C21", "C22", "C25", "C23"},                  # work-life balance
    "I24": {"C21", "C22", "C25", "C20"},                  # burnout risk
    "I25": {"C01", "C02", "C07", "C13"},                  # collegiality
    "I26": {"C01", "C02", "C03", "C08"},                  # psychological safety
    "I27": {"C01", "C04", "C16", "C05"},                  # diversity and inclusion
    "I28": {"C02", "C03", "C32", "C13"},                  # sense of belonging
    "I29": {"C27", "C28", "C30", "C29"},                  # family enrollment likelihood
    "I30": {"C04", "C09", "C26", "C05", "C06"},           # brand identity
}

def load_all():
    with open("/home/sandbox/experiments/commitments.json") as f:
        commitments = json.load(f)
    with open("/home/sandbox/experiments/indicators.json") as f:
        indicators = json.load(f)
    with open("/home/sandbox/experiments/m1_rankings.json") as f:
        m1 = json.load(f)
    with open("/home/sandbox/experiments/m2_rankings.json") as f:
        m2 = json.load(f)
    with open("/home/sandbox/experiments/m3_rankings.json") as f:
        m3 = json.load(f)
    return commitments, indicators, m1, m2, m3

def get_top5(rankings, ind_id):
    return [cid for cid, _ in rankings[ind_id][:5]]

def build_annotation_set(indicators, m1, m2, m3):
    """Union of top-5 from each method per indicator."""
    pairs = set()
    for ind in indicators:
        iid = ind["id"]
        for method_rankings in [m1, m2, m3]:
            for cid in get_top5(method_rankings, iid):
                pairs.add((iid, cid))
    return sorted(pairs)

def simulate_annotator(pairs, relevance, noise_prob_tp=0.05, noise_prob_tn=0.20, name="A"):
    """
    Simulate one annotator's judgments.
    True positives (in logical relevance) flipped with prob noise_prob_tp.
    True negatives flipped with prob noise_prob_tn (annotator sees soft relevance).
    """
    judgments = {}
    for (iid, cid) in pairs:
        is_relevant = cid in relevance.get(iid, set())
        if is_relevant:
            label = 0 if rng.random() < noise_prob_tp else 1
        else:
            label = 1 if rng.random() < noise_prob_tn else 0
        judgments[(iid, cid)] = label
    return judgments

def cohen_kappa(a1, a2, pairs):
    labels1 = [a1[p] for p in pairs]
    labels2 = [a2[p] for p in pairs]
    n = len(labels1)

    agree = sum(l1 == l2 for l1, l2 in zip(labels1, labels2))
    p_o = agree / n

    p1_pos = sum(labels1) / n
    p2_pos = sum(labels2) / n
    p_e = p1_pos * p2_pos + (1 - p1_pos) * (1 - p2_pos)

    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0
    return round(kappa, 4), round(p_o, 4)

def krippendorff_alpha(a1, a2, pairs):
    """Ordinal/nominal Krippendorff's alpha (binary, nominal)."""
    labels1 = [a1[p] for p in pairs]
    labels2 = [a2[p] for p in pairs]
    n = len(labels1)

    # Observed disagreement
    D_o = sum(l1 != l2 for l1, l2 in zip(labels1, labels2)) / (n if n > 0 else 1)

    # Expected disagreement from marginals
    all_labels = labels1 + labels2
    n_total = len(all_labels)
    freq = {0: all_labels.count(0), 1: all_labels.count(1)}
    D_e = 0.0
    for v in [0, 1]:
        for w in [0, 1]:
            if v != w:
                D_e += (freq[v] / n_total) * (freq[w] / n_total)
    alpha = 1 - D_o / D_e if D_e > 0 else 1.0
    return round(alpha, 4)

def build_gold(a1, a2, pairs):
    """Gold = 1 if BOTH annotators say 1."""
    return {p: 1 if (a1[p] == 1 and a2[p] == 1) else 0 for p in pairs}

def precision_at_k(gold, rankings, indicator_ids, k=5):
    """Mean P@k over all indicators."""
    scores = []
    for iid in indicator_ids:
        topk = [cid for cid, _ in rankings[iid][:k]]
        relevant_in_topk = sum(gold.get((iid, cid), 0) for cid in topk)
        scores.append(relevant_in_topk / k)
    return round(np.mean(scores), 4)

def average_precision(gold, ranked_list, iid, k=5):
    """AP for one indicator."""
    hits = 0
    sum_prec = 0.0
    for i, (cid, _) in enumerate(ranked_list[:k]):
        if gold.get((iid, cid), 0) == 1:
            hits += 1
            sum_prec += hits / (i + 1)
    n_relevant = sum(gold.get((iid, cid), 0) for cid, _ in ranked_list)
    if n_relevant == 0:
        return 0.0
    return sum_prec / min(n_relevant, k)

def map_score(gold, rankings, indicator_ids, k=5):
    aps = [average_precision(gold, rankings[iid], iid, k) for iid in indicator_ids]
    return round(np.mean(aps), 4)

def ndcg_at_k(gold, rankings, indicator_ids, k=5):
    """Mean nDCG@k."""
    scores = []
    for iid in indicator_ids:
        ranked_list = rankings[iid][:k]
        dcg = sum(gold.get((iid, cid), 0) / math.log2(i + 2) for i, (cid, _) in enumerate(ranked_list))
        # Ideal DCG: all relevant items at top
        n_rel = min(sum(gold.get((iid, cid), 0) for cid, _ in rankings[iid]), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return round(np.mean(scores), 4)

def main():
    commitments, indicators, m1, m2, m3 = load_all()
    all_ids = [ind["id"] for ind in indicators]

    # Build annotation pairs
    pairs = build_annotation_set(indicators, m1, m2, m3)
    print(f"Total annotation pairs: {len(pairs)}", flush=True)

    # Simulate annotators
    # A1 (Head of Law): more conservative on positives
    a1 = simulate_annotator(pairs, LOGICAL_RELEVANCE, noise_prob_tp=0.08, noise_prob_tn=0.25)
    # A2 (VP): slightly more liberal / holistic
    a2 = simulate_annotator(pairs, LOGICAL_RELEVANCE, noise_prob_tp=0.05, noise_prob_tn=0.30)

    # Inter-annotator agreement
    kappa, p_obs = cohen_kappa(a1, a2, pairs)
    alpha = krippendorff_alpha(a1, a2, pairs)

    agree_pos = sum(1 for p in pairs if a1[p] == 1 and a2[p] == 1)
    agree_neg = sum(1 for p in pairs if a1[p] == 0 and a2[p] == 0)
    disagree  = sum(1 for p in pairs if a1[p] != a2[p])
    a1_pos = sum(a1.values())
    a2_pos = sum(a2.values())

    print(f"\n=== Inter-Annotator Agreement ===")
    print(f"  Total pairs annotated:  {len(pairs)}")
    print(f"  A1 positive labels:     {a1_pos}")
    print(f"  A2 positive labels:     {a2_pos}")
    print(f"  Both positive:          {agree_pos}")
    print(f"  Both negative:          {agree_neg}")
    print(f"  Disagreed:              {disagree}")
    print(f"  Cohen's κ:              {kappa}")
    print(f"  Krippendorff's α:       {alpha}")
    print(f"  P(observed agreement):  {p_obs}")

    # Gold standard
    gold = build_gold(a1, a2, pairs)
    gold_positive = sum(gold.values())
    print(f"\n  Gold positives (both=1): {gold_positive} of {len(pairs)} ({100*gold_positive/len(pairs):.1f}%)")

    # ── Compute metrics ──────────────────────────────────────────────────────
    print(f"\n=== Evaluation Metrics ===")
    for k in [1, 3, 5]:
        m1_p = precision_at_k(gold, m1, all_ids, k)
        m2_p = precision_at_k(gold, m2, all_ids, k)
        m3_p = precision_at_k(gold, m3, all_ids, k)
        print(f"  P@{k}:  M1={m1_p}  M2={m2_p}  M3={m3_p}")

    m1_map = map_score(gold, m1, all_ids, 5)
    m2_map = map_score(gold, m2, all_ids, 5)
    m3_map = map_score(gold, m3, all_ids, 5)
    print(f"  MAP:   M1={m1_map}  M2={m2_map}  M3={m3_map}")

    m1_ndcg = ndcg_at_k(gold, m1, all_ids, 5)
    m2_ndcg = ndcg_at_k(gold, m2, all_ids, 5)
    m3_ndcg = ndcg_at_k(gold, m3, all_ids, 5)
    print(f"  nDCG@5: M1={m1_ndcg}  M2={m2_ndcg}  M3={m3_ndcg}")

    # ── Coverage analysis ─────────────────────────────────────────────────────
    # How many distinct commitments does each survey dimension cover (at P@5)?
    dim_map = {}
    for ind in indicators:
        dim_map.setdefault(ind["dim"], []).append(ind["id"])

    print(f"\n=== Theme Coverage (M3 gold-relevant commitments per dimension) ===")
    coverage = {}
    for dim, iids in sorted(dim_map.items()):
        covered_themes = set()
        covered_cids = set()
        for iid in iids:
            for cid, _ in m3[iid][:5]:
                if gold.get((iid, cid), 0) == 1:
                    covered_cids.add(cid)
        # Map CIDs to themes
        cid_to_theme = {c["id"]: c["theme"] for c in commitments}
        covered_themes = {cid_to_theme[cid] for cid in covered_cids}
        coverage[dim] = {"cids": sorted(covered_cids), "themes": sorted(covered_themes), "n": len(covered_cids)}
        print(f"  {dim}: {len(covered_cids)} commitments / {len(covered_themes)} themes: {sorted(covered_themes)}")

    # ── Save all results ──────────────────────────────────────────────────────
    results = {
        "annotation": {
            "total_pairs": len(pairs),
            "a1_positive": int(a1_pos),
            "a2_positive": int(a2_pos),
            "both_positive": int(agree_pos),
            "both_negative": int(agree_neg),
            "disagreed": int(disagree),
            "cohen_kappa": float(kappa),
            "krippendorff_alpha": float(alpha),
            "p_observed": float(p_obs),
            "gold_positives": int(gold_positive)
        },
        "metrics": {
            "M1_TF-IDF":  {"P@1": precision_at_k(gold, m1, all_ids, 1),
                           "P@3": precision_at_k(gold, m1, all_ids, 3),
                           "P@5": precision_at_k(gold, m1, all_ids, 5),
                           "MAP": m1_map, "nDCG@5": m1_ndcg},
            "M2_BM25":    {"P@1": precision_at_k(gold, m2, all_ids, 1),
                           "P@3": precision_at_k(gold, m2, all_ids, 3),
                           "P@5": precision_at_k(gold, m2, all_ids, 5),
                           "MAP": m2_map, "nDCG@5": m2_ndcg},
            "M3_Claude":  {"P@1": precision_at_k(gold, m3, all_ids, 1),
                           "P@3": precision_at_k(gold, m3, all_ids, 3),
                           "P@5": precision_at_k(gold, m3, all_ids, 5),
                           "MAP": m3_map, "nDCG@5": m3_ndcg},
        },
        "coverage": {dim: v for dim, v in coverage.items()},
        "gold_pairs": {str(k): int(v) for k, v in gold.items()},
        "a1_judgments": {str(k): int(v) for k, v in a1.items()},
        "a2_judgments": {str(k): int(v) for k, v in a2.items()},
    }

    with open("/home/sandbox/experiments/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results saved to /home/sandbox/experiments/results.json")

if __name__ == "__main__":
    main()
