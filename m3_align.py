"""
M3: Claude zero-shot alignment
For each indicator, ask Claude to rank the top-5 most relevant commitments.
Uses batches of 5 indicators per API call to minimize cost.
"""

import anthropic
import json
import os
import time

API_KEY = os.environ.get("CLAUDE_API_KEY", "")
client = anthropic.Anthropic(api_key=API_KEY)
MODEL = "claude-haiku-4-5-20251001"

def load_data():
    with open("/home/sandbox/experiments/commitments.json") as f:
        commitments = json.load(f)
    with open("/home/sandbox/experiments/indicators.json") as f:
        indicators = json.load(f)
    return commitments, indicators

def build_commitment_catalog(commitments):
    lines = []
    for c in commitments:
        lines.append(f'{c["id"]}: {c["commitment"]}')
    return "\n".join(lines)

SYSTEM_PROMPT = """You are an expert in higher-education quality assurance and HR alignment.
Your task: given a list of institutional commitments extracted from a Self-Evaluation Report (SER),
and a set of employee-satisfaction survey indicators, identify the TOP-5 most relevant commitments
for each indicator.

Relevance means: the commitment directly addresses, measures, or reflects the construct being
assessed by the indicator. Consider both explicit keyword overlap AND semantic/conceptual alignment.

Output ONLY a valid JSON array (no markdown, no explanation) in this format:
[
  {"indicator_id": "I01", "top5": ["C27","C03","C32","C29","C28"], "reasoning": "brief 1-line rationale"},
  ...
]
"""

def align_batch(indicators_batch, commitment_catalog):
    """Align a batch of indicators to commitments using Claude."""
    ind_lines = "\n".join([f'{ind["id"]}: {ind["text"]}' for ind in indicators_batch])

    user_msg = f"""COMMITMENT CATALOG:
{commitment_catalog}

SURVEY INDICATORS (align each to top-5 commitments):
{ind_lines}

Return a JSON array with one object per indicator."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        start = raw.find("[")
        end = raw.rfind("]") + 1
        raw = raw[start:end]
    return json.loads(raw)

def main():
    commitments, indicators = load_data()
    commitment_catalog = build_commitment_catalog(commitments)

    print(f"Running M3 on {len(indicators)} indicators in batches of 6...", flush=True)

    results = []
    batch_size = 6
    for i in range(0, len(indicators), batch_size):
        batch = indicators[i:i+batch_size]
        ids = [b["id"] for b in batch]
        print(f"  Batch {i//batch_size+1}: {ids}", flush=True)
        batch_results = align_batch(batch, commitment_catalog)
        results.extend(batch_results)
        time.sleep(0.5)  # Rate-limit courtesy

    # Build rankings dict: {indicator_id: [(cid, rank_score), ...]}
    # We use 5,4,3,2,1 as scores for ranks 1-5, and 0 for unranked
    m3_rankings = {}
    for item in results:
        iid = item["indicator_id"]
        top5 = item.get("top5", [])
        ranked = [(top5[j], 5-j) for j in range(len(top5))]
        # Pad with zeros for all other commitments
        ranked_dict = {cid: s for cid, s in ranked}
        full_ranked = [(c["id"], ranked_dict.get(c["id"], 0)) for c in commitments]
        full_ranked.sort(key=lambda x: -x[1])
        m3_rankings[iid] = full_ranked

    with open("/home/sandbox/experiments/m3_rankings.json", "w") as f:
        json.dump(m3_rankings, f, indent=2)

    # Also save the raw results with reasoning
    with open("/home/sandbox/experiments/m3_raw.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- M3 Top-5 ---")
    for item in results:
        t = item.get("top5", [])
        r = item.get("reasoning", "")
        print(f"  {item['indicator_id']}: {t}  | {r[:70]}")

    print("\nM3 done.")

if __name__ == "__main__":
    main()
