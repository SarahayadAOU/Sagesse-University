"""
M1: TF-IDF (pure numpy, no sklearn) cosine similarity alignment
M2: BM25 (rank_bm25) probabilistic retrieval alignment
Runs in-process, no heavy library init.
"""

import json
import numpy as np
import re
import math
from collections import Counter
from rank_bm25 import BM25Okapi

STOPWORDS = set("a an the is are was were be been being have has had do does did will would could should may might shall can of in on at to for with by from as this that these those it its"
                .split())

def tokenize(text):
    tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]

def tfidf_matrix(docs):
    """Compute TF-IDF matrix (n_docs x vocab) as numpy array."""
    tokenized = [tokenize(d) for d in docs]
    # Build vocab
    vocab = sorted(set(t for doc in tokenized for t in doc))
    word2idx = {w: i for i, w in enumerate(vocab)}
    N = len(tokenized)
    V = len(vocab)

    # TF (log-normalized)
    tf = np.zeros((N, V), dtype=np.float32)
    for i, doc in enumerate(tokenized):
        if not doc:
            continue
        counts = Counter(doc)
        for w, cnt in counts.items():
            if w in word2idx:
                tf[i, word2idx[w]] = 1 + math.log(cnt)

    # IDF
    df = np.sum(tf > 0, axis=0).astype(np.float32)
    idf = np.log((N + 1) / (df + 1)) + 1.0  # sklearn-style smooth idf

    tfidf = tf * idf

    # L2-normalize
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tfidf = tfidf / norms

    return tfidf, vocab, word2idx

def cosine_sim_matrix(A, B):
    """Both A and B already L2-normalised."""
    return A @ B.T

def load_data():
    with open("/home/sandbox/experiments/commitments.json") as f:
        commitments = json.load(f)
    with open("/home/sandbox/experiments/indicators.json") as f:
        indicators = json.load(f)
    return commitments, indicators

def run_m1(commitments, indicators):
    comm_texts = [c["commitment"] + " " + " ".join(c["keywords"]) for c in commitments]
    ind_texts  = [ind["text"] for ind in indicators]

    all_texts = ind_texts + comm_texts
    tfidf, vocab, w2i = tfidf_matrix(all_texts)

    ind_vecs  = tfidf[:len(ind_texts)]
    comm_vecs = tfidf[len(ind_texts):]

    sim = cosine_sim_matrix(ind_vecs, comm_vecs)   # (30, 34)

    rankings = {}
    for i, ind in enumerate(indicators):
        row = [(commitments[j]["id"], float(sim[i, j])) for j in range(len(commitments))]
        row.sort(key=lambda x: -x[1])
        rankings[ind["id"]] = row
    return rankings, sim

def run_m2(commitments, indicators):
    corpus = [tokenize(c["commitment"] + " " + " ".join(c["keywords"])) for c in commitments]
    bm25 = BM25Okapi(corpus)

    rankings = {}
    score_matrix = np.zeros((len(indicators), len(commitments)))
    for i, ind in enumerate(indicators):
        query = tokenize(ind["text"])
        scores = bm25.get_scores(query)
        score_matrix[i] = scores
        row = [(commitments[j]["id"], float(scores[j])) for j in range(len(commitments))]
        row.sort(key=lambda x: -x[1])
        rankings[ind["id"]] = row
    return rankings, score_matrix

def main():
    commitments, indicators = load_data()
    print(f"Loaded {len(commitments)} commitments, {len(indicators)} indicators.", flush=True)

    print("Running M1 (TF-IDF)...", flush=True)
    m1_rankings, m1_scores = run_m1(commitments, indicators)

    print("Running M2 (BM25)...", flush=True)
    m2_rankings, m2_scores = run_m2(commitments, indicators)

    with open("/home/sandbox/experiments/m1_rankings.json", "w") as f:
        json.dump(m1_rankings, f, indent=2)
    with open("/home/sandbox/experiments/m2_rankings.json", "w") as f:
        json.dump(m2_rankings, f, indent=2)
    np.save("/home/sandbox/experiments/m1_scores.npy", m1_scores)
    np.save("/home/sandbox/experiments/m2_scores.npy", m2_scores)

    print("\n--- Full Top-5 (M1 | M2) ---")
    for ind in indicators:
        t1 = [cid for cid, _ in m1_rankings[ind["id"]][:5]]
        t2 = [cid for cid, _ in m2_rankings[ind["id"]][:5]]
        s1 = [f"{s:.3f}" for _, s in m1_rankings[ind["id"]][:5]]
        s2 = [f"{s:.3f}" for _, s in m2_rankings[ind["id"]][:5]]
        print(f"  {ind['id']} {ind['name'][:22]:<22} M1:{t1}({s1})  M2:{t2}({s2})")

    print("\nDone.")

if __name__ == "__main__":
    main()
