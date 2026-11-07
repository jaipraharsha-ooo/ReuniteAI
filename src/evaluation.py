import pandas as pd

from .matching_engine import TfidfMatcher, KeywordBaseline, TFIDF_THRESHOLD, BASELINE_THRESHOLD


def evaluate(reports_path="data/found_items.csv", cases_path="data/evaluation_cases.csv", k=3):
    reports = pd.read_csv(reports_path)
    cases = pd.read_csv(cases_path)
    methods = {
        "Keyword baseline": (KeywordBaseline(reports), BASELINE_THRESHOLD),
        "TF-IDF + cosine": (TfidfMatcher(reports), TFIDF_THRESHOLD),
    }
    rows, details = [], []
    for name, (engine, thr) in methods.items():
        hits = rr_sum = n_match = n_nomatch = false_pos = 0
        for _, c in cases.iterrows():
            ranked = engine.search(c.query_text, top_k=len(reports))
            ids = ranked["report_id"].tolist()
            top_score = float(ranked["score"].iloc[0])
            if c.expected_report_id == "NO_MATCH":
                n_nomatch += 1
                fp = top_score >= thr
                false_pos += fp
                details.append((name, c.query_id, c.scenario_type, "NO_MATCH", ids[0], round(top_score, 3), "false positive" if fp else "correct no-match"))
            else:
                n_match += 1
                rank = ids.index(c.expected_report_id) + 1
                hits += rank <= k
                rr_sum += 1 / rank
                details.append((name, c.query_id, c.scenario_type, c.expected_report_id, ids[0], round(top_score, 3), f"rank {rank}"))
        rows.append({
            "Method": name,
            "Match queries": n_match,
            f"Hit Rate@{k}": round(hits / n_match, 3),
            "MRR": round(rr_sum / n_match, 3),
            "No-match queries": n_nomatch,
            "No-match false positives": false_pos,
        })
    return pd.DataFrame(rows), pd.DataFrame(details, columns=["method", "query_id", "scenario", "expected", "top1", "top1_score", "outcome"])


if __name__ == "__main__":
    summary, details = evaluate()
    print(summary.to_string(index=False))
    details.to_csv("data/evaluation_details.csv", index=False)
    print("\nMisses / false positives (TF-IDF):")
    d = details[(details.method == "TF-IDF + cosine") & (~details.outcome.isin(["rank 1", "correct no-match"]))]
    print(d.to_string(index=False))
