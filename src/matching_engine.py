import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text_processing import validate_reports, build_text, tokenize

TFIDF_THRESHOLD = 0.15   # below this, report "no strong match" (to be justified by evaluation)
BASELINE_THRESHOLD = 0.5


class TfidfMatcher:
    def __init__(self, reports: pd.DataFrame):
        self.reports = validate_reports(reports)
        self.texts = [build_text(r) for r in self.reports.to_dict("records")]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, top_k: int = 3, category: str | None = None) -> pd.DataFrame:
        if not query or not query.strip():
            raise ValueError("Query must not be empty")
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        out = self.reports.copy()
        out["score"] = scores
        if category and category != "Any":
            out = out[out["category"].str.lower() == category.lower()]
        return out.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)


class KeywordBaseline:
    """Score = fraction of (non-stopword) query terms that appear in the report text."""

    def __init__(self, reports: pd.DataFrame):
        self.reports = validate_reports(reports)
        self.token_sets = [set(tokenize(build_text(r))) for r in self.reports.to_dict("records")]

    def search(self, query: str, top_k: int = 3) -> pd.DataFrame:
        if not query or not query.strip():
            raise ValueError("Query must not be empty")
        q = set(tokenize(query))
        scores = [len(q & t) / len(q) if q else 0.0 for t in self.token_sets]
        out = self.reports.copy()
        out["score"] = scores
        return out.sort_values("score", ascending=False, kind="stable").head(top_k).reset_index(drop=True)
