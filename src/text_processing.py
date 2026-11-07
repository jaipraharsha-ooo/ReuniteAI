import re
import pandas as pd

REQUIRED_COLUMNS = ["report_id", "item_name", "category", "description"]
STOPWORDS = {
    "a", "an", "the", "i", "my", "me", "with", "and", "of", "on", "in", "at", "it",
    "is", "was", "to", "for", "lost", "left", "near", "found", "has", "have", "had",
}


def validate_reports(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df["report_id"].duplicated().any():
        raise ValueError("Duplicate report_id values found")
    df = df.fillna("")
    if (df["description"].str.strip() == "").any():
        raise ValueError("Empty description found")
    return df.reset_index(drop=True)


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return [w for w in words if w not in STOPWORDS]


def build_text(row) -> str:
    """Searchable text of a report: item name + category + description (+ location)."""
    parts = [row.get("item_name", ""), row.get("category", ""), row.get("description", "")]
    return " ".join(str(p) for p in parts if p)
