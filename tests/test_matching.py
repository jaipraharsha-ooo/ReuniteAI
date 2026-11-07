import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import pytest
from src.matching_engine import TfidfMatcher, KeywordBaseline

R = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "found_items.csv"))


def test_empty_query_rejected():
    with pytest.raises(ValueError):
        TfidfMatcher(R).search("   ")
    with pytest.raises(ValueError):
        KeywordBaseline(R).search("")


def test_sorted_descending():
    res = TfidfMatcher(R).search("blue water bottle", top_k=5)
    assert res["score"].is_monotonic_decreasing


def test_obvious_match_first():
    assert TfidfMatcher(R).search("silver pen drive red cap").iloc[0]["report_id"] == "F-013"


def test_nomatch_low_score():
    assert TfidfMatcher(R).search("gaming mouse rgb").iloc[0]["score"] < 0.15


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        TfidfMatcher(pd.concat([R, R.iloc[[0]]]))
