import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import run_eval


def test_first_gold_rank():
    assert run_eval.first_gold_rank(["a", "b", "c"], ["c"]) == 3
    assert run_eval.first_gold_rank(["a", "b", "c"], ["z", "b"]) == 2
    assert run_eval.first_gold_rank(["a"], ["z"]) is None
    assert run_eval.first_gold_rank([], ["z"]) is None


def test_recall_and_mrr():
    ranks = [1, 2, 4, None]
    assert run_eval.recall_at(ranks, 1) == 0.25
    assert run_eval.recall_at(ranks, 3) == 0.5
    assert run_eval.recall_at(ranks, 5) == 0.75
    assert run_eval.mrr(ranks) == pytest.approx((1 + 0.5 + 0.25 + 0) / 4)
