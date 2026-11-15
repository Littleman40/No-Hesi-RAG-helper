import pytest

from retrieve import rrf


def test_rrf_rewards_agreement():
    fused = rrf([["a", "b", "x"], ["c", "b", "y"]])
    assert [chunk_id for chunk_id, _ in fused] == ["b", "a", "c", "x", "y"]


def test_rrf_scores():
    scores = dict(rrf([["a", "b"], ["b"]]))
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_single_list():
    assert [chunk_id for chunk_id, _ in rrf([["x", "y", "z"]])] == ["x", "y", "z"]
