import numpy as np

import index


def test_tokenize_basic():
    assert index.tokenize("How do I FIX this?") == ["how", "do", "i", "fix", "this"]


def test_tokenize_keeps_codes_whole():
    assert index.tokenize("Error d3dx11_43.dll on CSP 0.2.11") == ["error", "d3dx11_43.dll", "on", "csp", "0.2.11"]
    assert index.tokenize("Code AC-1234.") == ["code", "ac-1234"]
    assert index.tokenize("Error d3dx11\\_43.dll") == ["error", "d3dx11_43.dll"]


def test_tokenize_drops_markdown():
    assert index.tokenize("> **Checksum failed**") == ["checksum", "failed"]


def test_normalize():
    vectors = index.normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
    assert np.allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])
    assert np.allclose(vectors[0], [0.6, 0.8])


def test_tokenize_joins_apostrophes():
    assert index.tokenize("doesn't doesnt does'nt doesn’t") == ["doesnt", "doesnt", "doesnt", "doesnt"]


def test_search_text_adds_alias_for_its_page_only():
    alias_page = next(iter(index.SEARCH_ALIASES))
    assert index.SEARCH_ALIASES[alias_page] in index.search_text({"id": alias_page + "-2", "text": "body"})
    assert index.search_text({"id": "other-page-1", "text": "body"}) == "body"
