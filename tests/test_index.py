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
