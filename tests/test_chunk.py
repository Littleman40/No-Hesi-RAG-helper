import chunk


def make_page(tmp_path, body, name="error-messages--test-page.md"):
    header = (
        "---\n"
        "title: Test Page\n"
        "url: https://faq.example.com/error-messages/test-page\n"
        "source: untitled/error-messages/test-page.md\n"
        "description: A test\n"
        "fetched: 2026-11-13\n"
        "---\n\n"
    )
    path = tmp_path / name
    path.write_text(header + "# Test Page\n\n" + body, encoding="utf-8")
    return path


def words(n):
    return " ".join(["word"] * n)


def test_make_anchor():
    assert chunk.make_anchor("Step 2.1") == "step-2.1"
    assert chunk.make_anchor("Method 1: Reinstall") == "method-1-reinstall"
    assert chunk.make_anchor("**Step 1:**") == "step-1"


def test_cuts_at_headings_and_fills_metadata(tmp_path):
    body = "## Method 1\n\n" + words(40) + "\n\n### Step 1\n\n" + words(40) + "\n"
    chunks = chunk.chunk_page(make_page(tmp_path, body))

    assert len(chunks) == 2
    first, second = chunks
    assert first["id"] == "error-messages--test-page-1"
    assert first["page_title"] == "Test Page"
    assert first["category"] == "error-messages"
    assert second["heading_path"] == "Test Page > Method 1 > Step 1"
    assert second["url"] == "https://faq.example.com/error-messages/test-page#step-1"
    assert second["text"].startswith("Test Page > Method 1 > Step 1\n\n")


def test_short_sections_are_merged(tmp_path):
    body = "## Step 1\n\nGo to Settings\n\n## Step 2\n\nClick Plugins\n\n## Step 3\n\n" + words(40) + "\n"
    chunks = chunk.chunk_page(make_page(tmp_path, body))

    assert len(chunks) == 1
    assert "Go to Settings" in chunks[0]["text"]
    assert "Step 2\nClick Plugins" in chunks[0]["text"]


def test_no_merging_across_different_methods(tmp_path):
    body = (
        "## Method 1\n\n### Step 1\n\n" + words(40) + "\n\n"
        "## Method 2\n\n### Step 1\n\nShort step\n"
    )
    chunks = chunk.chunk_page(make_page(tmp_path, body))

    assert len(chunks) == 2
    assert "Short step" not in chunks[0]["text"]
    assert chunks[1]["heading_path"] == "Test Page > Method 2 > Step 1"


def test_long_sections_are_split(tmp_path):
    paragraphs = ["> " + words(100) for _ in range(5)]
    body = "## Big FAQ\n\n" + "\n>\n".join(paragraphs) + "\n"
    chunks = chunk.chunk_page(make_page(tmp_path, body))

    assert len(chunks) > 1
    for piece in chunks:
        assert chunk.count_words(piece["text"]) <= chunk.MAX_WORDS
        assert piece["text"].startswith("Test Page > Big FAQ\n\n")


def test_noise_is_removed(tmp_path):
    body = (
        "## Fix\n\n" + words(40) + "\n\n***\n\n"
        "![screenshot.png](https://example.com/screenshot.png)\n\n"
        '<figure><img src="x.png"></figure>\n\n'
        "{% hint %}\n\nKeep this&#x20;line\n"
    )
    text = chunk.chunk_page(make_page(tmp_path, body))[0]["text"]

    assert "***" not in text
    assert "screenshot" not in text
    assert "<figure" not in text
    assert "{%" not in text
    assert "Keep this line" in text
