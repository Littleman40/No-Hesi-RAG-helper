import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

import ingest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def fake_repo(tmp_path):
    repo = tmp_path / "gitbook"
    write(
        repo / "untitled" / "SUMMARY.md",
        "# Table of contents\n\n"
        "* [Home](README.md)\n"
        "* [Guides](guides/README.md)\n"
        "  * [Fix: crash](guides/fix-crash.md)\n"
        "* [Sneaky](../internal-space/staff-notes.md)\n"
        "* [Staff only](staff-internal-notes.md)\n",
    )
    write(repo / "untitled" / "README.md", "---\ndescription: Welcome\n---\n\n# Home\n\nHello.\n")
    write(repo / "untitled" / "guides" / "README.md", "# Guides\n")
    write(repo / "untitled" / "guides" / "fix-crash.md", "# Fix: crash\n\nRestart.\n")
    write(repo / "untitled" / "staff-internal-notes.md", "# Staff\n\nSECRET\n")
    write(repo / "internal-space" / "staff-notes.md", "# Internal\n\nSECRET\n")
    return repo


def test_allow_list_accepts_public_pages(fake_repo):
    assert ingest.is_allowed(fake_repo, fake_repo / "untitled" / "guides" / "fix-crash.md")


@pytest.mark.parametrize(
    "rel",
    [
        "internal-space/staff-notes.md",
        "untitled/staff-internal-notes.md",
        "untitled/../internal-space/staff-notes.md",
        "../outside.md",
    ],
)
def test_allow_list_refuses_everything_else(fake_repo, rel):
    assert not ingest.is_allowed(fake_repo, fake_repo / rel)


def test_ingest_writes_only_public_pages(fake_repo, tmp_path):
    out = tmp_path / "raw"
    written = ingest.ingest(fake_repo, out, site_url="https://faq.example.com", today="2026-11-13")

    assert sorted(p.name for p in written) == ["guides--fix-crash.md", "guides.md", "index.md"]
    for path in written:
        text = path.read_text(encoding="utf-8")
        assert "SECRET" not in text
        assert "internal" not in text.lower()


def test_header_has_title_url_and_date(fake_repo, tmp_path):
    out = tmp_path / "raw"
    ingest.ingest(fake_repo, out, site_url="https://faq.example.com/", today="2026-11-13")

    page = (out / "guides--fix-crash.md").read_text(encoding="utf-8")
    assert page.startswith("---\n")
    assert "title: Fix: crash\n" in page
    assert "url: https://faq.example.com/guides/fix-crash\n" in page
    assert "fetched: 2026-11-13\n" in page
    assert "Restart." in page

    home = (out / "index.md").read_text(encoding="utf-8")
    assert "url: https://faq.example.com/\n" in home
    assert "description: Welcome\n" in home


def test_rerun_removes_deleted_pages(fake_repo, tmp_path):
    out = tmp_path / "raw"
    write(out / ".gitkeep", "")
    ingest.ingest(fake_repo, out)
    assert (out / "guides--fix-crash.md").exists()

    summary = fake_repo / "untitled" / "SUMMARY.md"
    summary.write_text(summary.read_text().replace("  * [Fix: crash](guides/fix-crash.md)\n", ""))
    ingest.ingest(fake_repo, out)

    assert not (out / "guides--fix-crash.md").exists()
    assert (out / ".gitkeep").exists()


def test_real_repo_has_no_internal_pages():
    load_dotenv(PROJECT_ROOT / ".env")
    repo_path = os.getenv("GITBOOK_REPO_PATH")
    if not repo_path or not Path(repo_path).is_dir():
        pytest.skip("GITBOOK_REPO_PATH not set; skipping real-repo check")

    repo = Path(repo_path)
    pages = ingest.list_pages(repo)
    assert pages, "no pages found; is GITBOOK_REPO_PATH pointing at the right folder?"
    for space_dir, page in pages:
        assert ingest.is_allowed(repo, page)
        assert "internal" not in page.resolve().relative_to(repo.resolve()).as_posix().lower()
