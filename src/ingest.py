# copies the gitbooks FAQ pages for No Hesi into the raw data, which can be found at data/raw

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ALLOWED_SPACE_DIRS = ["untitled"]
BLOCKED_WORDS = ["internal"]

# ensures the page is allowed to be put into data
def is_allowed(repo, page):
    try:
        relative = page.resolve().relative_to(repo.resolve())
    except ValueError:
        return False

    if relative.parts[0] not in ALLOWED_SPACE_DIRS:
        return False

    path_text = relative.as_posix().lower()
    for word in BLOCKED_WORDS:
        if word in path_text:
            return False

    return True

# lists all pages inside of gitbooks table of contents
def list_pages(repo):
    pages = []
    for space_dir in ALLOWED_SPACE_DIRS:
        space = repo / space_dir
        summary_text = (space / "SUMMARY.md").read_text(encoding="utf-8")

        for line in summary_text.splitlines():
            if "](" not in line or ".md)" not in line:
                continue
            link = line.split("](")[1].split(")")[0]
            page = space / link

            if is_allowed(repo, page):
                pages.append((space_dir, page))
            else:
                print("refused (not on allow-list):", link)
    return pages


# resolves the name
def page_route(space, page):
    relative = page.resolve().relative_to(space.resolve())
    parts = list(relative.with_suffix("").parts)
    if parts[-1].lower() == "readme":
        parts = parts[:-1]
    return "/".join(parts)

def page_slug(route):
    if route == "":
        return "index"
    return route.replace("/", "--")

def split_header(text):
    description = ""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            for line in text[4:end].splitlines():
                if line.startswith("description:"):
                    description = line[len("description:"):].strip()
            text = text[end + 5:]
    return description, text.lstrip("\n")

def page_title(body, fallback):
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


# builds the raw data
def ingest(repo, out_dir, site_url="", today=None):
    if today is None:
        today = date.today().isoformat()

    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("*.md"):
        old_file.unlink()

    written = []
    for space_dir, page in list_pages(repo):
        route = page_route(repo / space_dir, page)
        slug = page_slug(route)

        text = page.read_text(encoding="utf-8").replace("\r\n", "\n")
        description, body = split_header(text)

        if site_url:
            url = site_url.rstrip("/") + "/" + route
        else:
            url = ""

        header = "---\n"
        header += "title: " + page_title(body, slug) + "\n"
        header += "url: " + url + "\n"
        header += "source: " + page.resolve().relative_to(repo.resolve()).as_posix() + "\n"
        header += "description: " + description + "\n"
        header += "fetched: " + today + "\n"
        header += "---\n\n"

        out_path = out_dir / (slug + ".md")
        out_path.write_text(header + body, encoding="utf-8", newline="\n")
        written.append(out_path)

    return written

def main():
    load_dotenv(PROJECT_ROOT / ".env")
    repo_path = os.getenv("GITBOOK_REPO_PATH")
    site_url = os.getenv("FAQ_SITE_URL", "")

    if not repo_path or not Path(repo_path).is_dir():
        print("Set GITBOOK_REPO_PATH in .env to your local GitBook repo folder.")
        return
    if not site_url:
        print("Note: FAQ_SITE_URL is not set, so page headers will have an empty url.")

    print("Ingesting from", repo_path)
    written = ingest(Path(repo_path), RAW_DIR, site_url)
    print("Wrote", len(written), "pages to", RAW_DIR)


if __name__ == "__main__":
    main()
