# create chunks from the raw data
import html
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks.jsonl"

# can change later
MAX_WORDS = 300
MIN_WORDS = 30

def read_raw_page(path):
    text = path.read_text(encoding="utf-8")
    header = {}

    end = text.find("\n---\n", 4)
    for line in text[4:end].splitlines():
        key, _, value = line.partition(": ")
        header[key.strip(":")] = value
    body = text[end + 5:]
    return header, body


def clean_lines(body):
    kept = []
    for line in body.splitlines():
        bare = line.lstrip("> ").strip()
        if bare == "***":
            continue
        if bare.startswith("![") or bare.startswith("<figure") or bare.startswith("<table"):
            continue
        if bare.startswith("{%"):
            continue
        if line.strip() == ">":
            line = ""
        line = line.rstrip()
        if line.endswith("\\"):
            line = line[:-1].rstrip()
        line = html.unescape(line)
        kept.append(line)
    return kept


def clean_heading(text):
    return text.replace("*", "").replace("\\", "").strip()

def make_anchor(heading):
    anchor = ""
    for character in heading.lower():
        if character.isalnum() or character in ".-_":
            anchor += character
        elif character == " ":
            anchor += "-"
    while "--" in anchor:
        anchor = anchor.replace("--", "-")
    return anchor.strip("-.")


def split_into_sections(lines):
    sections = []
    h2 = None
    current = {"headings": [], "group": None, "anchor": "", "lines": []}

    for line in lines:
        if line.startswith("# "):
            continue
        if line.startswith("## ") or line.startswith("### "):
            sections.append(current)
            heading = clean_heading(line.lstrip("#"))
            if line.startswith("## "):
                h2 = heading
                path = [heading]
                group = None
            elif h2:
                path = [h2, heading]
                group = h2
            else:
                path = [heading]
                group = None
            current = {"headings": path, "group": group, "anchor": make_anchor(heading), "lines": []}
        else:
            current["lines"].append(line)

    sections.append(current)
    return sections


def section_text(lines):
    paragraphs = []
    for paragraph in "\n".join(lines).split("\n\n"):
        if paragraph.strip():
            paragraphs.append(paragraph.strip())
    return "\n\n".join(paragraphs)


def count_words(text):
    return len(text.split())


def split_long_text(text, max_words):
    pieces = []
    current = ""
    for paragraph in text.split("\n\n"):
        if current and count_words(current) + count_words(paragraph) > max_words:
            pieces.append(current)
            current = paragraph
        elif current:
            current += "\n\n" + paragraph
        else:
            current = paragraph
    if current:
        pieces.append(current)
    return pieces


def page_category(source):
    parts = source.split("/")
    if len(parts) >= 3:
        return parts[1]
    return "general"

def merge_short_sections(sections):
    merged = []
    for section in sections:
        if merged:
            previous = merged[-1]
            same_parent = previous["group"] == section["group"]
            previous_is_parent = section["group"] is not None and previous["headings"] == [section["group"]]
            same_group = same_parent or previous_is_parent
            too_short = count_words(section["text"]) < MIN_WORDS or count_words(previous["text"]) < MIN_WORDS
            if same_group and too_short:
                # Keep the merged section's own heading as a label, e.g. "Step 2".
                label = " > ".join(section["headings"])
                previous["text"] += "\n\n" + label + "\n" + section["text"]
                continue
        merged.append(section)
    return merged


def chunk_page(path):
    header, body = read_raw_page(path)
    title = header.get("title", path.stem)
    page_url = header.get("url", "")
    category = page_category(header.get("source", ""))

    sections = []
    for section in split_into_sections(clean_lines(body)):
        section["text"] = section_text(section["lines"])
        if section["text"]:
            sections.append(section)

    sections = merge_short_sections(sections)

    chunks = []
    for section in sections:
        heading_path = " > ".join([title] + section["headings"])

        if section["anchor"] and page_url:
            url = page_url + "#" + section["anchor"]
        else:
            url = page_url

        room = MAX_WORDS - count_words(heading_path)
        for piece in split_long_text(section["text"], room):
            chunks.append({
                "page_title": title,
                "heading_path": heading_path,
                "category": category,
                "url": url,
                "text": heading_path + "\n\n" + piece,
            })

    for number, chunk in enumerate(chunks, start=1):
        chunk["id"] = path.stem + "-" + str(number)
    return chunks


def print_stats(chunks, samples=10):
    lengths = [count_words(chunk["text"]) for chunk in chunks]
    print("Chunks:  ", len(chunks))
    print("Shortest:", min(lengths), "words")
    print("Longest: ", max(lengths), "words")
    print("Average: ", round(sum(lengths) / len(lengths)), "words")

    for chunk in random.sample(chunks, min(samples, len(chunks))):
        print("\n" + "=" * 70)
        print(chunk["id"], "|", count_words(chunk["text"]), "words |", chunk["url"])
        print("-" * 70)
        print(chunk["text"])


def main():
    all_chunks = []
    for path in sorted(RAW_DIR.glob("*.md")):
        all_chunks.extend(chunk_page(path))

    if not all_chunks:
        print("No pages found in", RAW_DIR, "- run src/ingest.py first.")
        return

    with open(CHUNKS_FILE, "w", encoding="utf-8", newline="\n") as file:
        for chunk in all_chunks:
            row = {
                "id": chunk["id"],
                "page_title": chunk["page_title"],
                "heading_path": chunk["heading_path"],
                "category": chunk["category"],
                "url": chunk["url"],
                "text": chunk["text"],
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Wrote", len(all_chunks), "chunks to", CHUNKS_FILE)
    print_stats(all_chunks)


if __name__ == "__main__":
    main()