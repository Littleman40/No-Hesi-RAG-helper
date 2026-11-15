# label questions

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import retrieve

QUESTIONS_FILE = PROJECT_ROOT / "eval" / "questions.jsonl"


def load_questions():
    if not QUESTIONS_FILE.exists():
        return []
    with open(QUESTIONS_FILE, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def candidates(text, n=8):
    bm25_ids = [chunk_id for chunk_id, _ in retrieve.bm25_search(text, 20)]
    vector_ids = [chunk_id for chunk_id, _ in retrieve.vector_search(text, 20)]
    return [chunk_id for chunk_id, _ in retrieve.rrf([bm25_ids, vector_ids])[:n]]


def next_id_and_split(existing, question_type):
    same_type = sum(1 for q in existing if q["type"] == question_type)
    return f"q{len(existing) + 1:03d}", "dev" if same_type % 2 == 0 else "test"


def label(question, existing):
    search_text = question
    while True:
        ids = candidates(search_text)
        for number, chunk_id in enumerate(ids, start=1):
            chunk = retrieve.get_chunk(chunk_id)
            preview = chunk["text"].split("\n\n", 1)[-1][:90].replace("\n", " ")
            print(f"  [{number}] {chunk['heading_path']}\n       {preview}")

        choice = input("Chunks (e.g. 1 3) / n = not in FAQ / e = human / s <words> / Enter = skip: ").strip()
        if not choice:
            return None
        if choice.startswith("s "):
            search_text = choice[2:]
            continue
        if choice == "n":
            question_type, gold = "out_of_scope", []
        elif choice == "e":
            question_type, gold = "escalate", []
        else:
            try:
                gold = [ids[int(number) - 1] for number in choice.split()]
            except (ValueError, IndexError):
                print("  Didn't understand that, try again.")
                continue
            question_type = "in_scope"

        question_id, split = next_id_and_split(existing, question_type)
        return {"id": question_id, "question": question, "gold_chunks": gold, "type": question_type, "split": split}


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows terminals default to an encoding that crashes on symbols like "≤"
    existing = load_questions()
    already_labelled = {q["question"] for q in existing}
    if len(sys.argv) > 1:
        questions = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    else:
        questions = iter(lambda: input("\nQuestion (blank to quit): ").strip(), "")

    for question in questions:
        question = question.strip()
        if not question or question in already_labelled:
            continue
        print("\n" + "=" * 70 + "\nQ:", question)
        item = label(question, existing)
        if item:
            with open(QUESTIONS_FILE, "a", encoding="utf-8") as file:
                file.write(json.dumps(item) + "\n")
            existing.append(item)
            print(f"  saved {item['id']} ({item['type']}, {item['split']})")

    counts = {}
    for q in existing:
        counts[q["type"]] = counts.get(q["type"], 0) + 1
    print("\nTotal labelled:", len(existing), counts)


if __name__ == "__main__":
    main()
