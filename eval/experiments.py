# tests different search strats to find the best one

import argparse
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

import answer
import chunk as chunker
import index
import retrieve
from run_eval import QUESTIONS_FILE, RESULTS_DIR, load_questions, mrr, percent, recall_at

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OTHER_MODELS = ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "sentence-transformers/all-mpnet-base-v2"]
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
POOL = 20
DEPTH = 10

models = {}


def get_model(name):
    if name not in models:
        models[name] = SentenceTransformer(name)
    return models[name]


def use_index(chunks, model_name):
    model = get_model(model_name)
    vectors = index.normalize(model.encode([index.search_text(c) for c in chunks], batch_size=32))
    retrieve.loaded.clear()
    retrieve.loaded.update({"chunks": chunks, "bm25": index.build_bm25(chunks), "model": model, "vectors": vectors})

def heading_chunks(max_words):
    chunker.MAX_WORDS = max_words
    chunks = []
    for path in sorted(chunker.RAW_DIR.glob("*.md")):
        chunks.extend(chunker.chunk_page(path))
    chunker.MAX_WORDS = 300
    return chunks


def fixed_chunks(size):
    chunks = []
    for path in sorted(chunker.RAW_DIR.glob("*.md")):
        header, body = chunker.read_raw_page(path)
        title = header.get("title", path.stem)
        lines = [line.lstrip("#").strip() for line in chunker.clean_lines(body) if not line.startswith("# ")]
        words = " ".join(lines).split()
        for number, start in enumerate(range(0, len(words), size), start=1):
            chunks.append({
                "id": f"{path.stem}-fixed-{number}",
                "page_title": title,
                "heading_path": f"{title} (words {start}-{start + size})",
                "url": header.get("url", ""),
                "text": title + "\n\n" + " ".join(words[start:start + size]),
            })
    return chunks



def body_words(chunk):
    return set(index.tokenize(chunk["text"].split("\n\n", 1)[-1]))

def make_is_hit(match, gold_by_id):
    def is_hit(candidate, gold_ids):
        if match == "id":
            return candidate["id"] in gold_ids
        for gold_id in gold_ids:
            gold = gold_by_id[gold_id]
            if candidate["page_title"] == gold["page_title"]:
                gold_words = body_words(gold)
                if gold_words and len(gold_words & body_words(candidate)) / len(gold_words) >= 0.5:
                    return True
        return False
    return is_hit

def bm25_ids(query):
    return [cid for cid, _ in retrieve.bm25_search(query, POOL)]


def vector_ids(query):
    return [cid for cid, _ in retrieve.vector_search(query, POOL)]


def hybrid_ids(query):
    return [cid for cid, _ in retrieve.rrf([bm25_ids(query), vector_ids(query)])]


reranker = None


def reranked_ids(query):
    global reranker
    if reranker is None:
        reranker = CrossEncoder(RERANKER)
    ids = hybrid_ids(query)[:POOL]
    scores = reranker.predict([(query, retrieve.get_chunk(cid)["text"]) for cid in ids])
    return [cid for _, cid in sorted(zip(scores, ids), key=lambda pair: pair[0], reverse=True)]



def floor_refuses(query, bm25_floor, vector_floor):
    best_bm25 = retrieve.bm25_search(query, 1)[0][1]
    best_vector = retrieve.vector_search(query, 1)[0][1]
    return best_bm25 < bm25_floor and best_vector < vector_floor

def run(name, questions, search, is_hit, prepare=None, floor=None):
    ranks, times = [], []
    out_refused, in_refused = [], []
    for q in questions:
        if q["type"] == "escalate":
            continue
        start = time.perf_counter()
        query = prepare(q["question"]) if prepare else q["question"]
        ids = search(query)
        times.append(time.perf_counter() - start)
        if floor:
            refused = floor_refuses(query, *floor)
            (out_refused if q["type"] == "out_of_scope" else in_refused).append(refused)
        if q["type"] == "in_scope":
            rank = None
            for position, cid in enumerate(ids[:DEPTH], start=1):
                if is_hit(retrieve.get_chunk(cid), q["gold_chunks"]):
                    rank = position
                    break
            ranks.append(rank)
    return {
        "name": name,
        "r1": percent(recall_at(ranks, 1)), "r3": percent(recall_at(ranks, 3)), "r5": percent(recall_at(ranks, 5)),
        "mrr": f"{mrr(ranks):.2f}",
        "refusal": f"{sum(out_refused)} of {len(out_refused)}" if floor else "",
        "false_refusal": f"{sum(in_refused)} of {len(in_refused)}" if floor else "",
        "latency": f"{statistics.median(times):.3f}",
        "ranks": ranks,
    }


def table(rows, title):
    lines = [f"### {title}", "", "| Experiment | Recall@1 | Recall@3 | Recall@5 | MRR | Refusal acc. | False refusals | Median latency (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['r1']} | {r['r3']} | {r['r5']} | {r['mrr']} | {r['refusal']} | {r['false_refusal']} | {r['latency']} |")
    return lines + [""]

def clean_query(text):
    import re
    text = re.sub(r"\[?\d{1,2}:\d{2}(:\d{2})?(\.\d+)?\]?", " ", text)          # 12:34:56
    text = re.sub(r"\d{4}-\d{2}-\d{2}", " ", text)                              # 2026-11-15
    text = re.sub(r"(?:[A-Za-z]:)?(?:[\\/][\w .-]+?)+[\\/]([\w.-]+)", r"\1", text)  # C:\a\b\x.dll -> x.dll
    text = re.sub(r"0x[0-9a-fA-F]+", " ", text)                                 # 0x7ff6...
    return " ".join(text.split())


REWRITE_PROMPT = """Rewrite the player's message into a short, clean search query for an Assetto Corsa / No Hesi FAQ.
Fix spelling and slang. Keep error messages, file names, car and track names exactly as written.
Output only the query, nothing else."""

rewrites = {}


def llm_rewrite(text):
    if text not in rewrites:
        rewrites[text] = answer.ollama_chat(REWRITE_PROMPT, text, think=False)
    return rewrites[text]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    questions = load_questions(QUESTIONS_FILE, "dev")
    original = index.load_chunks()
    gold_by_id = {c["id"]: c for c in original}
    by_id = make_is_hit("id", gold_by_id)
    by_overlap = make_is_hit("overlap", gold_by_id)
    default_floor = (retrieve.BM25_FLOOR, retrieve.VECTOR_FLOOR)
    out = [f"# Experiments (dev half, {sum(q['type'] == 'in_scope' for q in questions)} in-scope, "
           f"{sum(q['type'] == 'out_of_scope' for q in questions)} out-of-scope)", "",
           f"Run {time.strftime('%Y-%m-%d %H:%M')}. Recall/MRR are measured without the score floor.", ""]

    use_index(original, DEFAULT_MODEL)
    modes = [run("BM25 only (baseline)", questions, bm25_ids, by_id),
             run("Vector only", questions, vector_ids, by_id),
             run("Hybrid (RRF)", questions, hybrid_ids, by_id, floor=default_floor)]
    out += table(modes, "1-3. Search modes (current chunks, MiniLM)")

    rows = []
    for name, chunks in [("Heading-aware, max 150 words", heading_chunks(150)),
                         ("Heading-aware, max 300 words (current)", heading_chunks(300)),
                         ("Heading-aware, max 500 words", heading_chunks(500)),
                         ("Fixed-size, 150 words", fixed_chunks(150)),
                         ("Fixed-size, 300 words", fixed_chunks(300))]:
        use_index(chunks, DEFAULT_MODEL)
        row = run(f"{name} ({len(chunks)} chunks)", questions, hybrid_ids, by_overlap)
        rows.append(row)
    out += table(rows, "4. Chunking, hybrid search (scored by page + word overlap)")

    rows = []
    for model_name in [DEFAULT_MODEL] + OTHER_MODELS:
        use_index(original, model_name)
        short = model_name.split("/")[-1]
        rows.append(run(f"Vector only, {short}", questions, vector_ids, by_id))
        rows.append(run(f"Hybrid, {short}", questions, hybrid_ids, by_id))
    out += table(rows, "5. Embedding models (current chunks)")

    use_index(original, DEFAULT_MODEL)
    rows = [run("Hybrid, raw question", questions, hybrid_ids, by_id, floor=default_floor),
            run("Hybrid + rule-based cleaning", questions, hybrid_ids, by_id, prepare=clean_query, floor=default_floor)]
    if not args.skip_llm:
        rows.append(run("Hybrid + LLM rewrite", questions, hybrid_ids, by_id, prepare=llm_rewrite, floor=default_floor))
    rows.append(run("Hybrid + reranker", questions, reranked_ids, by_id, floor=default_floor))
    out += table(rows, "6-7. Query rewriting and reranker (current chunks, MiniLM)")
    if rewrites:
        out += ["LLM rewrites:", ""] + [f"- \"{q}\" -> \"{r}\"" for q, r in rewrites.items()] + [""]

    scored = []
    for q in questions:
        if q["type"] != "escalate":
            best_bm25 = retrieve.bm25_search(q["question"], 1)[0][1]
            best_vector = retrieve.vector_search(q["question"], 1)[0][1]
            scored.append((q, best_bm25, best_vector))
    out += ["### 8. Score floor", "", "Best scores per question (out-of-scope first, then the 6 weakest in-scope by cosine):", "",
            "| Question | Type | Best BM25 | Best cosine |", "|---|---|---|---|"]
    oos = [s for s in scored if s[0]["type"] == "out_of_scope"]
    weakest = sorted([s for s in scored if s[0]["type"] == "in_scope"], key=lambda s: s[2])[:6]
    for q, b, v in oos + weakest:
        out.append(f"| {q['question'][:60]} | {q['type']} | {b:.1f} | {v:.2f} |")
    out += ["", "| BM25 floor | Cosine floor | Out-of-scope refused | In-scope wrongly refused |", "|---|---|---|---|"]
    for bm25_floor in [6.0, 8.0, 10.0, 12.0]:
        for vector_floor in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
            oos_refused = sum(1 for _, b, v in oos if b < bm25_floor and v < vector_floor)
            in_refused = sum(1 for q, b, v in scored if q["type"] == "in_scope" and b < bm25_floor and v < vector_floor)
            out.append(f"| {bm25_floor} | {vector_floor} | {oos_refused} of {len(oos)} | {in_refused} of {len(scored) - len(oos)} |")

    report = "\n".join(out)
    print(report)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "experiments.md").write_text(report + "\n", encoding="utf-8")
    print("\nSaved to eval/results/experiments.md")


if __name__ == "__main__":
    main()
