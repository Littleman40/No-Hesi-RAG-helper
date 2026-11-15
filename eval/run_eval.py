# score bot against eval

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import answer
import retrieve

QUESTIONS_FILE = PROJECT_ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
MODES = ["bm25", "vector", "hybrid"]
DEPTH = 10

def load_questions(path, split):
    with open(path, encoding="utf-8") as file:
        questions = [json.loads(line) for line in file if line.strip()]
    if split != "all":
        questions = [q for q in questions if q.get("split") == split]
    return questions


def first_gold_rank(result_ids, gold_chunks):
    for rank, chunk_id in enumerate(result_ids, start=1):
        if chunk_id in gold_chunks:
            return rank
    return None

def recall_at(ranks, k):
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)

def mrr(ranks):
    return sum(1 / rank for rank in ranks if rank is not None) / len(ranks)

def percent(value):
    return f"{value * 100:.0f}%"

def score_mode(questions, mode):
    ranks, misses, times = [], [], []
    out_refused, in_refused = [], []
    for q in questions:
        if q["type"] == "escalate":
            continue
        start = time.perf_counter()
        result_ids = [chunk_id for chunk_id, _ in retrieve.search(q["question"], mode, DEPTH)]
        times.append((time.perf_counter() - start) * 1000)
        if q["type"] == "out_of_scope":
            out_refused.append(not result_ids)
        else:
            rank = first_gold_rank(result_ids, q["gold_chunks"])
            ranks.append(rank)
            in_refused.append(not result_ids)
            if rank is None or rank > 5:
                misses.append((q, result_ids[:1]))
    return {"ranks": ranks, "misses": misses, "times": times, "out_refused": out_refused, "in_refused": in_refused}


def retrieval_report(questions):
    in_scope = sum(1 for q in questions if q["type"] == "in_scope")
    lines = [
        "## Retrieval",
        "",
        f"{in_scope} in-scope questions. Refusal only applies to hybrid (the only mode with a score floor).",
        "",
        "| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Refusal accuracy | False refusals | Median ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    hybrid = None
    for mode in MODES:
        s = score_mode(questions, mode)
        if mode == "hybrid":
            hybrid = s
        refusal = percent(statistics.mean(s["out_refused"])) if mode == "hybrid" and s["out_refused"] else "n/a"
        false_refusal = percent(statistics.mean(s["in_refused"])) if mode == "hybrid" else "n/a"
        lines.append(
            f"| {mode} | {percent(recall_at(s['ranks'], 1))} | {percent(recall_at(s['ranks'], 3))} "
            f"| {percent(recall_at(s['ranks'], 5))} | {mrr(s['ranks']):.2f} | {refusal} | {false_refusal} "
            f"| {statistics.median(s['times']):.0f} |"
        )

    lines += ["", "### Hybrid misses (correct chunk not in the top 5)", ""]
    for q, top in hybrid["misses"]:
        got = retrieve.get_chunk(top[0])["heading_path"] if top else "(nothing, refused)"
        lines.append(f"- {q['id']} \"{q['question']}\" -> got: {got}")
    if not hybrid["misses"]:
        lines.append("- none")
    return lines


def escalation_report(questions):
    should = [q for q in questions if q["type"] == "escalate"]
    should_not = [q for q in questions if q["type"] != "escalate"]
    caught = sum(1 for q in should if answer.escalate_topic(q["question"]))
    wrongly = [q for q in should_not if answer.escalate_topic(q["question"])]
    lines = ["## Escalation", "", f"- Caught: {caught} of {len(should)} human-only questions"]
    lines.append(f"- Wrongly escalated: {len(wrongly)} of {len(should_not)} other questions")
    for q in wrongly:
        lines.append(f"  - {q['id']} \"{q['question']}\"")
    return lines


def generation_report(questions, answers_file):
    answer.log = lambda result: None
    rows = []
    for q in questions:
        result = answer.answer(q["question"])
        rows.append({
            "id": q["id"], "question": q["question"], "type": q["type"], "gold_chunks": q["gold_chunks"],
            "status": result["status"], "chunk_ids": result["chunk_ids"], "reply": result["reply"],
            "latency_ms": result["latency_ms"], "faithfulness": None,
        })
        print(f"  {q['id']} {result['status']} ({result['latency_ms']} ms)", file=sys.stderr)

    answers_file.parent.mkdir(parents=True, exist_ok=True)
    with open(answers_file, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")

    def share(question_type, status):
        group = [r for r in rows if r["type"] == question_type]
        return f"{sum(1 for r in group if r['status'] == status)} of {len(group)}"

    answered = [r["latency_ms"] for r in rows if r["status"] == "answered"]
    return [
        "## Full answers",
        "",
        f"- Model: {answer.MODEL}",
        f"- Out-of-scope correctly refused: {share('out_of_scope', 'refused')}",
        f"- In-scope wrongly refused: {share('in_scope', 'refused')}",
        f"- Human-only correctly escalated: {share('escalate', 'escalated')}",
        f"- Median seconds per answer: {statistics.median(answered) / 1000:.1f}" if answered else "- No answers generated",
        "- Cost per 100 questions: $0 if the model runs locally",
        f"- Answers saved to {answers_file.relative_to(PROJECT_ROOT)} for faithfulness scoring",
    ]


def faithfulness_report(path):
    with open(path, encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    scores = [row["faithfulness"] for row in rows if row["status"] == "answered"]
    scores = [s for s in scores if s is not None]
    if not scores:
        return ["No faithfulness scores filled in yet."]
    return [
        "## Faithfulness",
        "",
        f"- Scored answers: {len(scores)}",
        f"- Average: {statistics.mean(scores):.2f} / 2",
        f"- Fully faithful (2): {scores.count(2)}, partly (1): {scores.count(1)}, made things up (0): {scores.count(0)}",
    ]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--questions", default=QUESTIONS_FILE, type=Path)
    parser.add_argument("--generate", action="store_true", help="also run the AI model on every question")
    parser.add_argument("--save", metavar="NAME", help="save the report to eval/results/NAME.md")
    parser.add_argument("--faithfulness", metavar="FILE", type=Path, help="average the scores in an answers file")
    args = parser.parse_args()

    if args.faithfulness:
        print("\n".join(faithfulness_report(args.faithfulness)))
        return

    questions = load_questions(args.questions, args.split)
    if not any(q["type"] == "in_scope" for q in questions):
        sys.exit(f"No in-scope questions in the {args.split} split of {args.questions}. Label some with eval/label.py first.")

    retrieve.load()
    lines = [f"# Eval report: {args.split} split, {len(questions)} questions", "", f"Run {time.strftime('%Y-%m-%d %H:%M')}", ""]
    lines += retrieval_report(questions) + [""] + escalation_report(questions)
    if args.generate:
        answers_file = RESULTS_DIR / "answers" / f"{args.save or 'latest'}.jsonl"
        lines += [""] + generation_report(questions, answers_file)

    report = "\n".join(lines)
    print(report)
    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / f"{args.save}.md").write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved to eval/results/{args.save}.md")


if __name__ == "__main__":
    main()
