# turn question into an answer
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

import retrieve
from escalate_topics import ESCALATE_TOPICS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / "answers.jsonl"
load_dotenv(PROJECT_ROOT / ".env")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") # hosted on my mac mini
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

HANDOVER_REPLY = "That one needs a human, so I've passed it to the team. Someone will get back to you."
REFUSAL_REPLY = "I couldn't find that in the FAQ. I've passed it to the team."

SYSTEM_PROMPT = """You answer questions from No Hesi players using ONLY the numbered FAQ blocks you are given.

Rules:
- Use only facts from the blocks. Never add steps, fixes, links or advice from your own knowledge.
- Keep the answer under about 120 words. Plain text; short numbered steps are fine.
- End with one line "Source: <url>", copying the url of the block you used exactly.
- If the blocks don't answer the question, output exactly NO_ANSWER and nothing else."""

URL_PATTERN = re.compile(r"https?://[^\s<>()\"']+")

def escalate_topic(question):
    question = question.lower()
    for topic, phrases in ESCALATE_TOPICS.items():
        for phrase in phrases:
            if phrase in question:
                return topic
    return None


def build_prompt(question, chunks):
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{number}] {chunk['url']}\n{chunk['text']}")
    return "FAQ blocks:\n\n" + "\n\n".join(blocks) + "\n\nQuestion: " + question


def check_citations(text, allowed_urls):
    def keep_or_drop(match):
        url = match.group(0).rstrip(".,;:!?")
        return match.group(0) if url in allowed_urls else ""

    text = URL_PATTERN.sub(keep_or_drop, text)
    return re.sub(r"(?m)^Source:\s*$\n?", "", text).strip()


def generate(question, chunks):
    request = urllib.request.Request(
        OLLAMA_URL.rstrip("/") + "/api/chat",
        data=json.dumps({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, chunks)},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        text = json.loads(response.read())["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def log(result):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(result) + "\n")


def answer(question):
    start = time.perf_counter()
    topic = escalate_topic(question)
    chunk_ids = []

    if topic:
        status, reply = "escalated", HANDOVER_REPLY
    else:
        results = retrieve.search(question, "hybrid", k=5)
        chunk_ids = [chunk_id for chunk_id, _ in results]
        chunks = [retrieve.get_chunk(chunk_id) for chunk_id in chunk_ids]
        raw = generate(question, chunks) if chunks else "NO_ANSWER"
        if "NO_ANSWER" in raw:
            status, reply = "refused", REFUSAL_REPLY
        else:
            urls = {chunk["url"] for chunk in chunks}
            status, reply = "answered", check_citations(raw, urls)

    result = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "status": status,
        "escalate": status != "answered",
        "topic": topic,
        "chunk_ids": chunk_ids,
        "reply": reply,
        "latency_ms": round((time.perf_counter() - start) * 1000),
    }
    log(result)
    return result


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows terminals default to an encoding that crashes on symbols like "≤"
    questions = sys.argv[1:] or iter(lambda: input("\nQuestion (blank to quit): ").strip(), "")
    for question in questions:
        result = answer(question)
        print(f"\n[{result['status']}, {result['latency_ms']} ms]\n{result['reply']}")


if __name__ == "__main__":
    main()
