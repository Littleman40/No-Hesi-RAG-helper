# No Hesi RAG Helper

A support assistant that answers player questions using the official No Hesi FAQ. It finds the right FAQ section, writes a short answer with a source link, says "I don't know" when the FAQ doesn't cover the question, and hands refunds and account changes to a human.

Everything runs locally and costs nothing per question.

## Why I built it

To learn how retrieval-augmented generation (RAG) works, and how to measure it properly. Being the manager of the No Hesi FAQ Gitbook, made it super simple to scrape the data, to setup everything.

The bots goals:
- answers only from the FAQ, never from the model's own guesses
- every answer links to the FAQ section it came from
- refuses instead of guessing when the FAQ doesn't cover it, because a confident wrong is far worse than "I don't know"
- never tries to handle sensitive topics such as refunds or account changes

## How it works

```
ingest.py: copy gitbook public pages, skip internal ones
chunk.py: split by headings into 297 chunks (max 300 words)
index.py: BM25 keyword index + MiniLM embeddings (cached on disk)
retrieve.py: hybrid search: both rankings fused with reciprocal rank fusion
answer.py: 
- human-only topic? -> hand over to staff
- nothing relevant? -> refuse
- otherwise -> Qwen3 8B writes an answer from the top 5 chunks
```

- **BM25** is keyword search: it rewards chunks that share the question's rare words, like `d3dx11_43.dll`.
- **Vector search** turns text into embeddings, lists of numbers where similar meanings sit close together, so "colour" finds "color".
- **Reciprocal rank fusion (RRF)** combines the two rankings by position, so a chunk both methods like rises to the top.

## Tech stack

* Python
* rank-bm25
* NumPy
* Qwen3 8B - served by Ollama through my Mac mini
* pytest

## How I measured it

Before tuning anything I built an evaluation set: 53 real questions from the help channels, anonymised and labelled with the chunk that answers each one, including typos, slang and pasted error messages.

- 47 answerable, 3 the FAQ doesn't cover, 3 that must go to a human
- split into a **dev half** for tuning and a **held-out half** run only at the end, so the final number isn't flattered by tuning

Metrics:

- **Recall@k:** is the right chunk in the top k results?
- **MRR:** how high the right chunk ranks on average
- **Refusal accuracy:** share of uncovered questions the bot declines
- **Faithfulness:** manual 0–2 score on whether every claim comes from the retrieved chunks
- **Latency**

## Results

All experiments were run on the dev half, changing one thing at a time.

| Experiment | Recall@1 | Recall@3 | Recall@5 | MRR | Retrieval latency |
|---|---|---|---|---|---|
| BM25 only (baseline) | 33% | 58% | 62% | 0.47 | <1 ms |
| Vector only | 21% | 58% | 75% | 0.42 | 6 ms |
| **Hybrid (RRF)** | **50%** | **79%** | **88%** | **0.64** | **6 ms** |
| + query rewriting (fused) | 50% | 83% | 88% | 0.66 | 605 ms |
| + cross-encoder reranker | 33% | 58% | 67% | 0.48 | 729 ms |

**Held-out result:** 65% Recall@5 and 0.61 MRR, against 88% on dev. With about 24 questions per half, one question moves recall by 4 points, and three held-out misses were the same error message. This gap is exactly why the held-out half exists.

**After error analysis:** I fixed the two cheapest failure cases (below). Held-out Recall@5 rose to **83%** and MRR to **0.73**. These fixes were chosen by reading held-out failures, so 65% stays the honest held-out number.

**Answers** (dev baseline):
- faithfulness 1.70 / 2, with 17 of 20 answers fully supported and 3 that added something unsupported
- both uncovered questions refused
- median 23 s per answer on the Mac mini, $0 cost

## What didn't work, and why

Most of my "improvements" made things worse:

- **Bigger embedding models** (bge-small, bge-base, mpnet) all scored lower than the small MiniLM model, even with BGE's recommended query instruction. My guess is that short, slangy gaming questions are not what larger models were trained to match.
- **Cross-encoder reranker:** Recall@5 fell from 88% to 67% and it added 0.7 s per question. It was trained on web search queries and doesn't know the game's jargon.
- **LLM query rewriting:** the first prompt dropped Recall@5 to 58%. It added "in Assetto Corsa / No Hesi" to every query (words found in nearly every chunk), and once answered in Chinese. A stricter prompt fixed that. Fusing the original and rewritten queries gained one question for an extra 0.6 s and an extra model call, so I didn't keep it.
- **Fixed-size chunks** (plain word windows) lost to heading-aware chunks at every length. 300 words was the best limit.
- **The score floor** couldn't catch the uncovered questions: they are about servers, so they score like real questions. The model's `NO_ANSWER` does the real refusing, and the floor only catches true nonsense ("what is the capital of France"). I lowered it after it wrongly refused a real question.

## Fixes from error analysis

1. **"Car doesn't match the server" errors.** Players paste the in-game error, but the FAQ page is titled "Mismatch: COLLIDER / DATA" and never contains that text. I added `search_aliases.py`, extra search text per FAQ page that the model never sees. I also made the tokenizer treat "doesn't", "doesnt" and "does'nt" as the same word.
2. **Escalation keywords were too literal.** "unlink **a** steam account" didn't match "unlink steam". Phrases now match when each word appears, in any order. All 3 human-only questions are now escalated, and none of the other 50 are.

## Known limitations

- **Broad questions get specific answers:** "What do I need to download to play?" was answered from a Steam page instead of the install-guide overview.
- **The eval set is small:** 53 questions, with only 3 uncovered ones, so refusal numbers rest on very little data. The held-out half has been used once for fixes, so it is no longer a clean test.
- **Answers are slow**: about 23 s on an 8B model running on a Mac mini.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen3:8b

python src/ingest.py
python src/chunk.py
python src/index.py

python src/answer.py "I get checksum failed when joining a server"
python eval/run_eval.py
python -m pytest
```
