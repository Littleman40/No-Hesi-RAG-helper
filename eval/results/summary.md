# Results summary

The 53 eval questions are split in two. **Practice questions** (24 answerable, 2 uncovered) were used to test every change. **Unseen questions** (23 answerable, 1 uncovered, 1 human-only) were kept aside and checked once at the end, to get an honest score that wasn't tuned to.

Every experiment below was run on the practice questions, changing one thing at a time. Recall and MRR are measured without the score floor; latency is retrieval time per question.

**Final setup:** hybrid search (BM25 + MiniLM, fused with RRF), heading-aware chunks up to 300 words, no query rewriting, no reranker, score floor BM25 8.0 / cosine 0.30, and Qwen3 8B on Ollama for answers.

## Experiments

| Experiment | Recall@1 | Recall@3 | Recall@5 | MRR | Latency |
|---|---|---|---|---|---|
| BM25 only (baseline) | 33% | 58% | 62% | 0.47 | <1 ms |
| Vector only | 21% | 58% | 75% | 0.42 | 6 ms |
| **Hybrid (RRF)** | **50%** | **79%** | **88%** | **0.64** | **6 ms** |
| + query rewriting (fused) | 50% | 83% | 88% | 0.66 | 605 ms |
| + reranker | 33% | 58% | 67% | 0.48 | 729 ms |

### Chunking

| Chunking | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|
| Headings, max 150 words | 331 | 38% | 75% | 83% | 0.57 |
| **Headings, max 300 words** | **297** | **50%** | **79%** | **88%** | **0.64** |
| Headings, max 500 words | 293 | 46% | 79% | 79% | 0.63 |
| Fixed-size, 150 words | 245 | 25% | 58% | 67% | 0.45 |
| Fixed-size, 300 words | 191 | 25% | 71% | 79% | 0.50 |

Splitting by headings beat fixed-size windows at every length. Different chunkings give different chunk ids, so these were scored by page and word overlap instead.

### Embedding models (hybrid search)

| Model | Recall@1 | Recall@3 | Recall@5 | MRR | Latency |
|---|---|---|---|---|---|
| **all-MiniLM-L6-v2** (smallest) | **50%** | **79%** | **88%** | **0.64** | **7 ms** |
| bge-small-en-v1.5 | 46% | 62% | 75% | 0.56 | 11 ms |
| bge-small-en-v1.5 + query instruction | 46% | 62% | 71% | 0.57 | 11 ms |
| bge-base-en-v1.5 | 46% | 75% | 79% | 0.61 | 23 ms |
| bge-base-en-v1.5 + query instruction | 50% | 75% | 79% | 0.62 | 25 ms |
| all-mpnet-base-v2 (5x larger) | 46% | 71% | 79% | 0.61 | 23 ms |

Every larger model did worse. My guess is that short, slangy gaming questions aren't what they were trained to match.

### Query rewriting

| Variant | Recall@1 | Recall@3 | Recall@5 | MRR | Latency |
|---|---|---|---|---|---|
| **Raw question** | **50%** | **79%** | **88%** | **0.64** | **6 ms** |
| Rule-based cleaning | 50% | 79% | 88% | 0.64 | 6 ms |
| LLM rewrite, first prompt | 17% | 42% | 58% | 0.32 | 981 ms |
| LLM rewrite, stricter prompt | 38% | 75% | 83% | 0.57 | 629 ms |
| Raw + rewrite, fused | 50% | 83% | 88% | 0.66 | 605 ms |

- **Rule-based cleaning** (removing timestamps, file paths and memory addresses) changed nothing, because no practice question contained a pasted log.
- **The first rewrite prompt** added "in Assetto Corsa / No Hesi" to every query, words found in nearly every chunk, and once answered in Chinese.
- **Fusing** gained one question for an extra 0.6 s and a model call per question. That's within noise, so it wasn't kept.

### Reranker

A cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scoring the top 20 dropped Recall@5 from 88% to 67% and added 0.7 s. It was trained on web searches and doesn't know the game's jargon.

### Score floor

The two uncovered practice questions are about servers ("filter by specific servers", "admin password for my server"), so they score as high as real questions, and no floor could refuse them without also refusing real ones. The model's `NO_ANSWER` refused both. The old floor refused neither and wrongly refused one real question, so I lowered the cosine floor from 0.35 to 0.30. True nonsense like "what is the capital of France" is still refused. With only 2 uncovered questions, this is weak evidence.

## Result on the unseen questions

Run once with the final setup.

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| **Hybrid** | **57%** | **65%** | **65%** | **0.61** |
| BM25 only | 39% | 57% | 61% | 0.48 |
| Vector only | 35% | 65% | 70% | 0.49 |

**65% Recall@5 on unseen questions, against 88% on practice questions.** Each set has only about 24 questions, so one question moves recall by 4 points, and three of the misses were the same error message, which barely appeared in the practice questions.

## Failure cases

1. **"Car doesn't match the server" errors** *(fixed)*. Players paste the in-game error, but the FAQ page is titled "Mismatch: COLLIDER / DATA". Typos like "doesn'st" split into junk words, and vector search confused it with the "Car Locked" page.
2. **Escalation keywords too literal** *(fixed)*. "unlink **a** steam account" didn't contain the phrase "unlink steam", so it went to search instead of a human.
3. **Right chunk found, but the model refused.** The fix for "rpm maxed without my foot on the pedal" is titled "CSP SpeedBug fix", a name players don't know, and the model didn't connect the two.
4. **Broad questions get specific answers.** "What do I need to download to play?" was answered from a Steam page instead of the install-guide overview.

Also seen: the citation check once removed an invented link but left its Markdown brackets behind.

## After the fixes

- **Fix 1:** `src/search_aliases.py` adds the in-game error wording to the mismatch page, for search only; the model never sees it. The tokenizer now treats "doesn't", "doesnt" and "does'nt" as the same word.
- **Fix 2:** escalation phrases match when each word appears in the question, in any order. All 3 human-only questions are now escalated, and none of the other 50 are.

I chose these fixes by looking at the unseen questions' mistakes, so they're no longer truly unseen. **65% stays the honest number.**

| Hybrid search, with floor | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| Practice, before | 46% | 75% | 83% | 0.60 |
| Practice, after | 46% | 83% | 88% | 0.63 |
| Unseen, before | 57% | 65% | 65% | 0.61 |
| Unseen, after | 65% | 78% | 83% | 0.73 |

| Full pipeline, unseen questions | Before | After |
|---|---|---|
| Uncovered questions refused | 1 of 1 | 0 of 1 |
| Answerable questions wrongly refused | 4 of 23 | 3 of 23 |
| Human-only questions escalated | 0 of 1 | 1 of 1 |
| Median seconds per answer | 22.3 | 23.9 |

All three "doesn't match" questions now find the right chunk; two get correct answers, and the third hits failure case 3.

The one uncovered question, "are there any new cars?", is now answered from the Known Issues page, which does mention new cars. So the FAQ partly covers it, and the label is debatable rather than this being a real regression.
