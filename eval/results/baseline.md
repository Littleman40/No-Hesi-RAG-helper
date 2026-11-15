# Eval report: dev split, 28 questions

Run 2026-11-15 14:41

## Retrieval

24 in-scope questions. Refusal only applies to hybrid (the only mode with a score floor).

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Refusal accuracy | False refusals | Median ms |
|---|---|---|---|---|---|---|---|
| bm25 | 33% | 58% | 62% | 0.47 | n/a | n/a | 0 |
| vector | 21% | 58% | 75% | 0.42 | n/a | n/a | 5 |
| hybrid | 46% | 75% | 83% | 0.60 | 0% | 4% | 5 |

### Hybrid misses (correct chunk not in the top 5)

- q014 "where do i even find srp to download it" -> got: How to get better performance in SRP > Step 1
- q016 "i cant load into a srp map" -> got: How to get better performance in SRP > Step 1
- q041 "my gas pedal is on full throttle" -> got: (nothing, refused)
- q048 "Why can’t I join any servers it’s saying race canceled" -> got: Custom Servers - Custom Tracks - Custom Cars - The Full Guide > Custom Server FAQ

## Escalation

- Caught: 1 of 2 human-only questions
- Wrongly escalated: 0 of 26 other questions
