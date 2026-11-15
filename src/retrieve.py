# search function for the faq
import numpy as np

import index

loaded = {}

def load():
    if not loaded:
        chunks = index.load_chunks()
        model = index.load_model()
        loaded["chunks"] = chunks
        loaded["bm25"] = index.build_bm25(chunks)
        loaded["model"] = model
        loaded["vectors"] = index.load_or_build_embeddings(chunks, model)
    return loaded

def top_k(scores, k):
    chunks = load()["chunks"]
    best_positions = np.argsort(scores)[::-1][:k]
    results = []
    for position in best_positions:
        results.append((chunks[position]["id"], float(scores[position])))
    return results

def bm25_search(query, k=5):
    bm25 = load()["bm25"]
    scores = bm25.get_scores(index.tokenize(query))
    return top_k(scores, k)


def vector_search(query, k=5):
    data = load()
    query_vector = index.normalize(data["model"].encode(query))
    scores = np.dot(data["vectors"], query_vector)
    return top_k(scores, k)


def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)

BM25_FLOOR = 8.0
VECTOR_FLOOR = 0.35

def hybrid_search(query, k=5, pool=20):
    bm25_results = bm25_search(query, pool)
    vector_results = vector_search(query, pool)
    if bm25_results[0][1] < BM25_FLOOR and vector_results[0][1] < VECTOR_FLOOR:
        return []
    bm25_ids = [chunk_id for chunk_id, _ in bm25_results]
    vector_ids = [chunk_id for chunk_id, _ in vector_results]
    return rrf([bm25_ids, vector_ids])[:k]


SEARCH_MODES = {"bm25": bm25_search, "vector": vector_search, "hybrid": hybrid_search}

def search(query, mode="hybrid", k=5):
    return SEARCH_MODES[mode](query, k)

def get_chunk(chunk_id):
    for chunk in load()["chunks"]:
        if chunk["id"] == chunk_id:
            return chunk
    return None

def main():
    questions = [
        "I get checksum failed when joining a server",
        "game says d3dx11_43.dll is missing",
        "how do I change the colour of my car",
        "what is the capital of France",  # off-topic: hybrid should return nothing
    ]
    for question in questions:
        print("\n" + "=" * 70)
        print("Q:", question)
        for mode in SEARCH_MODES:
            print("-", mode)
            results = search(question, mode, k=3)
            if not results:
                print("    (nothing above the score floor)")
            for chunk_id, score in results:
                print("   ", round(score, 3), "|", get_chunk(chunk_id)["heading_path"])


if __name__ == "__main__":
    main()
