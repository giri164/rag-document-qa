"""
evaluate.py
-----------
Lightweight retrieval-quality check: for a small set of (question, expected
keyword) pairs, verifies that the top-k retrieved chunks contain the expected
keyword. This is a cheap sanity check you can run in CI without needing a
GPU or downloading the generation LLM.

Usage:
    python src/evaluate.py --index_dir vector_store
"""

import argparse

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

TEST_CASES = [
    {"question": "How many days can I work remotely?", "expected_keyword": "3 days"},
    {"question": "How much paid leave do employees get?", "expected_keyword": "18 days"},
    {"question": "When must expenses be submitted?", "expected_keyword": "30 days"},
]


def evaluate(index_dir: str, embedding_model: str, top_k: int = 4):
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = FAISS.load_local(
        index_dir, embeddings, allow_dangerous_deserialization=True
    )

    passed = 0
    for case in TEST_CASES:
        docs = vector_store.similarity_search(case["question"], k=top_k)
        combined = " ".join(d.page_content for d in docs)
        hit = case["expected_keyword"].lower() in combined.lower()
        passed += hit
        status = "PASS" if hit else "FAIL"
        print(f"[{status}] '{case['question']}' -> expected '{case['expected_keyword']}'")

    print(f"\n{passed}/{len(TEST_CASES)} retrieval checks passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", default="vector_store")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=4)
    args = parser.parse_args()
    evaluate(args.index_dir, args.embedding_model, args.top_k)


if __name__ == "__main__":
    main()
