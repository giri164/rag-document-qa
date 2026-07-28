"""
rag_pipeline.py
---------------
Loads a persisted FAISS vector store and answers questions by retrieving the
most relevant chunks and feeding them, along with the question, to an
open-source instruction-tuned LLM (default: Qwen2.5-1.5B-Instruct) via
HuggingFace `transformers`.

Usage:
    python src/rag_pipeline.py --index_dir vector_store --question "What is X?"
    python src/rag_pipeline.py --index_dir vector_store          # interactive mode
"""

import argparse

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using ONLY the
provided context. If the answer is not contained in the context, say
"I don't have enough information in the provided documents to answer that."

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    def __init__(
        self,
        index_dir: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        top_k: int = 4,
    ):
        self.top_k = top_k

        print(f"Loading embedding model '{embedding_model}' ...")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        print(f"Loading FAISS index from '{index_dir}' ...")
        self.vector_store = FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True
        )

        print(f"Loading LLM '{llm_model}' (this can take a while on first run) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            self.model.to("cpu")

    def retrieve(self, question: str):
        return self.vector_store.similarity_search(question, k=self.top_k)

    def generate(self, question: str, context_docs) -> str:
        context = "\n\n".join(d.page_content for d in context_docs)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(
            input_ids,
            max_new_tokens=400,
            temperature=0.3,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = output[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def answer(self, question: str):
        docs = self.retrieve(question)
        response = self.generate(question, docs)
        sources = sorted({d.metadata.get("source", "unknown") for d in docs})
        return response, sources


def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument("--index_dir", default="vector_store")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--question", default=None, help="Ask a single question and exit")
    args = parser.parse_args()

    pipeline = RAGPipeline(
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        top_k=args.top_k,
    )

    if args.question:
        answer, sources = pipeline.answer(args.question)
        print("\nAnswer:\n", answer)
        print("\nSources:", sources)
        return

    print("\nRAG Document Q&A -- type 'exit' to quit\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        answer, sources = pipeline.answer(question)
        print(f"\nAssistant: {answer}")
        print(f"Sources: {sources}\n")


if __name__ == "__main__":
    main()
