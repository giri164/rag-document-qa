# RAG Document Q&A System

A Retrieval-Augmented Generation pipeline that answers questions grounded in
your own documents (PDF / TXT / Markdown), using open-source models end to
end — no paid API keys required.

## Architecture

```
Documents (PDF/TXT/MD)
        |
   chunking (LangChain RecursiveCharacterTextSplitter)
        |
   embeddings (sentence-transformers/all-MiniLM-L6-v2)
        |
   FAISS vector store (persisted locally)
        |
   top-k retrieval  --->  Qwen2.5-1.5B-Instruct  --->  grounded answer
```

## Features

- Ingests PDF, TXT, and Markdown files from a folder
- Chunking with configurable size/overlap
- Local, free embeddings via `sentence-transformers`
- FAISS vector store persisted to disk (no external vector DB needed)
- Generation with an open-source instruction-tuned LLM (Qwen2.5-1.5B-Instruct
  by default — swappable for any HF causal LM)
- CLI for both single-question and interactive chat mode
- Lightweight retrieval evaluation script

## Project structure

```
rag-document-qa/
├── data/sample_docs/       # sample source documents
├── src/
│   ├── ingest.py           # build the FAISS index from documents
│   ├── rag_pipeline.py     # retrieval + generation
│   └── evaluate.py         # retrieval sanity-check tests
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Add your documents to `data/sample_docs/` (a sample HR-policy doc is
   included so you can try it immediately).

2. Build the vector index:
   ```bash
   python src/ingest.py --data_dir data/sample_docs --index_dir vector_store
   ```

3. Ask questions:
   ```bash
   # single question
   python src/rag_pipeline.py --index_dir vector_store --question "How many days can I work remotely?"

   # interactive chat
   python src/rag_pipeline.py --index_dir vector_store
   ```

4. (Optional) Run the retrieval sanity check:
   ```bash
   python src/evaluate.py --index_dir vector_store
   ```

## Configuration

All key parameters are CLI flags on `ingest.py` and `rag_pipeline.py`:
`--embedding_model`, `--llm_model`, `--chunk_size`, `--chunk_overlap`,
`--top_k`. Swap in any HuggingFace embedding or causal-LM checkpoint without
touching the code.

## Notes

- Runs fully on CPU (slower generation) or GPU if available (`torch` will
  use CUDA automatically).
- `Qwen/Qwen2.5-1.5B-Instruct` is downloaded from HuggingFace on first run
  (~3GB). Swap in a smaller model (e.g. `Qwen/Qwen2.5-0.5B-Instruct`) for
  faster iteration on limited hardware.

## Possible extensions

- Add re-ranking (e.g. a cross-encoder) after initial FAISS retrieval
- Swap FAISS for a hosted vector DB (Pinecone, Weaviate, Qdrant) for scale
- Add streaming token generation for a chat UI
- Add citation highlighting back to source PDF pages
