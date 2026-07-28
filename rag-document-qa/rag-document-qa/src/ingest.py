"""
ingest.py
---------
Loads documents (PDF / TXT / MD) from a folder, splits them into overlapping
chunks, embeds them with a HuggingFace sentence-transformer model, and
persists a FAISS vector store to disk.

Usage:
    python src/ingest.py --data_dir data/sample_docs --index_dir vector_store
"""

import argparse
import os

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_documents(data_dir: str):
    """Load every .pdf, .txt and .md file under data_dir."""
    docs = []

    pdf_loader = DirectoryLoader(
        data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    docs.extend(pdf_loader.load())

    for ext in ("**/*.txt", "**/*.md"):
        txt_loader = DirectoryLoader(
            data_dir, glob=ext, loader_cls=TextLoader, show_progress=True,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs.extend(txt_loader.load())

    if not docs:
        raise ValueError(
            f"No .pdf/.txt/.md files found under '{data_dir}'. "
            "Add some documents before running ingestion."
        )
    return docs


def split_documents(docs, chunk_size: int = 800, chunk_overlap: int = 120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vector_store(chunks, embedding_model: str, index_dir: str):
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    vector_store.save_local(index_dir)
    print(f"Saved FAISS index with {len(chunks)} chunks to '{index_dir}'")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into a FAISS vector store")
    parser.add_argument("--data_dir", default="data/sample_docs", help="Folder with source documents")
    parser.add_argument("--index_dir", default="vector_store", help="Where to persist the FAISS index")
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace embedding model name",
    )
    parser.add_argument("--chunk_size", type=int, default=800)
    parser.add_argument("--chunk_overlap", type=int, default=120)
    args = parser.parse_args()

    print(f"Loading documents from '{args.data_dir}' ...")
    docs = load_documents(args.data_dir)
    print(f"Loaded {len(docs)} raw documents")

    chunks = split_documents(docs, args.chunk_size, args.chunk_overlap)
    print(f"Split into {len(chunks)} chunks")

    build_vector_store(chunks, args.embedding_model, args.index_dir)


if __name__ == "__main__":
    main()
