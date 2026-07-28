# 📚 RAG Document Q&A System

An AI-powered **Retrieval-Augmented Generation (RAG)** application that answers questions using your own documents instead of relying only on the language model's knowledge. The system processes PDF, TXT, and Markdown files, retrieves the most relevant document sections using semantic search, and generates accurate responses with **Qwen2.5**.

---

## 🚀 Features

- 📄 Supports PDF, TXT, and Markdown documents
- 🔍 Semantic document retrieval using FAISS
- 🧠 Sentence Transformer embeddings
- 🤖 Open-source Qwen2.5 language model
- 💬 Interactive command-line chat mode
- ⚡ Fast local vector database
- 📊 Retrieval evaluation support
- 🔄 Easily replaceable embedding and LLM models

---

# 🏗️ Architecture

```
              Documents (PDF / TXT / MD)
                         │
                         ▼
                 Document Chunking
                         │
                         ▼
      Sentence Transformer Embeddings
                         │
                         ▼
              FAISS Vector Database
                         │
                         ▼
                   User Question
                         │
                         ▼
          Semantic Similarity Search
                         │
                         ▼
          Top Relevant Document Chunks
                         │
                         ▼
          Qwen2.5 Instruction Model
                         │
                         ▼
           Context-Aware AI Response
```

---

# 📂 Project Structure

```
rag-document-qa
│
├── src
│   ├── ingest.py
│   ├── rag_pipeline.py
│   └── evaluate.py
│
├── requirements.txt
├── README.md
└── documents/
```

---

# 🛠️ Technologies Used

- Python
- LangChain
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- Qwen2.5
- PyTorch

---

# 📦 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/rag-document-qa.git
```

Move to the project folder

```bash
cd rag-document-qa
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

## 1. Ingest Documents

```bash
python src/ingest.py
```

---

## 2. Start the RAG Chatbot

```bash
python src/rag_pipeline.py
```

---

## 3. Evaluate Retrieval

```bash
python src/evaluate.py
```

---

# 🔄 Workflow

1. Load documents.
2. Split documents into chunks.
3. Generate embeddings using Sentence Transformers.
4. Store embeddings in a FAISS vector database.
5. Accept user queries.
6. Retrieve the most relevant document chunks.
7. Provide retrieved context to the Qwen2.5 model.
8. Generate an accurate, grounded response.

---

# 📋 Requirements

- Python 3.10+
- langchain
- faiss-cpu
- sentence-transformers
- transformers
- torch
- accelerate
- pypdf

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

# 🎯 Advantages

- No paid APIs required
- Local document search
- Fast semantic retrieval
- Accurate context-based responses
- Easy to customize
- Supports multiple document formats

---

# 📈 Future Improvements

- Streamlit web interface
- FastAPI REST API
- OCR support for scanned PDFs
- Hybrid search (BM25 + Dense Retrieval)
- Conversation memory
- Docker deployment
- Multi-document collections

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push your branch.
5. Open a Pull Request.

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Giri Venkateswara Rao**

AI & Full Stack Developer

---

⭐ If you found this project helpful, please give it a Star!
