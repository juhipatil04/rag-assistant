# PupQuest — A Retrieval-Augmented Generation (RAG) Assistant

A fully local, interactive document intelligence system built to explore core techniques in applied LLM engineering. Upload any PDF and interrogate it through natural conversation, structured quiz modes, clinical role-play, or financial decision games — all grounded in verifiable citations from your document.

---

## What This Project Explores

PupQuest is a working implementation of **Retrieval-Augmented Generation**, a dominant pattern in production LLM systems that addresses a fundamental limitation of language models: their knowledge is frozen at training time and they hallucinate when asked about documents they haven't seen.

RAG solves this by giving the model a runtime retrieval step — the LLM never has to memorize your document; it only has to reason over the relevant excerpts retrieved on demand.

Beyond vanilla RAG, this project layers in:

- **Automatic document classification** with an LLM-as-classifier pattern
- **Mode-conditioned generation** — the same retrieval corpus drives fundamentally different interaction styles (quiz, role-play, decision scenarios, step challenges)
- **Conversational memory** over retrieved context, persisted across turns
- **LLM-generated follow-up suggestions** to guide information exploration
- **Page-level source citation** so every answer is traceable to specific document locations

---

## Architecture

```
                        ┌─────────────────────────────────┐
                        │           User (Browser)         │
                        └────────────────┬────────────────┘
                                         │ Upload PDF(s)
                                         ▼
                        ┌─────────────────────────────────┐
                        │         Streamlit Frontend       │
                        │         (app.py)                 │
                        └────────────────┬────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
           ┌──────────────┐    ┌──────────────────┐  ┌──────────────────┐
           │  PDF Renderer │    │  Ingestion        │  │  Groq LLM API    │
           │  (PyMuPDF)   │    │  Pipeline         │  │  (Llama 3.1 8B)  │
           └──────────────┘    └────────┬─────────┘  └──────────────────┘
                                        │                     ▲
                               ┌────────▼─────────┐          │
                               │  Text Splitter    │          │
                               │  (RecursiveChar)  │          │
                               └────────┬─────────┘          │
                                        │                     │
                               ┌────────▼─────────┐          │
                               │  Embedding Model  │          │
                               │  (all-MiniLM-L6)  │          │
                               └────────┬─────────┘          │
                                        │                     │
                               ┌────────▼─────────┐          │
                               │   ChromaDB        │──────────┘
                               │  Vector Store     │  top-k retrieval
                               └──────────────────┘
```

### Data Flow

1. **Ingestion** — PDFs are loaded with `PyPDFLoader`, split into 500-character chunks with 50-character overlap via `RecursiveCharacterTextSplitter`, and embedded with `all-MiniLM-L6-v2` (runs entirely on-device via SentenceTransformers).
2. **Indexing** — Chunk embeddings are stored in ChromaDB, an open-source vector database. The session uses an ephemeral in-memory instance; `ingest.py` writes a persistent `chroma_db/` on disk.
3. **Query** — At inference time, the user's question is embedded with the same model. ChromaDB returns the top-k=20 most semantically similar chunks.
4. **Generation** — Retrieved chunks are injected into the LLM prompt as context. Groq serves Llama 3.1 8B Instant, providing very low-latency completions (~200–400ms on typical hardware).
5. **Memory** — `ConversationBufferMemory` accumulates the full exchange and feeds it back into each new retrieval call, enabling multi-turn reference.

---

## Key AI / LLM Concepts Demonstrated

### Retrieval-Augmented Generation (RAG)
The `ConversationalRetrievalChain` in LangChain orchestrates the retrieve-then-read loop. For every user turn, the chain first retrieves relevant chunks (`k=20`), then constructs a prompt that includes those chunks alongside the conversation history before calling the LLM. This grounds the model's answer in the document rather than its parametric knowledge.

### Dense Passage Retrieval
Embeddings from `all-MiniLM-L6-v2` — a 22M-parameter bi-encoder model trained on over 1B sentence pairs — map text into a 384-dimensional semantic space where cosine similarity approximates meaning overlap. This is qualitatively different from keyword search: a query about "myocardial infarction" will retrieve passages discussing "heart attack" even with no lexical overlap.

### Chunking Strategy
`RecursiveCharacterTextSplitter` with `chunk_size=500, chunk_overlap=50` is a standard production heuristic. Smaller chunks improve retrieval precision (less irrelevant context per chunk); overlap ensures that concepts spanning chunk boundaries are not lost. This is an active research area — alternative approaches include semantic chunking, sentence-level splitting, and parent-document retrieval.

### LLM-as-Classifier
Before the chat session begins, the raw document text is passed to the LLM with a zero-shot classification prompt that maps documents into five categories: `lecture`, `medical`, `finance`, `howto`, `other`. This pattern — using a language model as a lightweight classifier rather than training a dedicated model — is increasingly common in production pipelines.

### Mode-Conditioned Prompting
Once the document type is known, the system selects a ranked list of interaction modes and injects a mode-specific system prompt at the start of the conversation. This demonstrates how the same retrieval infrastructure can produce radically different user experiences purely through prompt engineering — no fine-tuning required.

### Conversational Memory
`ConversationBufferMemory` with `return_messages=True` retains the full message history and appends it to each new retrieval call. This enables pronouns, references ("what did you mean by that?"), and follow-up questions to resolve correctly across turns. In production systems, this buffer must eventually be compressed or summarized as it grows — an open problem in long-context RAG.

### Source Attribution
Every response from the retrieval chain returns `source_documents`, a list of the actual chunks used. The app extracts page numbers from chunk metadata and renders them as citation pills. This traceability is critical for high-stakes domains (medicine, law, finance) where hallucination cannot be tolerated without recourse to primary sources.

---

## Project Structure

```
rag-assistant/
├── app.py              # Main Streamlit application (PupQuest UI)
├── ingest.py           # Standalone ingestion script — builds persistent ChromaDB
├── query.py            # Standalone CLI query loop — reads from ChromaDB on disk
├── requirements.txt    # Pinned dependencies
├── *.gif / *.jpg       # UI assets (animated banner, dog sprite)
└── chroma_db/          # Persisted vector store (created by ingest.py, gitignored)
```

### `app.py` — Interactive RAG Application
The primary application. Handles the full pipeline end-to-end in a session: upload → chunk → embed → index → classify → generate summary and suggestions → interactive chat. The vector store is ephemeral (held in memory for the session) and rebuilt on each upload, which trades persistence for simplicity.

### `ingest.py` — Offline Ingestion Script
A standalone ingestion pipeline that writes the ChromaDB index to disk. Useful for pre-building indices over large document corpora before launching the app, or for scripted batch ingestion.

### `query.py` — CLI Query Interface
A minimal REPL that reads the persisted ChromaDB created by `ingest.py` and runs a `RetrievalQA` chain. Demonstrates RAG without any UI layer — useful for debugging retrieval behavior or running headless evaluation.

---

## Interaction Modes

PupQuest automatically detects document type and surfaces the most relevant modes first, though all modes are available for all document types.

| Mode | Best For | Behavior |
|------|----------|----------|
| **Quiz** | Lectures, textbooks | LLM generates multiple-choice questions, grades answers, and advances through the document |
| **Role Play** | Medical, psychology | LLM adopts the role of a patient presenting symptoms; user plays clinician |
| **Decision Game** | Finance, business | LLM presents scenarios from the document; user chooses from options and receives consequence analysis |
| **Step Challenge** | How-to guides, manuals | LLM presents the first step; user must identify what comes next |
| **Chat** | All document types | Standard conversational RAG with full citation and follow-up suggestions |

Modes other than Chat run the LLM without the retrieval chain — context is loaded from the PDF at session start and injected directly into the system prompt. This is a deliberate design choice: game modes prioritize interactivity and narrative coherence over precise document attribution.

---

## Setup

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier available)

### Installation

```bash
git clone https://github.com/juhipatil04/rag-assistant.git
cd rag-assistant

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Environment

Create a `.env` file (or export directly):

```bash
export GROQ_API_KEY=your_key_here
```

### Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Standalone Scripts (Optional)

```bash
# Pre-ingest a PDF to disk
python ingest.py

# Query from the CLI
python query.py
```

---

## Technology Stack

| Component | Library | Role |
|-----------|---------|------|
| UI | Streamlit 1.57 | Web interface and session state |
| PDF rendering | PyMuPDF (fitz) | Per-page raster preview at 130 DPI |
| PDF text extraction | LangChain PyPDFLoader | Page-level document loading |
| Text splitting | RecursiveCharacterTextSplitter | Chunk production with overlap |
| Embedding model | `all-MiniLM-L6-v2` (SentenceTransformers) | Local dense embeddings, 384-dim |
| Vector store | ChromaDB 1.5 | ANN similarity search |
| LLM | Llama 3.1 8B Instant via Groq | Completion, classification, summarization |
| LLM client | LangChain-Groq | LangChain-compatible Groq wrapper |
| RAG chain | ConversationalRetrievalChain | Retrieve-then-read with memory |
| Memory | ConversationBufferMemory | Full-history multi-turn context |

---

## Design Decisions and Trade-offs

**Why Groq / Llama 3.1 8B?**
Groq's LPU inference provides ~750 tokens/second throughput, making latency negligible for interactive use. An 8B model is sufficient for retrieval-grounded tasks where the model's job is reasoning over provided context rather than broad world knowledge.

**Why `all-MiniLM-L6-v2`?**
It is fast, small (80MB), runs entirely on CPU, and achieves strong performance on semantic similarity benchmarks. For most PDF Q&A use cases, retrieval quality bottlenecks on chunk quality and `k` selection rather than embedding model expressiveness.

**Why ephemeral ChromaDB in `app.py`?**
Rebuilding the index on each upload keeps session state simple and avoids stale-index bugs when users upload new documents. The trade-off is that large documents take longer to process. A production system would use a persistent or remote vector store with document deduplication.

**Why `k=20` chunks?**
Higher `k` improves recall at the cost of increased context length and thus LLM latency and cost. With 500-character chunks and `k=20`, the retrieved context is roughly 10,000 characters — comfortably within the 128K context window of Llama 3.1. Lowering `k` would reduce latency; increasing it would improve coverage for broad queries.

**Game modes without retrieval**
Game modes inject a fixed amount of document text at session start rather than retrieving per-turn. This avoids mid-game context shifts that would break the narrative, but means responses are not page-cited. For quiz and role-play, coherence matters more than attribution.

---

## Limitations and Future Directions

- **Context window degradation** — As `ConversationBufferMemory` grows, older context gets pushed out. Summarization-based memory (`ConversationSummaryMemory`) or a sliding window would mitigate this.
- **Chunk-level citation** — Currently citations show page numbers; chunk-level highlights (text selection in the PDF preview) would improve traceability.
- **Multi-modal documents** — Charts, figures, and tables are not extracted; a vision-capable model pipeline (e.g., `colpali` for document retrieval) would extend coverage.
- **Retrieval evaluation** — There is no automated evaluation of retrieval quality (precision@k, MRR, NDCG). Adding a `ragas` or `BEIR`-style eval harness would enable systematic prompt and chunking experiments.
- **Persistent user sessions** — Each page refresh reinitializes state. Adding a session backend (Redis, SQLite) would support longer research workflows.
- **Hybrid retrieval** — Combining dense retrieval (current) with sparse BM25 retrieval (keyword matching) typically improves recall for named entities, codes, and domain-specific terminology not well-represented in the embedding space.


*Built as an exploration of applied LLM engineering — RAG pipelines, prompt engineering, and interactive document intelligence.*
