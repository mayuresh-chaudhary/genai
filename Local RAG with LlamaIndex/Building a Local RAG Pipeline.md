# Building a Local RAG Pipeline with LlamaIndex, Ollama & Weaviate 

## Overview

This chapter demonstrates building a complete **Retrieval‑Augmented Generation (RAG)** system using only local, open-source components:

- **LlamaIndex** for document parsing, chunking, and retrieval
- **Ollama** with `gemma:2b` model for local LLM inference and embeddings
- **Weaviate** as a local vector store for semantic search
- **Document Summarizer** for preprocessing multiple file formats

By keeping everything local, you avoid cloud costs, maintain data privacy, and get full control over model selection and inference parameters.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline Flow                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input Files (PDF, DOCX, PPTX, VTT, HTML, TXT)                    │
│         ↓                                                           │
│  [Document Summarizer] → Extract text → ./docs/*.txt              │
│         ↓                                                           │
│  [LlamaIndex] Chunk with SentenceSplitter (512 char chunks)       │
│         ↓                                                           │
│  [Ollama Embeddings] nomic-embed-text (locally embedded)          │
│         ↓                                                           │
│  [Weaviate] Vector Store (local DB, HTTP @ :8080)                │
│         ↓                                                           │
│  User Query → [Retriever] Top-3 similarity search                 │
│         ↓                                                           │
│  Retrieved Context + Query → [Ollama gemma:2b] Generate Answer   │
│         ↓                                                           │
│  Final Response                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Document Summarizer Notebook

**Purpose**: Extract text from various file formats and optionally summarize.

**Key Functions**:

```python
# Configure local Ollama
OLLAMA_BASE = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'gemma:2b')

def call_llm(messages: List[Dict[str, str]], model: str | None = None, temperature: float = 0.1):
    """Call local Ollama chat endpoint and return parsed content."""
    url = f"{OLLAMA_BASE}/api/chat"
    payload = {"model": model or OLLAMA_MODEL, "messages": messages, "temperature": temperature}
    resp = requests.post(url, json=payload, timeout=120)
    data = resp.json()
    # Parse response and return content
    return {'content': extracted_text, 'raw': data}
```

**Supported Formats**:
- DOCX (Word), PPTX (PowerPoint), PDF, VTT (subtitles), HTML, TXT

**Workflow**:
1. Place files in `./input/`
2. Run extraction → saves `*_extracted.txt` to `./docs/`
3. Optionally generate summaries using `gemma:2b`

**Example Usage**:
```python
result = process_document(
    input_path="./input/requirements.pdf",
    output_dir="./docs",
    summarize=True,  # Generate AI summary
    max_summary_words=500
)
```

### 2. LlamaIndex RAG Notebook

**Purpose**: Index extracted documents in Weaviate and run retrieval-augmented queries.

**Key Setup**:

```python
# Configure Ollama LLM
llm = Ollama(
    model="gemma:2b",
    base_url="http://localhost:11434",
    temperature=0.1,
    keep_alive=120,  # Keep model in memory for 120 seconds
)

# Configure embeddings
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)

# Set global defaults
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.chunk_overlap = 50
```

**Weaviate Connection** (with fallback attempts):
```python
connection_attempts = [
    {"host": "localhost", "port": 8080, "grpc_port": 50051},
    {"host": "127.0.0.1", "port": 8080, "grpc_port": 50051},
]

for config in connection_attempts:
    try:
        weaviate_client = weaviate.connect_to_local(**config)
        break
    except Exception as e:
        print(f"Failed: {str(e)[:50]}")
        continue
```

**Query Interface**:
```python
def ask_question(question, show_sources=True):
    """Query the RAG index."""
    print(f"❓ Question: {question}")
    response = query_engine.query(question)
    print(f"💡 Answer: {response}")
    
    if show_sources and hasattr(response, 'source_nodes'):
        for i, node in enumerate(response.source_nodes, 1):
            print(f"\n  Source {i}: {node.text[:200]}...")
    return response
```

---

## Setup & Quick Start

### Prerequisites

1. **Python 3.10+** (tested on 3.12)
2. **Ollama** running locally:
   - Download from https://ollama.ai
   - Ensure `gemma:2b` model is available
3. **Weaviate** running locally (Docker or Podman):
   ```bash
   docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest
   # or
   podman run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest
   ```
4. **Python Dependencies**:
   ```bash
   pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama \
               llama-index-vector-stores-weaviate weaviate-client python-docx \
               python-pptx PyPDF2 webvtt-py beautifulsoup4 python-dotenv httpx
   ```

### Health Checks

Before running notebooks, verify services are running:

```bash
# Check Ollama
curl http://localhost:11434/api/show

# Check Weaviate
curl http://localhost:8080/v1/
```

### Running the Notebooks

**Step 1: Extract Documents**
- Open `document_summariser.ipynb`
- Place source files in `./input/`
- Run cells in order to extract and optionally summarize
- Outputs saved to `./docs/` as `.txt` files

**Step 2: Index & Query**
- Open `llama_index_rag.ipynb`
- Ensure `./docs/` has `.txt` files from step 1
- Run cells in order:
  1. Imports
  2. Configure Ollama LLM & embeddings
  3. Connect to Weaviate
  4. Load documents and create index
  5. Create query engine
  6. Run interactive queries with `ask_question(...)`

---

## Code Highlights

### Handling Large Documents

The document summarizer uses chunked summarization for large texts:

```python
def summarize_with_chunks(text: str, chunk_size: int = 10000, max_words: int = 500):
    """Summarize very long documents by chunking."""
    if len(text) <= chunk_size:
        return summarize_text_with_ai(text, max_words)
    
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Summarize each chunk
    chunk_summaries = []
    for chunk in chunks:
        summary = summarize_text_with_ai(chunk, max_words // len(chunks))
        chunk_summaries.append(summary)
    
    # Final summary of summaries
    combined = "\n\n".join(chunk_summaries)
    return summarize_text_with_ai(combined, max_words)
```

### Batch Processing

Extract and summarize multiple documents in one go:

```python
result = process_folder(
    input_folder="./input",
    output_dir="./docs",
    summarize=True,
    file_extensions=['.docx', '.pptx', '.pdf', '.vtt', '.html', '.htm', '.txt']
)
print(f"Processed: {result['successful']}/{result['total']} files")
```

### RAG Query with Source Attribution

```python
response = query_engine.query("Create Jira User Stories for Personalised Product Recommendation")

print("Answer:", response)
print("\nSources:")
for node in response.source_nodes:
    print(f"  - {node.text[:150]}...")
    print(f"    Relevance: {node.score}")
```

---

## Troubleshooting

### ReadTimeout from Ollama

**Symptom**: `httpx.ReadTimeout: timed out`

**Solution**:
- Increase `keep_alive` in Ollama client (e.g., `keep_alive=120`)
- Add retry/backoff logic around `query_engine.query()` calls
- Test Ollama directly to measure latency:
  ```bash
  curl -X POST "http://localhost:11434/api/chat" \
    -H "Content-Type: application/json" \
    -d '{"model":"gemma:2b","messages":[{"role":"user","content":"hello"}]}'
  ```

### Weaviate Connection Failed

**Symptom**: `Could not connect to Weaviate`

**Solution**:
- Verify container is running: `docker ps | grep weaviate`
- Check port mapping: `docker port <container_id>`
- Inspect logs: `docker logs <container_id>`
- Ensure mapped ports (8080, 50051) are not in use

### Model Not Available in Ollama

**Symptom**: `model "gemma:2b" not found`

**Solution**:
- List available models: `curl http://localhost:11434/api/show`
- Download model: `ollama pull gemma:2b` (from Ollama CLI)

### Slow Embedding/Indexing

**Symptom**: Indexing takes a long time

**Solution**:
- Use a faster embedding model (`nomic-embed-text` is already optimized)
- Reduce chunk size (e.g., `Settings.chunk_size = 256`)
- Use a lighter LLM for development (e.g., `mistral:latest` instead of `gemma:2b`)

---

## Next Steps & Improvements

1. **Persistence**: Save the Weaviate index to disk to avoid reindexing on every run
2. **Web UI**: Add a simple Streamlit or Gradio interface for queries
3. **Metrics**: Track retrieval latency, LLM response time, and accuracy
4. **Model Switching**: Experiment with different Ollama models (`mistral`, `neural-chat`, `orca`)
5. **Advanced Chunking**: Use semantic-based chunking instead of fixed sentence splits
6. **Multi-turn Chat**: Implement conversation history with context management

---

## Summary

This chapter provides a complete, reproducible local RAG pipeline:
- **Extract** text from multiple file formats
- **Summarize** using local LLMs (optional)
- **Index** embeddings in a local vector database
- **Retrieve** relevant context using semantic search
- **Synthesize** answers using local inference

All components run on your machine with no cloud dependencies, making it ideal for prototyping, private data, and cost-sensitive applications.

---

**Files in this chapter:**
- `document_summariser.ipynb` — extraction & summarization pipeline
- `llama_index_rag.ipynb` — indexing, retrieval & answer synthesis
- `./input/` — place source documents here
- `./docs/` — extracted `.txt` files (auto-generated)

