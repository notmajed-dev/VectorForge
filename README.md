# ⚡ VectorForge

> A lightweight, zero-dependency C++ vector database built from scratch — complete with an interactive 2D PCA visualizer, multi-index search (HNSW, KD-Tree, Brute Force), and a 100% local RAG pipeline powered by Ollama.

[![Language](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-blue)](#quickstart)
[![Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(header--only)-brightgreen)](#prerequisites)
[![LLM](https://img.shields.io/badge/Local%20AI-Ollama-purple)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)

---

## Why VectorForge?

Most vector databases today are massive systems wrapped in heavy runtimes, complicated setups, and cloud dependencies. If you just want to understand how vector indexing actually works under the hood — or want a fast, self-contained vector store for local experiments — you shouldn't need a Kubernetes cluster or a paid API key.

**VectorForge** was built to strip away the magic:
- **No external C++ libraries**: Built with vanilla C++17 and a single-header HTTP server ([`httplib.h`](file:///c:/Error404/VectorForge/httplib.h)).
- **Transparent algorithms**: Brute-force KNN, KD-Tree, and HNSW (Hierarchical Navigable Small World) implemented by hand so you can compare their trade-offs in real time.
- **Visual by design**: Includes a built-in dashboard that projects high-dimensional embeddings down to 2D using Principal Component Analysis (PCA) in an interactive HTML5 canvas.
- **Local RAG that actually works**: Ingest documents, chunk them on the fly, embed them through Ollama (`nomic-embed-text`), and query a local LLM (`llama3.2`) with zero telemetry or cloud bills.

---

## Core Features

- 🧠 **Three Search Engines**:
  - **HNSW**: Multi-layer navigable graph for sub-millisecond Approximate Nearest Neighbor (ANN) search.
  - **KD-Tree**: Spatial binary tree partitioning for exact low-to-mid dimensional pruning.
  - **Brute Force**: Exhaustive exact linear scan used as a baseline ground truth.
- 📐 **Multiple Distance Metrics**: Cosine Similarity, Euclidean ($L_2$), and Manhattan ($L_1$) distance functions.
- ⏱️ **Live Algorithm Benchmarking**: Run any query across all three indexing strategies side-by-side with microsecond-level latency tracking.
- 🎨 **Real-Time 2D PCA Canvas**: Watch semantic clusters (Computer Science, Math, Food, Sports, Documents) group together, and inspect neighbor connections when querying.
- 📚 **Full Local RAG Pipeline**:
  - Automated sliding-window word chunker (250 words with 30-word overlap).
  - Dynamic dimension support (adapts to any embedding size, e.g. 768D from Nomic).
  - Context grounding with Ollama LLMs directly in your browser.
- 🌐 **Clean HTTP REST API**: Query, insert, delete, inspect graph layers, or run RAG workflows via straightforward JSON endpoints.

---

## System Architecture

```text
 ┌───────────────────────────────────────────────────────────────┐
 │                   Browser UI (index.html)                    │
 │  2D PCA Scatter Canvas │ Vector Search │ Docs & Local RAG Chat│
 └───────────────────────────────▲───────────────────────────────┘
                                 │ HTTP (REST + JSON) :8080
 ┌───────────────────────────────▼───────────────────────────────┐
 │                      VectorForge Engine                       │
 │  ┌─────────────────────────────────────────────────────────┐  │
 │  │                      HTTP Router                        │  │
 │  │   /search   /insert   /benchmark   /hnsw-info   /doc/*  │  │
 │  └────────────────────────────┬────────────────────────────┘  │
 │                               │                               │
 │         ┌─────────────────────┴─────────────────────┐         │
 │         ▼                                           ▼         │
 │  ┌──────────────┐                            ┌──────────────┐ │
 │  │ VectorDB     │                            │ DocumentDB   │ │
 │  │ (16D Demo)   │                            │ (Ollama)     │ │
 │  ├──────────────┤                            ├──────────────┤ │
 │  │ • HNSW Graph │                            │ • HNSW Index │ │
 │  │ • KD-Tree    │                            │ • Text Store │ │
 │  │ • BruteForce │                            │ • Chunker    │ │
 │  └──────┬───────┘                            └──────┬───────┘ │
 └─────────┼───────────────────────────────────────────┼─────────┘
           │                                           ▼ HTTP :11434
           │ Metrics:                         ┌──────────────────┐
           │ • Cosine                         │      Ollama      │
           │ • Euclidean                      │ nomic-embed-text │
           │ • Manhattan                      │ llama3.2         │
           ▼                                  └──────────────────┘
   Top-K Nearest Items
```

---

## How It Works Under the Hood

### 1. HNSW (Hierarchical Navigable Small World)
HNSW builds a multi-layer graph where upper layers contain sparse, long-distance "expressways" between distant vectors, and lower layers contain dense, localized connections.
- **Search**: Starts at the top entry point, greedily routing to the closest node at each layer until it drops to layer 0, where it performs an expanded beam search (`efSearch`).
- **Insertion**: Assigns each vector an exponentially decaying maximum layer (`randLevel()`), connects it to its $M$ nearest neighbors, and prunes stale edges.

### 2. KD-Tree
A $k$-dimensional spatial partitioning tree that alternates split dimensions (`depth % dims`) at each level.
- Prunes entire sub-trees when the distance between the query point and the splitting hyperplane exceeds the current $k$-th best distance in the priority queue.

### 3. PCA Projection & Visualizer
To help you visualize high-dimensional embeddings on a 2D screen, the front-end runs **Power Iteration PCA** directly in JavaScript:
- Centers all vectors around their empirical mean.
- Computes the top two principal eigenvectors ($PC_1$ and $PC_2$) iteratively.
- Projects points onto the 2D plane with real-time hovering, category coloring, distance rays, and layer inspection.

---

## Quickstart

### Prerequisites
- A modern C++17 compiler (`g++`, `clang++`, or MSVC).
- (Optional, for RAG) [Ollama](https://ollama.com) installed and running.

### 1. Build & Run

#### On Linux / macOS
```bash
# Compile with optimizations and pthread
g++ -std=c++17 -O3 main.cpp -pthread -o vectorforge

# Run the server
./vectorforge
```

#### On Windows (MinGW / GCC)
```powershell
g++ -std=c++17 -O3 main.cpp -lws2_32 -o vectorforge.exe
.\vectorforge.exe
```

#### On Windows (MSVC)
```cmd
cl /EHsc /std:c++17 /O2 main.cpp /link ws2_32.lib wsock32.lib /out:vectorforge.exe
vectorforge.exe
```

### 2. Open the Dashboard
Once started, the server listens on `http://localhost:8080`.
Open your browser and navigate to:
```
http://localhost:8080
```
You'll see the demo dataset loaded and projected into 2D space immediately.

---

## Enabling Local RAG (Ask AI)

VectorForge includes a full Retrieval-Augmented Generation pipeline using Ollama. Everything runs on your machine — no API keys required.

1. **Install and start Ollama**:
   Download from [ollama.com](https://ollama.com), then start the daemon:
   ```bash
   ollama serve
   ```

2. **Pull the recommended models**:
   ```bash
   # Embeddings model (fast & high quality, 768 dimensions)
   ollama pull nomic-embed-text

   # Lightweight chat model
   ollama pull llama3.2
   ```

3. **Use the UI**:
   - Go to the **Documents** tab in the dashboard.
   - Paste in your text (lecture notes, documentation, articles).
   - Click **⚡ Embed & Insert** — VectorForge will slice your text into overlapping chunks, fetch real embeddings from Ollama, and index them in HNSW.
   - Switch to the **Ask AI** tab and ask questions. The engine will retrieve the most relevant context and generate an accurate, grounded response.

---

## API Reference

The server exposes clean REST endpoints for automation and scripts:

### Demo Vectors & Search
| Method | Endpoint | Query / Body | Description |
|---|---|---|---|
| `GET` | `/search` | `?v=0.1,0.2,...&k=5&metric=cosine&algo=hnsw` | Nearest neighbor search (algos: `hnsw`, `kdtree`, `bruteforce`) |
| `POST` | `/insert` | `{"metadata":"...","category":"cs","embedding":[...]}` | Insert a new vector into all indices |
| `DELETE` | `/delete/:id` | — | Remove a vector by ID |
| `GET` | `/items` | — | List all stored demo items |
| `GET` | `/benchmark` | `?v=0.1,0.2,...&k=5&metric=cosine` | Compares latency across BruteForce, KD-Tree, and HNSW (in $\mu s$) |
| `GET` | `/hnsw-info` | — | Graph topology metadata (layers, nodes, and edges) |
| `GET` | `/stats` | — | Database size, dimensions, supported algorithms, and metrics |

### Document Store & RAG
| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Automatically chunks text, embeds via Ollama, and stores in HNSW |
| `DELETE` | `/doc/delete/:id` | — | Delete a document chunk |
| `GET` | `/doc/list` | — | List all ingested document chunks with text previews |
| `POST` | `/doc/search` | `{"question":"...","k":3}` | Pure semantic search over documents (returns context chunks) |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | Full RAG pipeline: embeds question, retrieves context, prompts local LLM |
| `GET` | `/status` | — | Checks Ollama availability, active models, and doc counts |

---

## Example cURL Usage

#### 1. Search with custom parameters:
```bash
curl "http://localhost:8080/search?k=3&metric=cosine&algo=hnsw&v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.0,0.0,0.0,0.0,0.1,0.1,0.1,0.1"
```

#### 2. Insert a document into the RAG engine:
```bash
curl -X POST http://localhost:8080/doc/insert \
  -H "Content-Type: application/json" \
  -d '{"title":"Binary Trees","text":"A binary tree is a tree data structure in which each node has at most two children."}'
```

#### 3. Ask a question grounded on your documents:
```bash
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is a binary tree?","k":3}'
```

---

## Project Structure

```text
VectorForge/
├── main.cpp       # Core engine: VectorDB, DocumentDB, HNSW, KDTree, Ollama client, HTTP routes
├── index.html     # Interactive frontend: PCA canvas, algorithm switcher, latency meters, RAG chat
├── httplib.h      # Single-header C++ HTTP/HTTPS networking library
├── .gitignore     # Ignored binaries, object files, and IDE metadata
└── README.md      # You are here!
```

---

## Contributing & Experiments

Got ideas for:
- Product Quantization (PQ) or Scalar Quantization (SQ8) to shrink memory?
- SIMD acceleration for vector distance dot products (AVX2 / NEON)?
- Persistent disk storage (mmap / WAL)?

Pull requests, discussions, and benchmark experiments are welcome! Feel free to open an issue or submit a PR.

---

## License

VectorForge is open-source under the [MIT License](LICENSE).

<p align="center">
  <i>Built with C++, curiosity, and caffeine. </i>
</p>
