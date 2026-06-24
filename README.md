# Local RAG & Vector DB Sandbox Laboratory 🧪

An interactive, educational **RAG (Retrieval-Augmented Generation)** dashboard and playground built to run 100% locally on your machine. This repository is designed to help engineers master document chunking, indexing, hybrid retrieval, vector databases, and LLM orchestration without setting up paid Google Cloud Platform (GCP) configurations.

---

## 🌟 Key Features

* **Interactive Chunking Playground**: Adjust parameters (chunk size, overlap, strategy) and visualize the text splits.
* **Hybrid Search Engine**: Combines **Dense Retrieval (FAISS)** and **Sparse Retrieval (BM25)** using **Reciprocal Rank Fusion (RRF)**.
* **Orchestrated Generations**: Use 100% offline models via **Ollama**, cloud-based models via **Google Gemini API (Free developer key)**, or an **Offline Simulator**.
* **Visual Search Trace**: Inspect the pipeline thought process including search ranks, similarities, and combined RRF scores.
* **Vector DB Architecture Guide**: Learn about FAISS vs. Pinecone vs. Vertex AI Vector Search.

---

## 📐 RAG Core Insights

### 1. Chunking Strategies
* **Fixed-Size Chunking**: Splits text strictly at char-count borders (e.g. 500 characters). Simple but often breaks sentences or headers.
* **Recursive Character Chunking**: Splits text recursively by a sequence of characters (`\n\n`, `\n`, `. `, ` `, `""`) to keep paragraphs and sentences whole.

### 2. Dense vs. Sparse Search
* **Dense (FAISS)**: Encodes text into 384-dimension embeddings (`all-MiniLM-L6-v2`) to perform semantic search. Matches concepts rather than words (e.g., matching "puppy" when query is "young dog").
* **Sparse (BM25)**: Evaluates term frequency and inverse document frequency to find keyword matches. Matches exact product IDs, serial codes, or specific terms.

### 3. Reciprocal Rank Fusion (RRF)
RRF combines the rankings of multiple retrieval systems (Sparse + Dense) to optimize final relevance. The formula is:

$$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where $r_m(d)$ is the rank of document $d$ in retrieval system $m$, and $k$ is a smoothing constant (typically set to 60). This ensures document relevance is maximized even if one system ranks a result high while another ranks it low.

---

## 📊 Vector Database Comparison

| Feature | FAISS | Pinecone | Vertex AI Vector Search |
| :--- | :--- | :--- | :--- |
| **Architecture** | Local/In-Memory library | Managed SaaS Cloud DB | Enterprise GCP Infrastructure |
| **Pricing** | Free / Open-source | Free tier + Paid serverless | Paid GCP Resource |
| **Latency** | Extremely low (< 1ms) | Medium (Network overhead) | Low (Distributed cloud index) |
| **Persistence** | Manual file serialization | Auto-persisted in cloud | Auto-persisted on GCP buckets |
| **Scalability** | Up to millions (Single Node) | Billions (Managed) | Billions (Enterprise-scale) |
| **Best For** | Edge devices, local testing | Production SaaS MVP | Enterprise GCP production |

---

## 🚀 Setup & Running Locally

### Prerequisites
* Python 3.10 or higher.
* *Optional*: [Ollama](https://ollama.com/) running locally for offline LLM support.

### 1. Environment Setup
Clone the repository and run:

```bash
# Create python virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt
```

### 2. Launch the Application
Double-click `run.bat` (Windows) or execute:

```bash
python backend/main.py
```

Open your browser and navigate to **`http://localhost:8000`**.

---

## 🤖 Configuring LLM Providers

1. **RAG Simulation**: The default offline mode. It returns a mockup showing the exact chunk extraction and formatting, without calling any external services.
2. **Google Gemini**: Get a free API key from [Google AI Studio](https://aistudio.google.com/). Input it into the UI configuration bar to test generation.
3. **Ollama**: Install Ollama, pull a model (e.g., `ollama pull llama3`), run it locally, and choose **Ollama** in the UI dropdown.
