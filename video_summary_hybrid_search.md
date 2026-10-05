# Video Summary: Hybrid Search & Retrieval Masterclass

- **File Path**: [`/Users/dssanthosh/workspace/anti-gravity-workspace/runpod-demo/input.mp4`](file:///Users/dssanthosh/workspace/anti-gravity-workspace/runpod-demo/input.mp4)
- **Duration**: 03:55:15 (3 Hours, 55 Minutes, 15 Seconds)
- **Resolution & Codec**: 1280×720 @ 25 fps (H.264 / AAC 48kHz Stereo)
- **File Size**: 819.2 MB
- **Instructor / Speaker**: Pradeep Pai Kateel (Interview Kickstart)
- **Session Topic**: *Hybrid Search & Retrieval: Dense Vectors, Sparse Vectors, SPLADE, HNSW, Reciprocal Rank Fusion (RRF), Qdrant Vector Database & FastEmbed*

---

## Executive Summary & Architecture Overview

This masterclass is a deep-dive, industry-grade technical workshop designed for machine learning and AI systems engineers. It addresses the fundamental challenges of modern Information Retrieval (IR) and Retrieval-Augmented Generation (RAG) pipelines: **Why neither pure keyword search (lexical) nor pure vector search (dense semantic) is sufficient on its own**, and how modern production search engines fuse both into a **Hybrid Search Pipeline**.

### Key Architectural Takeaways

1. **The Semantic-Lexical Duality**:
   - **Dense Embeddings (e.g., BGE-Large, OpenAI text-embedding-3)** capture conceptual similarity, semantic intent, synonyms, and cross-lingual meaning, but struggle with precise product codes, serial numbers, acronyms, and rare keywords.
   - **Classic Sparse Lexical Methods (TF-IDF, BM25)** excel at exact keyword precision, but suffer from the *vocabulary mismatch problem* and lack contextual awareness.
   - **Learned Sparsity (SPLADE)** bridges this gap by using a Masked Language Model (BERT/DistilBERT) to project text into the model's vocabulary space (~30,522 dimensions), performing automatic **Term Expansion** while retaining sparse inverted-index efficiency.

2. **Approximate Nearest Neighbor (ANN) Indexing via HNSW**:
   - Brute-force $k$-NN scales as $O(N \cdot D)$, making it infeasible for millions of vectors.
   - **Hierarchical Navigable Small World (HNSW)** structures vectors into multi-layer proximity graphs inspired by skip-lists, providing logarithmic $O(\log N)$ search complexity with high recall.

3. **Rank Fusion (Reciprocal Rank Fusion - RRF)**:
   - Dense cosine scores (ranging from $-1$ to $1$ or $0$ to $1$) and sparse scores (BM25 or SPLADE dot products ranging from $0$ to $\infty$) have incompatible distributions that cannot simply be added together without distortion.
   - **RRF** bypasses score calibration by fusing *order ranks* using the formula:
     $$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
     where $k \approx 60$ acts as a smoothing parameter to prevent top-ranked outliers from dominating.

4. **Production Implementation Stack**:
   - **Vector Database**: Qdrant (using named dual-vector schema: `"text-dense"` with Cosine distance and `"text-sparse"` with DotProduct).
   - **Embedding Engine**: FastEmbed (`BAAI/bge-large-en-v1.5` for dense, `prithivida/Splade_PP_en_v1` for sparse).
   - **Dataset**: Amazon ESCI (Shopping Queries / Product Catalog).

---

## Detailed Chronological Breakdown

```
[00:00:00 - 00:25:00] Part 1: Course Overview & Foundation
[00:25:00 - 01:05:00] Part 2: Dense Vectors, Embedding Spaces & Distance Metrics
[01:05:00 - 01:45:00] Part 3: Approximate Nearest Neighbors (ANN) & HNSW Deep Dive
[01:45:00 - 02:20:00] Part 4: Sparse Vectors, Inverted Indexes & SPLADE Term Expansion
[02:20:00 - 03:00:00] Part 5: Hybrid Search Architecture & Rank Fusion (RRF vs RSF)
[03:00:00 - 03:55:15] Part 6: Live Hands-On Google Colab Workshop & Code Implementation
```

---

### Part 1: Course Overview & Foundations (00:00:00 - 00:25:00)
- **UI Context**: Google Slides (`docs.google.com`), Speaker video feed (top right).
- **Core Learning Objectives**:
  - Dense Vectors vs. Sparse Vectors
  - Approximate Nearest Neighbor (ANN) search algorithms
  - Navigable Small World (NSW) & Hierarchical Navigable Small World (HNSW)
  - Learned Sparsity & SPLADE (Sparse Lexical AnD Expansion)
  - Hybrid Search Pipeline architecture
  - Reciprocal Rank Fusion (RRF)
- **Recap on Vector Databases**:
  - Storing and indexing high-dimensional embeddings for low-latency similarity queries.
  - Production capabilities: CRUD, metadata filtering, horizontal scaling, strict mode constraints, and hybrid indexing.

---

### Part 2: Dense Vectors, Embedding Spaces & Metrics (00:25:00 - 01:05:00)
- **UI Context**: Switching between Google Slides and Miro Whiteboard (`miro.com`).
- **Dense Vector Characteristics**:
  - Continuous float arrays of fixed dimensions ($D \in \{384, 768, 1024, 1536\}$).
  - Almost all dimensions have non-zero values.
  - Encodes semantic meaning learned from transformer encoders.
- **Distance Metrics Compared**:
  - **Euclidean Distance ($L_2$ Norm)**:
    $$d(u, v) = \sqrt{\sum_{i=1}^D (u_i - v_i)^2}$$
    Sensitive to vector magnitude.
  - **Dot Product**:
    $$\langle u, v \rangle = \sum_{i=1}^D u_i v_i$$
    Combines angle and magnitude; identical to cosine similarity when vectors are $L_2$-normalized.
  - **Cosine Similarity**:
    $$\cos(\theta) = \frac{u \cdot v}{\|u\| \|v\|}$$
    Measures angular distance, invariant to length.

---

### Part 3: ANN Search & HNSW Graph Indexing (01:05:00 - 01:45:00)
- **UI Context**: Miro Board whiteboard hand-drawn graphs transitioning to Google Slides diagrams.
- **The Problem with Exhaustive Search**:
  - Comparing query vector $q$ against $N$ vectors requires $N$ distance computations: $O(N \cdot D)$.
  - Unacceptable at scale ($N > 10^6$).
- **Navigable Small Worlds (NSW)**:
  - Vectors are graph nodes; edges connect nearby neighbors with several long-range "highways".
  - Greedy search: Start at an entry point, compute distance to all neighbors, hop to the closest neighbor, and repeat until no closer neighbor exists.
  - Problem: Prone to getting stuck in local minima; search complexity scales poorly on high-density clusters.
- **Hierarchical Navigable Small Worlds (HNSW)**:
  - Extends NSW using the concept of **Skip Lists**.
  - **Layered Hierarchy**:
    - **Top Layer ($L_{\max}$)**: Sparsest layer with long-range links for coarse global navigation.
    - **Intermediate Layers**: Progressively denser graphs.
    - **Bottom Layer (Layer 0)**: Contains every vector in the dataset connected with local proximity links.
  - **Search Traversal**:
    1. Start at top layer entry point.
    2. Greedy route to local minimum on current layer.
    3. Drop down to the next layer using the found node as the new entry point.
    4. Repeat until Layer 0 is reached; perform beam search with priority queue of size $efSearch$.
  - **Key Hyperparameters**:
    - $M$: Max number of bi-directional links per node (typical: 16 to 64).
    - $efConstruction$: Search queue size during indexing (controls build quality vs. build speed).
    - $efSearch$: Search queue size at query time (controls recall vs. query latency).

---

### Part 4: Sparse Vectors, Inverted Indexes & SPLADE (01:45:00 - 02:20:00)
- **UI Context**: Miro board derivations of posting lists, then Google Slides on Learned Sparsity.
- **Classic Sparse Vectors (TF-IDF & BM25)**:
  - Dimensionality equals the size of the vocabulary ($|V| \approx 30,000 - 100,000$).
  - Vectors are 99.9% zeros; only words explicitly present in the document have non-zero weights.
  - Stored in an **Inverted Index** (Posting list: $\text{term} \to [(\text{doc\_id}_1, w_1), (\text{doc\_id}_2, w_2), \dots]$).
  - **Failure Modes**:
    - Synonym blind spot: "automobile" does not match "car".
    - Spelling variations and conceptual queries fail.
- **SPLADE (Sparse Lexical AnD Expansion)**:
  - Uses a pretrained Masked Language Model (BERT) with a linear head over the vocabulary.
  - Predicts token relevance across the entire vocabulary for a given input sentence.
  - **Term Expansion**: Non-occurring but semantically relevant terms get non-zero weights!
    - *Example*: For `"Fastembed is a great library for text embeddings!"`, SPLADE assigns weights to `"fast"`, `"text"`, `"library"`, but also expands to `"software"`, `"tool"`, `"database"`, `"wonderful"`.
  - Stored inside standard sparse vector posting lists, enabling lightning-fast inverted index retrieval while gaining semantic intelligence.

---

### Part 5: Hybrid Search Architecture & Rank Fusion (02:20:00 - 03:00:00)
- **UI Context**: Slides showing the dual-branch retrieval pipeline and Miro Board rank matrix calculation.
- **Pipeline Architecture**:
  ```
               ┌──> [Dense Model: BGE-Large] ──> Vector Search (HNSW)  ──> Ranked List A
  Query Text ──┤
               └──> [Sparse Model: SPLADE]   ──> Inverted Index Search ──> Ranked List B
                                                        │
                                                        ▼
                                           Reciprocal Rank Fusion (RRF)
                                                        │
                                                        ▼
                                              Final Unified Ranking
  ```
- **Why Simple Score Addition Fails**:
  - Dense cosine scores: $[-1, 1]$ (usually clustered between $0.6$ and $0.9$).
  - Sparse dot products: unbounded $[0, \infty)$ (often between $5$ and $40$).
  - Adding or averaging requires score normalization (Min-Max, Z-score) which is sensitive to distribution drift and query difficulty.
- **Reciprocal Rank Fusion (RRF)**:
  - Score independent, rank-based algorithm:
    $$\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
  - Where $k = 60$ (standard Cormack et al. constant).
  - Penalizes documents ranked low in any list, strongly rewards documents appearing near the top of multiple lists (consensus).

---

### Part 6: Hands-On Google Colab Implementation (03:00:00 - 03:55:15)
- **UI Context**: Google Colab (`Class Notebook.ipynb - Colab`), dark theme, 10 numbered sections.

#### 1. Environment & Package Imports
```python
import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    SparseVector,
    PointStruct,
    QueryRequest,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    ScoredPoint,
)
from transformers import AutoTokenizer
import fastembed
from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding

print(f"FastEmbed version: {fastembed.__version__}")
# Output: FastEmbed version: 0.8.0
```

#### 2. Dataset Preparation (Amazon ESCI Shopping Queries)
```python
# Load dataset from Hugging Face
source_df = dataset.to_pandas()

# Deduplicate products across multiple search queries
df = source_df.drop_duplicates(
    subset=["product_text", "product_title", "product_bullet_point", "product_brand"]
)

# Drop missing values in essential text columns
df = df.dropna(
    subset=["product_text", "product_title", "product_bullet_point", "product_brand"]
)

print(f"Unique products in catalog: {len(df)}")        # 176 products in sample subset
print(f"Total search queries: {len(source_df)}")       # 919 queries

# Combine text fields to give embedding models richest signal
df["combined_text"] = (
    df["product_title"] + "\n" +
    df["product_text"] + "\n" +
    df["product_bullet_point"]
)
product_texts = df["combined_text"].tolist()
```

#### 3. Model Loading (FastEmbed Dense & Sparse)
```python
# Models used
sparse_model_name = "prithivida/Splade_PP_en_v1"
dense_model_name = "BAAI/bge-large-en-v1.5"

sparse_model = SparseTextEmbedding(model_name=sparse_model_name, batch_size=32)
dense_model = TextEmbedding(model_name=dense_model_name, batch_size=32)

print("Both models loaded!")

# Helper functions
def make_sparse_embedding(texts: list[str]) -> list[SparseEmbedding]:
    return list(sparse_model.embed(texts, batch_size=32))

def make_dense_embedding(texts: list[str]):
    return list(dense_model.embed(texts))
```

#### 4. Embedding Generation
```python
# Sparse embeddings via SPLADE
df["sparse_embedding"] = make_sparse_embedding(product_texts)

# Dense embeddings via BGE-Large (1024 dimensions)
df["dense_embedding"] = make_dense_embedding(product_texts)
```

#### 5. Qdrant Collection Creation with Dual-Vector Schema
```python
client = QdrantClient(":memory:")  # In-memory for demo; url="http://localhost:6333" for prod
collection_name = "esci"

client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "text-dense": VectorParams(
            size=1024,                  # BGE-Large dimension
            distance=Distance.COSINE    # Cosine similarity for dense
        )
    },
    sparse_vectors_config={
        "text-sparse": SparseVectorParams(
            index=SparseIndexParams(
                on_disk=False           # Kept in RAM for maximum speed
            )
        )
    }
)
print(f"Collection '{collection_name}' created with dense + sparse support.")
```

#### 6. Ingesting Data into Qdrant (`PointStruct`)
```python
def make_points(df: pd.DataFrame) -> list[PointStruct]:
    sparse_vectors = df["sparse_embedding"].tolist()
    product_texts = df["combined_text"].tolist()
    dense_vectors = df["dense_embedding"].tolist()
    rows = df.to_dict(orient="records")

    points = []
    for idx, (text, s_vec, d_vec) in enumerate(
        zip(product_texts, sparse_vectors, dense_vectors)
    ):
        sparse_vector = SparseVector(
            indices=s_vec.indices.tolist(),
            values=s_vec.values.tolist()
        )
        point = PointStruct(
            id=idx,
            payload={
                "text": text,
                "product_id": rows[idx]["product_id"],
                "product_title": rows[idx]["product_title"]
            },
            vector={
                "text-sparse": sparse_vector,
                "text-dense": d_vec.tolist()
            }
        )
        points.append(point)
    return points

points = make_points(df)
client.upsert(collection_name=collection_name, points=points)
print(f"Successfully indexed {len(points)} points into Qdrant.")
```

#### 7. Parallel Batch Querying (Dense + Sparse)
```python
query_text = "revent 80 cfm"
top_k = 10

# Embed query with both models
query_sparse_vectors = make_sparse_embedding([query_text])
query_dense_vector = make_dense_embedding([query_text])

# Execute both searches in a single atomic batch call
search_results = client.query_batch_points(
    collection_name=collection_name,
    requests=[
        # Branch 1: Dense Semantic Similarity
        QueryRequest(
            query=query_dense_vector[0].tolist(),
            using="text-dense",
            limit=top_k,
            with_payload=True
        ),
        # Branch 2: Sparse Lexical Exact Match
        QueryRequest(
            query=SparseVector(
                indices=query_sparse_vectors[0].indices.tolist(),
                values=query_sparse_vectors[0].values.tolist()
            ),
            using="text-sparse",
            limit=top_k,
            with_payload=True
        )
    ]
)
```

#### 8. Reciprocal Rank Fusion (RRF) Implementation
```python
def rrf(rank_lists: list[list[tuple]], alpha: int = 60, default_rank: int = 1000):
    all_items = set()
    for rl in rank_lists:
        for item, _ in rl:
            all_items.add(item)
            
    item_to_index = {item: i for i, item in enumerate(all_items)}
    rank_matrix = np.full((len(all_items), len(rank_lists)), default_rank)
    
    for list_idx, rank_list in enumerate(rank_lists):
        for item, rank in rank_list:
            rank_matrix[item_to_index[item], list_idx] = rank
            
    # RRF formula: sum of 1 / (alpha + rank)
    rrf_scores = np.sum(1.0 / (alpha + rank_matrix), axis=1)
    sorted_indices = np.argsort(-rrf_scores)
    
    sorted_items = [
        (list(item_to_index.keys())[idx], rrf_scores[idx])
        for idx in sorted_indices
    ]
    return sorted_items

# Convert Qdrant ScoredPoint results to (id, rank) pairs
def rank_list(search_result: list[ScoredPoint]):
    return [(point.id, rank + 1) for rank, point in enumerate(search_result)]

dense_rank_list = rank_list(search_results[0])
sparse_rank_list = rank_list(search_results[1])

# Apply fusion
fused_rank_list = rrf([dense_rank_list, sparse_rank_list], alpha=60)

# Fetch final ranked items
fused_records = client.retrieve(
    collection_name=collection_name,
    ids=[item[0] for item in fused_rank_list[:top_k]]
)
record_by_id = {r.id: r for r in fused_records}

print(f"\nHYBRID SEARCH RESULTS for: '{query_text}'")
print("=" * 70)
for rank, (item_id, score) in enumerate(fused_rank_list[:top_k], 1):
    rec = record_by_id[item_id]
    title = rec.payload["text"].split("\n")[0][:75]
    print(f"Rank {rank:02d} | Score: {score:.5f} | ID: {item_id} | {title}")
```

---

## Evaluation & Critical Insights from the Session

1. **When Dense Fails**:
   - Query: `"revent 80 cfm"` (a specific brand name and fan specification).
   - Dense model returns generic bathroom fans, quiet fans, or air-circulating blowers that match the general meaning of ventilation, but misses the specific brand (`Revent`) and airflow capacity (`80 CFM`).
2. **When Sparse Fails**:
   - Query: `"silent exhaust for small room"`.
   - Classic BM25 completely misses products that say "quiet bathroom ventilation fan" because the words "silent", "exhaust", and "small" don't match literally.
   - SPLADE partially recovers through term expansion, but dense embedding understands the conceptual nuance immediately.
3. **Hybrid Victory**:
   - By fusing both via RRF, items that satisfy both semantic intent and precise keyword parameters consistently achieve Rank 1.
