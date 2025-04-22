# Resume-Based Job Recommender System

### CSL7110 – Machine Learning with Big Data

**Contributors:**
- Mudit Kumar (M24DE3051)
- Likhith Gunjal (M24DE3044)

---

## 💡 Project Overview

This project presents a scalable and efficient Resume-to-Job Matching system using three main techniques:
1. **TF-IDF + Cosine Similarity**
2. **SBERT (Sentence-BERT)**
3. **MinHash + Locality Sensitive Hashing (LSH)**

Each approach is benchmarked on a job dataset with 1M records. The system extracts resume content, transforms it into comparable embeddings, and matches it with job descriptions based on similarity.

---

## ⚖️ Theoretical Background

### TF-IDF
- Sparse vector-based similarity
- Good for keyword matching
- Lightweight and interpretable

### SBERT
- Dense semantic vector using Transformer architecture
- Best semantic understanding (Cosine Similarity of contextual embeddings)
- Uses `all-MiniLM-L6-v2` model for efficiency

### MinHash + LSH
- Fast approximate matching using shingling and hashing
- Ideal for massive datasets
- Uses `datasketch` library and 128 permutations

---

## ⚙️ Methodology

### Common Preprocessing
- PDF resume text extraction using `PyMuPDF`
- Cleaning (lowercase, remove punctuation, whitespace)
- Weighted text construction:
  - Job Title (3.0)
  - Role (2.5)
  - Skills (2.0)
  - Job Description (1.0)
  - Company (0.8)

### TF-IDF
- `TfidfVectorizer` from `sklearn`
- N-grams: (1,2), stopwords removed
- Similarity via cosine distance
- Embeddings cached with `joblib`

### SBERT
- `SentenceTransformer` from HuggingFace
- Chunked and batched embedding (10,000 chunks, batch size 256)
- GPU-supported (if available)
- Cached embeddings as `.npy`

### MinHash + LSH
- 5-gram shingles
- `MinHashLSHForest` with 128 permutations
- Jaccard similarity for final filtering
- Parallelized and cached

---

## ⏳ Performance Benchmarks

| Approach     | Time (1M jobs)       | Score (sample) | Caching Impact |
|--------------|----------------------|----------------|----------------|
| **TF-IDF**   | 8.9s (cached)        | 0.403          | ✅            |
| **SBERT**    | 8s (GPU, cached)     | 0.628          | ✅            |
| **LSH**      | 6s (cached) / 2239s  | 0.140          | ✅            |

---

## 📊 Results Summary

All models successfully identified relevant jobs such as:
- "Digital Marketing Specialist"
- "Social Media Manager"

Differences in employer and location highlight sensitivity to various fields.
- TF-IDF: Matched with **Advanced Micro Devices**
- SBERT: Matched with **Associated British Foods**
- LSH: Matched with **Community Health Systems**

---

## 🔗 Integration and Deployment

- ✔ Snowflake backend for job descriptions
- ✔ Streamlit UI integration
- ✔ GitHub: [Project Repository](https://github.com/MuditKumarR/MLBD_Project_M24DE3051)

---

## 🧵 Recommended Architecture

1. Use LSH to **filter top 1000 jobs**
2. Apply SBERT to **refine top 10**
3. Deploy via Streamlit with PDF upload and dynamic pagination (20/page)

---

## 🛠️ Tech Stack
- **PyMuPDF** - Resume parsing
- **scikit-learn** - TF-IDF
- **sentence-transformers** - SBERT embeddings
- **datasketch** - MinHash & LSH
- **Snowflake** - Job table backend
- **Streamlit** - Web application

---

## 🎯 Future Enhancements
- FAISS integration for real-time search
- Resume parsing with Named Entity Recognition (NER)
- Feedback-driven personalization
- Dynamic weighting via learning-to-rank

---

## 🎓 Developed Under Guidance Of

**Dr. Dip Sankar Banerjee**  
Indian Institute of Technology - Jodhpur

---

## 📃 License
Open-source for academic use.

---

Thank you for exploring our Resume Recommender!

> For questions or contributions, raise an issue or fork the GitHub repo.

