# Enterprise Data & ML Knowledge Assistant
## GenAI / Retrieval-Augmented Generation Proof of Concept
**Submission date:** 30 October 2026  
**Project status:** Working local functional POC; evaluation completed on a small synthetic test set.

### Objective
Build a practical enterprise-oriented GenAI assistant that answers questions about ML feature QA, Hexagon scoring, and troubleshooting using an indexed, synthetic knowledge base. Demonstrate embeddings, vector search, grounded generation, citations, and initial evaluation.

### Architecture
User question -> Sentence Transformers (`all-MiniLM-L6-v2`) embedding -> ChromaDB similarity search (top 3) -> retrieved context and grounding instructions -> Ollama local API (`llama3.2:3b`) -> answer with source chunk IDs -> Streamlit interface and performance metrics.

### Implemented functionality
- Python 3.11 virtual environment and local-only model inference.
- Synthetic knowledge ingestion and persistent ChromaDB vector storage.
- Semantic document retrieval with chunk metadata and distances.
- Prompt-grounded LLM generation with abstention instructions and source references.
- Streamlit question-and-answer interface with retrieved evidence, latency, and token counts.
- CSV-based evaluation of top-1 retrieval match, answer keyword coverage, and latency.

### Observed evaluation (3 test questions)
| Metric | Observed result |
|---|---|
| Top-1 retrieval match | 3/3 (100%) |
| Keyword-check pass rate | 3/3 (100%) |
| Average end-to-end latency | 12.75 seconds |
| Contact BP question latency | 8.573 seconds |
| Hexagon scoring question latency | 12.545 seconds |
| QA troubleshooting question latency | 17.144 seconds |
| Unsupported revenue question | Correctly abstained (1/1) |
| Unsupported question latency | 10.578 seconds |

**Evaluation caveat:** These are functional smoke tests using three known questions and one unsupported question, not production accuracy, comprehensive hallucination measurement, or statistical performance estimates. Keyword coverage is not a factual correctness metric. No automated agentic workflow has been demonstrated.

### Sample supported use cases
1. Contact BP QA: non-null identifiers, feature/target checks, boolean checks, ratios in [0, 1].
2. Hexagon scoring: explain activity probabilities, time-dependent weights, and example formula.
3. QA troubleshooting: investigate organization/product, source data, SQL, nulls, schema changes, and thresholds.

### Security and data handling
Uses synthetic demonstration text. Do not upload proprietary client or employer documentation into this personal-laptop POC without explicit authorization. Local inference avoids sending prompts to a hosted LLM, although model installation can access public model repositories.

### Current limitations / future work
- Only four knowledge chunks; extend with permissioned, versioned documentation.
- No calibrated relevance threshold or formal citation faithfulness check.
- Evaluate on a larger, labeled set including unsupported, ambiguous, and adversarial questions.
- Optimize response latency; benchmark warm and cold starts separately.
- Add an explicitly scoped, read-only agentic tool workflow in a future iteration.
- Add reproducible unit tests, dependency pinning, and deployment controls.

### Reproduction (Windows PowerShell, from project folder)
```powershell
.\.venv\Scripts\Activate.ps1
python ingest.py
python evaluate.py
python -m streamlit run app.py
```
Ollama must be installed and `llama3.2:3b` available locally. Application URL: `http://localhost:8501`.

### Submission checklist
- Attach this project report.
- Zip and attach the **actual local project source** (not this report alone); exclude `.venv/`, `chroma_db/`, `__pycache__/`, and sensitive files.
- Attach screenshot of working Streamlit response and `results/evaluation_results.csv`.
- Note that the agentic AI component is planned, not delivered.
