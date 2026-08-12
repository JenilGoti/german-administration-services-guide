# Setup Guide

This guide explains how to set up the German Administrative Assistant locally, build the Neo4j knowledge base, run the frontend/CLI, and execute the evaluation flow.

## 1. Requirements

Install these first:

- Python 3.11 or newer
- Neo4j with vector index support
- Ollama
- Git

Optional:

- Groq API key for stronger supervisor, revision, or judge models
- PostgreSQL if you want persistent LangGraph checkpoints or Postgres conversation memory

## 2. Create Python Environment

```bash
cd WRAITH
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Pull Ollama Models

The project keeps embeddings on Ollama.

```bash
ollama pull mxbai-embed-large
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5:7b-instruct
ollama pull aya:8b
ollama pull llama3.1:latest
```

You can use different models, but changing the embedding model after building the graph requires rebuilding embeddings.

## 4. Configure Environment

Create a `.env` file in the project root:

```env
GDB_URL=bolt://localhost:7687
GDB_USER=neo4j
GDB_PASSWORD=your-password
KNOWLEDGE_DB=dev-graph

LLM_PROVIDER=ollama
QUALITY_LLM_PROVIDER=groq
GROQ_API_KEY=

OLLAMA_DEFAULT_MODEL=qwen2.5-coder:7b
OLLAMA_TRANSLATION_MODEL=aya:8b
OLLAMA_REASONING_MODEL=qwen2.5:7b-instruct
OLLAMA_STRUCTURED_MODEL=llama3.1:latest
OLLAMA_SUPERVISOR_MODEL=llama3.1:latest

GROQ_DEFAULT_MODEL=llama-3.3-70b-versatile
GROQ_TRANSLATION_MODEL=llama-3.3-70b-versatile
GROQ_REASONING_MODEL=llama-3.3-70b-versatile
GROQ_STRUCTURED_MODEL=llama-3.3-70b-versatile
GROQ_SUPERVISOR_MODEL=llama-3.3-70b-versatile

OLLAMA_EMBEDDING_MODEL=mxbai-embed-large

CONVERSATION_MEMORY_BACKEND=auto
CONVERSATION_POSTGRES_URL=
LANGGRAPH_POSTGRES_URL=
LANGGRAPH_POSTGRES_SETUP=true

MCP_LOG_MAX_CHARS=12000
```

## 5. Prepare Neo4j

Start Neo4j and create/select the database named in `KNOWLEDGE_DB`, for example `dev-graph`.

The project creates constraints and vector indexes from code. If the vector index configuration changes, the index may be recreated automatically.

## 6. Build The Knowledge Base

### Step 1: Crawl Service-BW Situation/Service Listing

```bash
python3 -m scrapping.all_links_scrapping
```

This creates:

```text
scrapping/service_bw_output.json
```

The file contains:

- top-level life situations
- sub-situations from `lebenslagenbaum`
- linked `leistungen` service IDs
- Service-BW URLs for later enrichment

### Step 2: Insert Situation Mapping

```bash
python3 -m scrapping.flush_all_situations_mapping
```

This creates the first graph structure:

```text
Situation -> SubSituation -> Service
```

This command also refreshes the Service-BW listing before inserting the basic
mapping, so it can be used as a combined crawl-and-import step.

### Step 3: Enrich Sub-Situations

```bash
python3 -m scrapping.flush_sub_situations_from_listing
```

This reads each sub-situation page through the Service-BW API, summarizes it, stores it in Neo4j, chunks it, and embeds it.

### Step 4: Enrich Services

```bash
python3 -m scrapping.flush_services_from_listing
```

This reads each service through:

```text
https://www.service-bw.de/rest/api/leistungen/{service_id}
```

It extracts sections, requirements, documents, forms, authorities, process steps, legal basis, goals, dependency problems, summaries, and ServiceQA facts. These are stored in Neo4j and embedded through the existing chunk pipeline.

### Step 5: Backfill ServiceQA For Existing Services

If services already exist in Neo4j and you only need the new ServiceQA layer:

```bash
python3 scrapping/backfill_service_qa.py --limit 10
python3 scrapping/backfill_service_qa.py
```

Useful options:

```bash
python3 scrapping/backfill_service_qa.py --progress-every 20
python3 scrapping/backfill_service_qa.py --overwrite
```

## 7. Run The Application

### Streamlit Frontend

```bash
streamlit run frontend/streamlit_app.py
```

### CLI

```bash
python3 app.py
```

## 8. Run Evaluation

Small test:

```bash
python3 evaluation_flow/run_evaluation.py --limit 10 --out-dir evaluation_flow/results/test_10
```

Full evaluation:

```bash
python3 evaluation_flow/run_evaluation.py
```

Results are written to:

```text
evaluation_flow/results/<timestamp>/
```

## 9. Common Problems

### Neo4j Package Missing

If Python says `No module named neo4j`, activate the virtual environment and install requirements again:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Ollama Embeddings Fail

Check that Ollama is running and the embedding model exists:

```bash
ollama list
ollama pull mxbai-embed-large
```

### Vector Index Recreated

This can happen when the configured embedding dimension changes. It is expected after changing embedding models, but existing chunks should be rebuilt for consistent retrieval.

### Evaluation Scores Look Bad

Check three things first:

- Did questions route to `admin`?
- Are `SubSituation`, `Service`, `ServiceQA`, and `service_details` counts non-zero?
- Did the judge return valid JSON?

If retrieval is zero, the problem is likely intake or graph setup. If retrieval is noisy, the problem is retrieval ranking or missing graph coverage.
