# Technologies And Rationale

This document explains which technologies are used in the German Administrative Assistant and why they were chosen.

## Python

Python is used for the backend, scraping, graph ingestion, agent workflow, and evaluation.

Why:

- Strong ecosystem for LLM applications.
- Good support for Neo4j, LangChain, LangGraph, MCP, scraping, and data processing.
- Fast iteration during a thesis/prototype project.

## LangGraph

LangGraph coordinates the multi-agent workflow.

Used for:

- Intake routing
- Retrieval planning
- Knowledge search
- Answer drafting
- Supervision
- Revision
- Final translation/response

Why:

- The assistant is not a single prompt. It has different responsibilities that need stateful routing.
- LangGraph makes the flow explicit and testable.
- Supervisor loops can be limited, for example max two extra search rounds.

## Neo4j

Neo4j stores the Service-BW knowledge graph.

Used nodes include:

- `Situation`
- `SubSituation`
- `Service`
- `ServiceSection`
- `Requirement`
- `Document`
- `Authority`
- `Form`
- `ProcessStep`
- `LegalBasis`
- `Goal`
- `DependencyProblem`
- `ServiceQA`
- `Chunk`

Why:

- Administrative services are naturally relational.
- A graph can represent which services belong to which situations and which requirements/documents/authorities belong to a service.
- It supports GraphRAG: vector search finds a relevant node, and graph traversal loads the connected service context.

## Neo4j Vector Search

Service-BW content is chunked and embedded into `Chunk` nodes.

Why:

- Users do not always know official terms.
- Vector search can match user intent to semantically similar situations, services, and Q&A facts.
- Graph expansion then gives structured details instead of only returning text chunks.

## Ollama

Ollama is used for local chat models and embeddings.

Main embedding model:

```text
mxbai-embed-large
```

Why:

- Keeps embeddings local.
- Avoids external embedding API cost.
- Makes the project easier to run in a controlled academic environment.

## Groq

Groq can be used for higher-quality or faster cloud LLM calls.

Used especially for:

- supervisor agent
- revision agent
- evaluation judge

Why:

- Some local models return invalid JSON or weaker reviews.
- A stronger judge/supervisor improves quality control and evaluation stability.

## MCP

The project exposes tools through an MCP server:

- `search_problem_knowledge`
- `service_details`
- `web_search`
- `scrape`

Why:

- The agent can call tools through a clean boundary.
- Tool logs are separated from normal answer generation.
- The same knowledge search can be reused by different agents.

## Service-BW REST APIs

Service-BW pages are not treated as normal static HTML pages. The project uses REST endpoints:

```text
/rest/api/lebenslagen/gruppen
/rest/api/lebenslagen/{id}
/rest/api/leistungen/{id}
```

Why:

- HTML pages are not reliable for structured extraction.
- The REST API gives cleaner data for life situations, sub-situations, linked services, text blocks, forms, authorities, and process information.
- API-based extraction is more repeatable than browser scraping.

## BeautifulSoup And Requests

Used for:

- generic web page scraping
- cleaning HTML text from API text blocks

Why:

- Service-BW API fields can still contain HTML.
- Official fallback web pages may need normal HTML extraction.

## Streamlit

Streamlit provides the frontend.

Why:

- Fast to build a usable chat interface.
- Useful for thesis/demo purposes.
- Allows a right-side case panel showing known facts, documents, links, and debug state.

## SQLite / PostgreSQL / Qdrant

Conversation memory is separate from LangGraph checkpoint state.

Used options:

- SQLite for simple local conversation memory.
- PostgreSQL for persistent conversation memory and LangGraph checkpoints.
- Qdrant exists as an optional semantic memory implementation, but it is not the default path.

Why:

- Conversation memory should store clean user/assistant turns.
- Graph checkpoints should store workflow state.
- Keeping them separate avoids mixing chat history with internal agent state.

## Evaluation Runner

The evaluation flow is in `evaluation_flow/`.

Why:

- It keeps experiments separate from production code.
- It runs a fixed question set through the same assistant.
- A second LLM judge gives structured scores and reasons.
- Results are saved as JSONL, CSV, and summary JSON for analysis.
