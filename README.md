# Engineering Knowledge & Operations Assistant (EKA) v1.0

EKA is a local, read-only knowledge and operations assistant MVP designed for governed technical documentation. It enforces fail-closed admission gates, section-aware Markdown chunking, deterministic lexical TF-IDF retrieval, and an Answer Engine that returns structured citations and provenance—or abstains when evidence is missing, unapproved, or contradictory.

## Key Capabilities

- **Governed Ingestion & Persistence:** Allowlist-based Markdown extraction (`app/ingestion.py`) and local JSONL chunk storage (`app/corpus.py`) gated by formal catalog admission and reviewed SHA-256 digests.
- **Section-Aware Chunking:** Heading-preserving Markdown splitter (`app/chunking.py`) that attaches provenance, evidence status, operational validity, and confidentiality metadata to each chunk.
- **Deterministic Lexical Retrieval:** Local TF-IDF search (`app/retrieval.py`) over approved JSONL corpus files.
- **Fail-Closed Answer Engine:** Enforces both question-level and chunk-level authorization before invoking any LLM provider (`app/answer_engine.py`), validates JSON output schema and citations, and abstains when evidence is insufficient.
- **Local Loopback Web UI:** Minimal `127.0.0.1` HTTP interface (`app/web.py`) defaulting to `MockLLMProvider` (`app/llm/mock.py`). External LLM calls (`app/llm/anthropic.py`) are disabled by default and require explicit configuration (`EKA_LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`) plus per-request and per-chunk authorization.

## Running the Reproducible Synthetic Demo

A self-contained, network-free demo (`demo.py`) creates an isolated temporary workspace with fictitious synthetic documents, formal catalog admission (`external_llm_approved=false`), and `MockLLMProvider` to demonstrate:
1. An answerable question with structured citations.
2. A question without supporting evidence resulting in abstention.
3. A non-admitted (`PENDING`) document being blocked by the governance gate.

```bash
python -m demo
```

## Running Tests

The test suite uses isolated temporary directories, synthetic fixtures, and `MockLLMProvider` (no real corpus or external API calls required):

```bash
python -m unittest discover -s tests
```

## Running the Local Web Interface

```bash
python -m app.web --port 8000
```

> **Note:** Real corpus files (`data/corpus/*.jsonl`) and internal infrastructure catalogs are intentionally excluded from this public repository. In default `mock` mode without a local corpus, the web interface starts cleanly and abstains safely (`no_externally_authorized_evidence`).

## Documentation

- [Architecture](docs/architecture.md) — documented system roles and data flow.
- [RAG Design](docs/rag-design.md) — MVP architecture, chunk metadata, query flow, and acceptance criteria.
- [Knowledge Governance](docs/knowledge-governance.md) — document lifecycle, confidentiality, and RAG admission rules.
- [Architecture Diagram](diagrams/architecture.mmd) — logical design diagram.
