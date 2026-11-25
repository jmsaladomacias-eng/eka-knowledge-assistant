# Minimal RAG Design — MVP

## Objective and scope

The MVP is a knowledge and operations assistant for engineering questions, grounded in reviewed project documentation. Version one is read-only: it retrieves and explains information but does not execute commands, access SSH, call the Proxmox API, write to infrastructure, or run autonomous agents.

## Approved logical architecture

- Original documents and any curated corpus remain in the local environment.
- Extraction, section-aware chunking, embedding generation, and the retrieval index run locally.
- An external LLM is called through an API; no provider or model is selected here.
- Only the user's question and retrieved fragments that pass authorization and security checks may be sent externally. Questions are also checked locally because they can contain sensitive information.
- Full documents, credentials, tokens, signed URLs, personal data, and `RESTRICTED` material are never sent to the external LLM.
- Before integrating any API, review and approve the provider's privacy, retention, and data-handling terms for the intended audience and data.

This is a logical design, not an implementation or approval to process any current source externally.

## Corpus governance

Admission is **deny by default**. The original Obsidian vault is not an indexing source. These documents are candidates for review:

- `docs/architecture.md`
- `docs/storage.md`
- `docs/operations.md`
- `docs/infrastructure/current-state.md`
- `docs/infrastructure/documentation-reconciliation.md`

`README.md` may orient users and retrieval, but is not a primary technical source. Inclusion in this candidate list does not authorize indexing or external processing. Each approval must name the corpus, audience, destination, and explicitly state whether external processing is authorized. Create a sanitized derivative when necessary, retaining source provenance and limitations. Do not automatically change catalog eligibility values; update them only after the governance review and admission gates are completed.

## Minimum chunk metadata

| Field | Purpose |
|---|---|
| `source_document` | Human-readable source title. |
| `source_path` | Relative path to the approved source or derivative. |
| `section` | Heading hierarchy locating the source passage. |
| `chunk_id` | Stable identifier for the indexed chunk. |
| `evidence_status` | Evidence character, such as verified, historical, experimental, hypothesis, or planned, using the governed source status. |
| `operational_validity` | Whether present operational validity is assessed, currently validated, or requires revalidation. Independent of `evidence_status`. |
| `confidentiality` | Governed classification of the approved source/derivative. |
| `source_date` | Source observation or publication date when known; otherwise explicitly unknown. |
| `review_date` | Review date; otherwise `NOT_REVIEWED`. |
| `external_llm_approved` | Explicit allow/deny decision for external processing of this chunk under the named approval scope. Default: `false`. |

Evidence status describes the nature of the recorded evidence; operational validity describes whether the claim was checked against present operations. Neither implies the other.

## Chunking and retrieval

Split documents by sections and headings. Begin with an approximate chunk size of 400–700 tokens; use 60–100 tokens of overlap only where it preserves necessary continuity. Keep section headings, provenance, evidence status, operational validity, and limitations attached to each chunk.

Start with semantic retrieval over the local index. Consider hybrid lexical/vector search only if evaluation shows semantic retrieval misses exact names, error strings, or other terms that matter.

## Query flow

1. Receive the user's question locally.
2. Check it for sensitive content and determine whether it may be sent to an external service; otherwise abstain or answer locally without external processing.
3. Search the local index and retrieve candidate chunks.
4. Check corpus/destination authorization, chunk confidentiality, external-LLM approval, and required sanitization.
5. Send only the approved question and approved retrieved fragments to the external LLM API.
6. Return an answer with source citations, evidence status, operational-validity status, and uncertainty.
7. Abstain when the retrieved evidence is insufficient, contradictory without resolution, or not approved for the requested processing.

No full document or unapproved fragment crosses the local boundary.

## Evaluation plan

Prepare an initial set of approximately 25 questions before implementation acceptance. Include questions about historical facts, hypotheses, contradictions, questions with no supported answer, and security-boundary cases. Measure retrieval relevance, answer grounding, citation accuracy, and appropriate abstention. Record the test set and results; do not claim metrics or successful results before running evaluation.

## MVP acceptance criteria

- Every retrieved claim is traceable to an approved source and section.
- Answers are grounded in retrieved evidence and cite it.
- Responses distinguish documented facts, historical evidence, hypotheses, and operational validity.
- The assistant abstains when evidence is missing or inadequate.
- No fragment lacking explicit external-processing approval reaches the external LLM.
- The MVP performs no infrastructure operations or command execution.

## Out of scope

Autonomous agents, Telegram, real-time monitoring, infrastructure writes, operational automation, and multi-agent architectures.

## Decisions pending

- LLM API provider.
- Embedding model.
- Vector database or local index technology.
- RAG framework.
- User interface.
