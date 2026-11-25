# Knowledge Governance

This policy governs technical knowledge documented in the Resilient Lab Platform and any later curated corpus. It applies to source notes, repository documentation, runbooks, incident records, decision records, and material considered for retrieval-augmented generation (RAG). It does not authorize copying the private Obsidian archive, publishing material, or ingesting it into a RAG system.

## 1. Document lifecycle

Use the following lifecycle for each source or curated derivative:

1. **Discovered:** Record its original relative path and basic topic/type in `docs/knowledge-catalog.md`. Do not move or rename the source as part of cataloging.
2. **Triaged:** Assign exactly one `status`, `operational_validity`, `confidentiality`, `rag_eligibility`, and `knowledge_domain` to each catalog entry. Record missing context, review metadata, duplicate relationships, and known sensitive content without copying any secret or identifier into the catalog. Use the explicit unknown values defined below; do not infer owners, reviews, current operation, or canonical versions.
3. **Curated:** If reuse is justified, create a separate derivative in the intended destination. Preserve attribution/provenance, scope, capture date, commands' prerequisites, and uncertainty. Redact secrets and unnecessary infrastructure identifiers. Never edit the source archive as a substitute for making a derivative.
4. **Reviewed:** A knowledgeable reviewer checks factual claims, security classification, links, and any commands with operational impact. A document is not considered approved merely because it is in Git.
5. **Approved for use:** Record review date, reviewer/owner role, scope, and eligibility decision. Public release and external RAG each require explicit review for that destination; approval for one does not imply approval for the other.
6. **Superseded or retired:** Link to the replacing record and retain history where appropriate. Mark the old record HISTORICAL; do not silently erase evidence or treat a newer filename as proof of correctness.

Empty, fragmentary, duplicate, or unreviewed records remain discoverable in the catalog but are not authority for operational answers. Any local source archive remains governed by its owner; this policy does not modify it.

## 2. Provenance requirements

For every source and derivative, capture what is available from this minimum set:

- `source_path`: original path relative to the source archive or repository; preserve the original filename exactly.
- `document_type`: for example runbook, incident, project log, architecture/decision, tutorial, backlog, attachment, or prototype.
- `topic` and `system/component`.
- Exactly one each of `status`, `operational_validity`, `confidentiality`, `rag_eligibility`, and primary `knowledge_domain`, using the controlled values below.
- `observed_at` or source date, if the source provides one; do not infer dates from file modification time.
- `owner`, `reviewed_at`, `reviewer_role`, and `review_scope`. Use `NOT_ASSIGNED` when no owner/reviewer role is assigned and `NOT_REVIEWED` when no review date or scope exists. Do not invent values.
- Optional document relationships: `canonical_candidate`, `duplicate_of`, `related_documents`, `supersedes`, and `derived_from`. Record only relationships supported by evidence; a relationship pointer does not itself select an authoritative version.
- `limitations`: unverified assumptions, omitted context, unknown root cause, and point-in-time scope.

For operational observations, retain the evidence and exact scope needed to reproduce the claim. Distinguish “command output showed X at capture time” from “X is the current state.” A summary derived from multiple notes should cite all of them and identify conflicting evidence rather than silently selecting one.


## 3. Operational validity

Operational validity is independent of document verification `status`. `status` describes the evidence and purpose represented by the record; `operational_validity` describes whether the documented configuration or procedure has been checked against the present system. A historically verified test can therefore have `REVALIDATION_REQUIRED`, and a planned or experimental record can remain `NOT_ASSESSED`. Do not infer present validity from a document's date, title, successful past test, or status.

Every catalog entry has exactly one value:

| Value | Meaning and use |
|---|---|
| `NOT_ASSESSED` | No authorized present-state check is recorded. This is the default when evidence is absent. |
| `CURRENTLY_VALIDATED` | The specified operational claim was checked against a named present system/version and scope on a recorded date. It is bounded to that scope and date, not a continuing guarantee. |
| `REVALIDATION_REQUIRED` | The document may describe an earlier state or procedure, but must be checked before relying on it for current operations. |

## 4. Knowledge domains

Every catalog entry has exactly one primary `knowledge_domain`. Choose the domain that best represents the entry's primary use; secondary topics belong in its topic/limitations or relationships, not as extra primary domains. Use only these controlled values:

- `Infrastructure`
- `Storage`
- `Networking`
- `Backup & Recovery`
- `Monitoring`
- `AI & LLM`
- `Troubleshooting`
- `Cloud & OCI`
- `Documentation & Project Management`

## 5. Document relationships

Relationship metadata is optional and should be populated only when supported by the source or exact-file comparison:

- `canonical_candidate`: a candidate for owner review, not an approved canonical record. Use `UNDECIDED` where duplicate material exists and no candidate has been selected.
- `duplicate_of`: relative path(s) of byte-identical duplicate file(s). A pointer is for traceability only and does not make the target canonical. For multi-file duplicate groups, choosing a stable reference path for the graph is not a canonical decision.
- `related_documents`: relative path(s) with a documented or directly evidenced relationship; topic similarity alone is insufficient.
- `supersedes`: relative path(s) the source explicitly replaces, with evidence of replacement. A suffix, newer timestamp, or “final” title alone does not establish supersession.
- `derived_from`: source path(s) from which this document was directly derived, where documented.

Do not infer a canonical candidate or supersession from chronology or filenames. Leave unknown or unsupported relationships unset (`—` in the catalog).

## 6. Verification status

Each catalog entry has exactly one primary status. These values describe the evidence represented by the document, not the runtime availability of a system:

| Status | Meaning and use |
|---|---|
| `VERIFIED` | The specific claim is supported by recorded evidence or a documented test. State the scope and date. It does not mean continuously current or production-ready. |
| `HISTORICAL` | A record of past state, event, or configuration. Useful for history and incident response, but not authority for present configuration without revalidation. |
| `EXPERIMENTAL` | A trial, prototype, learning exercise, or unsuccessful/incomplete configuration attempt. Do not present as an available service or established procedure. |
| `HYPOTHESIS` | An explanation, causal claim, or recommendation not yet supported by sufficient evidence. Keep it labeled as such. |
| `PLANNED` | Proposed or intended work not yet evidenced as completed. |

Documents may contain claims with different evidence strengths. Choose the status that best describes their dominant use and call out verified subsections or weaker claims in `limitations`. A status is not a quality score and does not determine `operational_validity`. Revalidation must create a dated review/update; never change a historical record to imply it was always current.

## 7. Confidentiality classification

Assign one level based on the most sensitive material in the source. When uncertain, choose the more restrictive level until an owner reviews it.

| Level | Handling |
|---|---|
| `PUBLIC` | Explicitly reviewed and approved for public release. Contains no secrets, personal/customer data, or unnecessary private infrastructure detail. |
| `INTERNAL` | Operational or organizational detail intended for the owner's private environment. Do not publish or send to external RAG without a reviewed, sanitized derivative. |
| `CONFIDENTIAL` | Personal data, customer/project information, account identifiers, or commercially sensitive assumptions. Restrict access to authorized recipients; do not publish or externally ingest without owner authorization and redaction. |
| `RESTRICTED` | Credentials, tokens, signed/bearer URLs, private keys, recovery codes, or material whose exposure could enable access. Keep out of Git and RAG. Assess validity through an authorized channel and revoke/rotate if active or still valid; sanitized derivatives require security review. |

Classify attachments and links as well as visible prose. A document with one restricted value is `RESTRICTED` even if most of its content is ordinary technical guidance. Never copy secret values into this catalog, commit messages, examples, or audit excerpts. Finding a credential-like value in historical documentation does not prove that the service is active or that the value is currently valid; assess validity through an authorized channel and rotate/revoke if it is active or still valid. For signed URLs, URL expiry and the underlying temporary credential lifetime are distinct: URL expiration does not by itself prove that the associated temporary credentials have expired, nor does an unexpired credential make an expired URL usable. Do not reproduce either value. Private IPs and hostnames are usually at least `INTERNAL`; they are not credentials, but can disclose topology.

## 8. RAG eligibility

Eligibility describes whether a reviewed derivative may be considered for a specific RAG corpus. It is not a command to ingest.

| Value | Meaning |
|---|---|
| `ELIGIBLE` | Reviewed for the named corpus and audience, with provenance, status, and classification recorded. Safe for ingestion into that approved corpus under its access controls. |
| `ELIGIBLE_AFTER_REVIEW` | Potentially useful, but must be curated, redacted, fact-checked, and approved for the target audience/corpus first. |
| `EXCLUDED` | Do not ingest this source into the proposed corpus. This includes empty/duplicate stubs, restricted content as-is, personal/customer material without authorization, and content with unresolved rights or security concerns. A separate sanitized derivative can receive a new decision. |

### RAG admission gates

Before changing an item to `ELIGIBLE`, the reviewer must confirm:

1. The destination and audience are named (internal, public, or another explicit boundary).
2. Confidentiality is compatible with that destination and access controls.
3. Secrets, signed URLs, personal data, and unnecessary private identifiers are absent from the derivative, metadata, and linked attachments.
4. Provenance, status, scope/date, and limitations travel with each chunk or remain reliably retrievable.
5. Duplicate or superseded records are identified so stale and current claims are not blended.
6. Commands and operational steps have been reviewed for prerequisites and risk. RAG eligibility alone never grants an agent permission to execute them.

Default ingestion is **deny by default**. The raw source tree is not an ingestion target. No source in the initial catalog is marked `ELIGIBLE` because this governance/corpus review has not yet occurred.

## 9. Review, freshness, and supersession

- Review on material change, after a relevant incident, before public release, and before each new corpus/audience approval. Do not expire a document automatically merely because it is old.
- For operational instructions, include the product/version and last evidence date when known. Revalidate before relying on historical steps against a changed system.
- Preserve incident chronology and unresolved root causes. A recovery that restored service does not by itself prove root cause or permanent resolution.
- When two documents conflict, record both paths, compare timestamps and evidence, and ask the owner or inspect the system when authorized. Do not resolve conflicts by filename suffix, “final” in a title, or recency alone.
- A superseding document should identify the source(s) it replaces, what changed, and the date/scope of validation. Retain older records as HISTORICAL when useful for audit or incident learning.
- Exact duplicates should be linked with `duplicate_of` in the catalog. Do not set a canonical candidate unless evidence supports a candidate for review, and do not delete or rename source documents under this policy.
- Reassess confidentiality and RAG eligibility whenever a source is edited, attached files change, or the destination changes.

## 10. OpenClaw handling

The two OpenClaw notes are classified `EXPERIMENTAL` because they record an unsuccessful installation/configuration attempt. No operational OpenClaw deployment exists in the evidence reviewed; the notes do not establish a currently available service. A token found in historical documentation is a sensitive-content finding, but does not establish that a corresponding service is active or that the token remains valid. Preserve `RESTRICTED` and `EXCLUDED` for these raw sources because sensitive access material and private access context are present. Verify validity only through an authorized channel; any future sanitized learning summary requires security review.

## 11. Initial implementation boundary

`docs/knowledge-catalog.md` is the initial index of the 58 audited raw source files. It records relative original paths and conservative first-pass decisions. Catalog entries are not endorsements of correctness, authorization to publish, or proof of live state. The OCI Backup Calculator remains independently reviewable; its placement in the catalog does not imply integration into this repository or the homelab RAG corpus.
