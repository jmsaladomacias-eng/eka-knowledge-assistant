"""Local JSONL persistence for explicitly approved internal-corpus chunks."""

from dataclasses import asdict
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

from app.chunking import MarkdownChunk, RAG_ELIGIBILITY_VALUES
from app.ingestion import ALLOWED_SOURCE_PATHS, ALLOWED_CONFIDENTIALITY, INTERNAL_CORPUS, SourceApproval


class CorpusRejected(ValueError):
    """Raised when a source or chunk is not approved for local persistence."""


SOURCE_FILES = {
    "docs/architecture.md": "architecture.jsonl",
    "docs/storage.md": "storage.jsonl",
    "docs/operations.md": "operations.jsonl",
    "docs/infrastructure/current-state.md": "current-state.jsonl",
    "docs/infrastructure/documentation-reconciliation.md": "documentation-reconciliation.jsonl",
}

RAG_TARGET_CORPUS = "EKA v1.0 internal local"
RAG_AUTHORIZED_AUDIENCE = "Authorized internal local EKA users"
ADMISSION_SECTION = "## Repository RAG admission decisions"
ADMISSION_FIELDS = (
    "source_path",
    "admission_status",
    "rag_eligibility",
    "confidentiality",
    "evidence_status",
    "operational_validity",
    "owner",
    "reviewed_at",
    "reviewer_role",
    "review_scope",
    "target_corpus",
    "authorized_audience",
    "reviewed_sha256",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _catalog_rows(repository_root: str | Path) -> dict[str, dict[str, str]]:
    """Read the machine-readable admission table embedded in the governance catalog."""
    root = Path(repository_root).resolve(strict=True)
    catalog = root / "docs" / "knowledge-catalog.md"
    if catalog.is_symlink() or catalog.parent.is_symlink():
        raise CorpusRejected("Symlinked knowledge catalog paths are not permitted.")
    try:
        catalog.resolve(strict=True).relative_to(root)
        text = catalog.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise CorpusRejected("Knowledge catalog is missing or outside the repository.") from exc

    lines = text.splitlines()
    try:
        section_index = lines.index(ADMISSION_SECTION)
    except ValueError as exc:
        raise CorpusRejected("Formal RAG admission section is missing from the catalog.") from exc

    table_lines = []
    for line in lines[section_index + 1 :]:
        if line.startswith("## "):
            break
        if line.strip().startswith("|"):
            table_lines.append(line.strip())
    if len(table_lines) < 2:
        raise CorpusRejected("Formal RAG admission table is missing or incomplete.")

    def cells(line: str) -> list[str]:
        return [cell.strip().strip("`").strip() for cell in line.strip("|").split("|")]

    if tuple(cells(table_lines[0])) != ADMISSION_FIELDS:
        raise CorpusRejected("Formal RAG admission table has an unsupported schema.")
    separator = cells(table_lines[1])
    if len(separator) != len(ADMISSION_FIELDS) or not all(
        re.fullmatch(r":?-{3,}:?", value) for value in separator
    ):
        raise CorpusRejected("Formal RAG admission table separator is invalid.")

    entries: dict[str, dict[str, str]] = {}
    for line in table_lines[1:]:
        values = cells(line)
        if values and all(re.fullmatch(r":?-{3,}:?", value) for value in values):
            continue
        if len(values) != len(ADMISSION_FIELDS):
            raise CorpusRejected("Formal RAG admission row has an invalid field count.")
        row = dict(zip(ADMISSION_FIELDS, values))
        source_path = row["source_path"]
        if source_path in entries:
            raise CorpusRejected(f"Duplicate RAG admission record for {source_path}.")
        entries[source_path] = row
    return entries


def _source_digest(repository_root: str | Path, source_path: str) -> str:
    root = Path(repository_root).resolve(strict=True)
    relative = Path(source_path)
    if relative.is_absolute() or relative.as_posix() not in SOURCE_FILES:
        raise CorpusRejected("Source path is not an approved repository document.")
    candidate = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise CorpusRejected("Symlinks are not permitted in admitted source paths.")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise CorpusRejected("Source document is missing or outside the repository.") from exc
    if not resolved.is_file():
        raise CorpusRejected("Admitted source must be a regular file.")
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def validate_document_admission(
    repository_root: str | Path,
    source_path: str | Path,
    document_sha256: str | None,
    *,
    require_approved: bool,
) -> dict[str, str]:
    """Validate catalog scope and optionally formal admission and source digest."""
    if not isinstance(source_path, (str, Path)):
        raise CorpusRejected("Source path is invalid for formal RAG admission.")
    normalized = Path(source_path).as_posix()
    if normalized not in SOURCE_FILES:
        raise CorpusRejected(f"Source has no formal RAG candidate mapping: {normalized}")
    row = _catalog_rows(repository_root).get(normalized)
    if row is None:
        raise CorpusRejected(f"No formal RAG admission record for {normalized}.")
    if row["target_corpus"] != RAG_TARGET_CORPUS:
        raise CorpusRejected("RAG admission targets a different corpus.")
    if row["authorized_audience"] != RAG_AUTHORIZED_AUDIENCE:
        raise CorpusRejected("RAG admission targets a different audience.")
    if row["admission_status"] not in {"PENDING", "APPROVED", "REVOKED"}:
        raise CorpusRejected("RAG admission has an unsupported admission status.")
    if row["confidentiality"] not in ALLOWED_CONFIDENTIALITY:
        raise CorpusRejected("RAG admission confidentiality is incompatible.")
    if row["rag_eligibility"] not in RAG_ELIGIBILITY_VALUES:
        raise CorpusRejected("RAG admission has an unsupported eligibility value.")

    current_digest = _source_digest(repository_root, normalized)
    if document_sha256 is not None and document_sha256 != current_digest:
        raise CorpusRejected("Chunk source digest does not match the current document.")
    if not require_approved:
        return row

    if row["admission_status"] != "APPROVED" or row["rag_eligibility"] != "ELIGIBLE":
        raise CorpusRejected("Document has no formal ELIGIBLE RAG admission.")
    if row["owner"] in ("", "NOT_ASSIGNED"):
        raise CorpusRejected("Formal RAG admission owner is missing.")
    if row["reviewer_role"] in ("", "NOT_ASSIGNED", "NOT_REVIEWED"):
        raise CorpusRejected("Formal RAG admission reviewer role is missing.")
    if row["review_scope"] in ("", "NOT_REVIEWED"):
        raise CorpusRejected("Formal RAG admission review scope is missing.")
    if row["evidence_status"] not in {"VERIFIED", "HISTORICAL", "EXPERIMENTAL", "HYPOTHESIS", "PLANNED"}:
        raise CorpusRejected("Formal RAG admission evidence status is invalid.")
    if row["operational_validity"] not in {
        "NOT_ASSESSED", "CURRENTLY_VALIDATED", "REVALIDATION_REQUIRED"
    }:
        raise CorpusRejected("Formal RAG admission operational validity is invalid.")
    try:
        date.fromisoformat(row["reviewed_at"])
    except ValueError as exc:
        raise CorpusRejected("Formal RAG admission review date is invalid.") from exc
    approved_digest = row["reviewed_sha256"]
    if not _SHA256.fullmatch(approved_digest):
        raise CorpusRejected("Formal RAG admission has no valid reviewed SHA-256.")
    if approved_digest != current_digest or document_sha256 != approved_digest:
        raise CorpusRejected("Document or chunk content changed since formal review.")
    return row


def _validate_source_and_approval(source_path: str | Path, approval: SourceApproval | None) -> str:
    normalized = Path(source_path).as_posix()
    if normalized not in ALLOWED_SOURCE_PATHS or normalized not in SOURCE_FILES:
        raise CorpusRejected(f"Source is not authorized for corpus persistence: {normalized}")
    if approval is None or approval.corpus != INTERNAL_CORPUS:
        raise CorpusRejected("Explicit approval for the internal corpus is required.")
    if approval.confidentiality not in ALLOWED_CONFIDENTIALITY:
        raise CorpusRejected("Confidentiality is incompatible with the internal corpus.")
    if not approval.security_review_complete:
        raise CorpusRejected("Security review is pending; corpus write rejected.")
    return normalized


def _document_path(repository_root: str | Path, source_path: str) -> Path:
    root = Path(repository_root).resolve(strict=True)
    data_dir = root / "data"
    corpus_dir = data_dir / "corpus"
    target = corpus_dir / SOURCE_FILES[source_path]

    # Check each existing path component before any read, write, mkdir, or unlink.
    for directory in (data_dir, corpus_dir):
        if directory.is_symlink():
            raise CorpusRejected("Symlinked corpus directories are not permitted.")

    if target.is_symlink():
        raise CorpusRejected("Symlinked JSONL files are not permitted.")

    resolved_data = data_dir.resolve()
    resolved_corpus = corpus_dir.resolve()
    resolved_target = target.resolve()
    try:
        resolved_data.relative_to(root)
        resolved_corpus.relative_to(root)
        resolved_target.relative_to(resolved_corpus)
    except ValueError as exc:
        raise CorpusRejected("Resolved corpus path is outside the repository corpus directory.") from exc

    if resolved_corpus != resolved_data / "corpus":
        raise CorpusRejected("Resolved corpus directory is outside the repository data directory.")
    if target.parent != corpus_dir or target.suffix != ".jsonl":
        raise CorpusRejected("Resolved corpus path is outside the approved corpus directory.")
    return target


def _validate_chunks(chunks: Iterable[MarkdownChunk], source_path: str, approval: SourceApproval) -> list[MarkdownChunk]:
    items = list(chunks)
    for chunk in items:
        if chunk.source_path != source_path:
            raise CorpusRejected("Chunk source_path does not match the approved source.")
        if chunk.confidentiality != approval.confidentiality:
            raise CorpusRejected("Chunk confidentiality does not match its approval.")
        if chunk.external_llm_approved is not False:
            raise CorpusRejected("Chunks approved for external LLM processing cannot be persisted here.")
        if chunk.rag_eligibility is not None and (
            not isinstance(chunk.rag_eligibility, str)
            or chunk.rag_eligibility not in RAG_ELIGIBILITY_VALUES
        ):
            raise CorpusRejected("Unsupported rag_eligibility value.")
    return items


def read_document(repository_root: str | Path, source_path: str | Path) -> list[dict]:
    """Read one source's JSONL records; a missing corpus/document returns []."""
    normalized = Path(source_path).as_posix()
    if normalized not in ALLOWED_SOURCE_PATHS or normalized not in SOURCE_FILES:
        raise CorpusRejected(f"Source is not authorized for corpus persistence: {normalized}")
    path = _document_path(repository_root, normalized)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise CorpusRejected(f"Invalid JSONL at line {line_number} in {path.name}.") from exc
    return records


def write_chunks(
    repository_root: str | Path,
    source_path: str | Path,
    chunks: Iterable[MarkdownChunk],
    approval: SourceApproval | None = None,
) -> Path:
    """Upsert chunks to the predetermined JSONL file for one approved source."""
    normalized = _validate_source_and_approval(source_path, approval)
    items = _validate_chunks(chunks, normalized, approval)
    admission = validate_document_admission(
        repository_root,
        normalized,
        None,
        require_approved=False,
    )
    if admission["confidentiality"] != approval.confidentiality:
        raise CorpusRejected("Catalog confidentiality does not match the ingestion approval.")
    for chunk in items:
        if chunk.rag_eligibility == "ELIGIBLE":
            approved = validate_document_admission(
                repository_root,
                normalized,
                chunk.document_sha256,
                require_approved=True,
            )
            if chunk.confidentiality != approved["confidentiality"]:
                raise CorpusRejected("Chunk confidentiality does not match the formal catalog admission.")
        elif chunk.document_sha256 is not None:
            validate_document_admission(
                repository_root,
                normalized,
                chunk.document_sha256,
                require_approved=False,
            )
    path = _document_path(repository_root, normalized)
    existing = read_document(repository_root, normalized)

    records_by_id = {record["chunk_id"]: record for record in existing}
    for chunk in items:
        record = asdict(chunk)
        previous = records_by_id.get(chunk.chunk_id)
        if previous is not None and previous != record:
            raise CorpusRejected(
                f"Conflicting content or metadata for existing chunk_id: {chunk.chunk_id}"
            )
        records_by_id[chunk.chunk_id] = record

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records_by_id.values():
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def remove_source(repository_root: str | Path, source_path: str | Path) -> bool:
    """Remove the JSONL file for one allowlisted source; return whether it existed."""
    normalized = Path(source_path).as_posix()
    if normalized not in ALLOWED_SOURCE_PATHS or normalized not in SOURCE_FILES:
        raise CorpusRejected(f"Source is not authorized for corpus persistence: {normalized}")
    path = _document_path(repository_root, normalized)
    if not path.exists():
        return False
    path.unlink()
    return True
