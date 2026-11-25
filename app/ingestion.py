"""Fail-closed validation and local Markdown extraction for approved sources."""

from dataclasses import dataclass
from pathlib import Path


ALLOWED_SOURCE_PATHS = frozenset(
    {
        "docs/architecture.md",
        "docs/storage.md",
        "docs/operations.md",
        "docs/infrastructure/current-state.md",
        "docs/infrastructure/documentation-reconciliation.md",
    }
)
INTERNAL_CORPUS = "internal"
ALLOWED_CONFIDENTIALITY = frozenset({"PUBLIC", "INTERNAL"})


class IngestionRejected(ValueError):
    """Raised when a source fails an ingestion admission check."""


@dataclass(frozen=True)
class SourceApproval:
    """Explicit per-source approval; no approvals are configured by default."""

    corpus: str | None = None
    confidentiality: str | None = None
    security_review_complete: bool = False


def validate_source(
    repository_root: str | Path,
    source_path: str | Path,
    approval: SourceApproval | None = None,
) -> Path:
    """Validate an allowlisted Markdown source and return its resolved path.

    The caller must supply explicit approval for the internal corpus, a compatible
    confidentiality classification, and completed security review. No secret
    detection is attempted here; that review must happen outside this module.
    """
    root = Path(repository_root).resolve(strict=True)
    relative = Path(source_path)

    if relative.is_absolute():
        raise IngestionRejected("Source path must be repository-relative.")

    if relative.suffix != ".md":
        raise IngestionRejected("Only .md sources are permitted.")

    normalized = relative.as_posix()
    if normalized not in ALLOWED_SOURCE_PATHS:
        raise IngestionRejected(f"Source is not in the ingestion allowlist: {normalized}")

    candidate = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise IngestionRejected("Symlinks are not permitted in source paths.")

    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise IngestionRejected(f"Source file does not exist: {normalized}") from exc

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise IngestionRejected("Resolved source path is outside the repository.") from exc

    if not resolved.is_file():
        raise IngestionRejected("Source path is not a regular file.")

    if approval is None or approval.corpus != INTERNAL_CORPUS:
        raise IngestionRejected("Explicit approval for the internal corpus is required.")
    if approval.confidentiality not in ALLOWED_CONFIDENTIALITY:
        raise IngestionRejected(
            "Confidentiality is not compatible with the internal corpus admission policy."
        )
    if not approval.security_review_complete:
        raise IngestionRejected(
            "Security review is pending; sensitive content must be reviewed before ingestion."
        )

    return resolved


def extract_markdown(
    repository_root: str | Path,
    source_path: str | Path,
    approval: SourceApproval | None = None,
) -> str:
    """Validate and extract UTF-8 Markdown locally; performs no network access."""
    resolved = validate_source(repository_root, source_path, approval)
    try:
        return resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise IngestionRejected("Markdown source is not valid UTF-8 text.") from exc
