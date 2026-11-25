"""Section-aware Markdown chunking with source metadata preservation.

Token counts are estimates only: this module estimates 1.3 tokens per
whitespace-delimited word and does not use a tokenizer.
"""

from dataclasses import dataclass, replace
import hashlib
import re
from pathlib import Path

from app.ingestion import SourceApproval, validate_source


UNKNOWN = "UNKNOWN"
NOT_REVIEWED = "NOT_REVIEWED"
RAG_ELIGIBILITY_VALUES = frozenset(
    {"ELIGIBLE", "ELIGIBLE_AFTER_REVIEW", "EXCLUDED"}
)


@dataclass(frozen=True)
class ChunkMetadata:
    source_document: str = UNKNOWN
    source_path: str = UNKNOWN
    evidence_status: str = UNKNOWN
    operational_validity: str = UNKNOWN
    confidentiality: str = UNKNOWN
    source_date: str = UNKNOWN
    review_date: str = NOT_REVIEWED
    external_llm_approved: bool = False
    rag_eligibility: str | None = None
    document_sha256: str | None = None


@dataclass(frozen=True)
class MarkdownChunk:
    content: str
    source_document: str
    source_path: str
    section: str
    chunk_id: str
    evidence_status: str
    operational_validity: str
    confidentiality: str
    source_date: str
    review_date: str
    external_llm_approved: bool
    estimated_tokens: int
    rag_eligibility: str | None = None
    document_sha256: str | None = None


def estimate_tokens(text: str) -> int:
    """Return a rough token estimate, not an exact tokenizer count."""
    words = len(text.split())
    return (words * 13 + 9) // 10


def _sections(markdown: str):
    """Yield original (level, title) heading stacks and body text."""
    heading_stack: list[tuple[int, str]] = []
    body_lines: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush():
        nonlocal body_lines
        value = "\n".join(body_lines).strip()
        body_lines = []
        if value:
            return value
        return None

    for line in markdown.splitlines():
        fence_match = re.match(r"^\s*(```+|~~~+)", line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            body_lines.append(line)
            continue

        if not in_fence:
            heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if heading:
                previous = flush()
                if previous:
                    yield tuple(heading_stack), previous
                level = len(heading.group(1))
                title = heading.group(2).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                continue
        body_lines.append(line)

    remaining = flush()
    if remaining:
        yield tuple(heading_stack), remaining

def _body_blocks(body: str) -> list[str]:
    """Split on blank lines outside fenced code blocks only."""
    blocks: list[str] = []
    lines: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in body.splitlines():
        match = re.match(r"^\s*(```+|~~~+)", line)
        if match:
            marker = match.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            lines.append(line)
        elif not line.strip() and not in_fence:
            if lines:
                blocks.append("\n".join(lines).strip())
                lines = []
        else:
            lines.append(line)
    if lines:
        blocks.append("\n".join(lines).strip())
    return [block for block in blocks if block]


def _split_oversized_block(block: str, max_tokens: int) -> list[str]:
    """Split oversized prose by sentence; preserve fenced code blocks intact."""
    if block.lstrip().startswith(("```", "~~~")):
        return [block]
    sentences = re.split(r"(?<=[.!?])\s+", block)
    pieces: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = " ".join(current + [sentence]).strip()
        if current and estimate_tokens(candidate) > max_tokens:
            pieces.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        pieces.append(" ".join(current))

    bounded: list[str] = []
    for piece in pieces:
        if estimate_tokens(piece) <= max_tokens:
            bounded.append(piece)
            continue
        words = piece.split()
        word_group: list[str] = []
        for word in words:
            candidate = " ".join(word_group + [word])
            if word_group and estimate_tokens(candidate) > max_tokens:
                bounded.append(" ".join(word_group))
                word_group = [word]
            else:
                word_group.append(word)
        if word_group:
            bounded.append(" ".join(word_group))
    return bounded


def chunk_markdown(
    markdown: str,
    metadata: ChunkMetadata,
    *,
    source_authorized: bool = False,
    max_tokens: int = 600,
    overlap_tokens: int = 80,
) -> list[MarkdownChunk]:
    """Chunk previously authorized Markdown; rejects unless caller confirms authorization.

    The defaults are approximate, within the design's 400–700 token target.
    Overlap is applied only between chunks of the same section.
    """
    if not source_authorized:
        raise ValueError("Source must be explicitly authorized before chunking.")
    if max_tokens <= 0 or overlap_tokens < 0:
        raise ValueError("Chunk and overlap estimates must be non-negative; max_tokens must be positive.")
    if not markdown.strip():
        return []

    assembled: list[tuple[str, str]] = []
    active_section: tuple[str, ...] = ()
    current_blocks: list[str] = []

    def emit():
        nonlocal current_blocks
        if current_blocks:
            body = "\n\n".join(current_blocks).strip()
            section_label = " > ".join(title for _, title in active_section) if active_section else UNKNOWN
            prefix = "\n".join(f"{'#' * level} {title}" for level, title in active_section)
            content = f"{prefix}\n\n{body}" if prefix else body
            if content.strip():
                assembled.append((section_label, content))
            current_blocks = []

    for section, body in _sections(markdown):
        if section != active_section:
            emit()
            active_section = section
        heading_prefix = "\n".join(
            f"{'#' * level} {title}" for level, title in active_section
        )
        body_token_budget = max(1, max_tokens - estimate_tokens(heading_prefix))
        for block in _body_blocks(body):
            pieces = _split_oversized_block(block, body_token_budget)
            for piece in pieces:
                candidate_blocks = current_blocks + [piece]
                prefix = "\n".join(f"{'#' * level} {title}" for level, title in active_section)
                candidate = (prefix + "\n\n" if prefix else "") + "\n\n".join(candidate_blocks)
                if current_blocks and estimate_tokens(candidate) > max_tokens:
                    tail = current_blocks[-1] if overlap_tokens else None
                    emit()
                    if tail is not None and estimate_tokens(tail) <= overlap_tokens:
                        current_blocks = [tail]
                    candidate_blocks = current_blocks + [piece]
                current_blocks = candidate_blocks
    emit()

    chunks: list[MarkdownChunk] = []
    for index, (section, content) in enumerate(assembled, start=1):
        digest_input = "\0".join((metadata.source_path, section, str(index), content))
        chunk_id = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:20]
        chunks.append(
            MarkdownChunk(
                content=content,
                source_document=metadata.source_document,
                source_path=metadata.source_path,
                section=section,
                chunk_id=chunk_id,
                evidence_status=metadata.evidence_status,
                operational_validity=metadata.operational_validity,
                confidentiality=metadata.confidentiality,
                source_date=metadata.source_date,
                review_date=metadata.review_date,
                external_llm_approved=metadata.external_llm_approved,
                estimated_tokens=estimate_tokens(content),
                rag_eligibility=metadata.rag_eligibility,
                document_sha256=metadata.document_sha256,
            )
        )
    return chunks


def chunk_source(
    repository_root: str | Path,
    source_path: str | Path,
    approval: SourceApproval,
    metadata: ChunkMetadata,
    *,
    max_tokens: int = 600,
    overlap_tokens: int = 80,
) -> list[MarkdownChunk]:
    """Extract through the existing fail-closed validator, then chunk locally."""
    normalized_source_path = Path(source_path).as_posix()
    if metadata.source_path != normalized_source_path:
        raise ValueError(
            "Chunk metadata source_path does not match the authorized source path."
        )
    if metadata.confidentiality != approval.confidentiality:
        raise ValueError(
            "Chunk metadata confidentiality does not match the source approval."
        )
    source_file = validate_source(repository_root, source_path, approval)
    source_bytes = source_file.read_bytes()
    # Import locally because corpus.py imports MarkdownChunk from this module.
    from app.corpus import validate_document_admission

    document_sha256 = hashlib.sha256(source_bytes).hexdigest()
    admission = validate_document_admission(
        repository_root,
        normalized_source_path,
        document_sha256,
        require_approved=False,
    )
    if admission["rag_eligibility"] == "ELIGIBLE":
        validate_document_admission(
            repository_root,
            normalized_source_path,
            document_sha256,
            require_approved=True,
        )
    elif (
        metadata.rag_eligibility == "ELIGIBLE"
        and (
            admission["admission_status"] != "APPROVED"
            or admission["rag_eligibility"] != "ELIGIBLE"
        )
    ):
        raise ValueError(
            "Document has no formal ELIGIBLE RAG admission."
        )
    markdown = source_bytes.decode("utf-8")
    source_metadata = replace(
        metadata,
        document_sha256=document_sha256,
    )
    return chunk_markdown(
        markdown,
        source_metadata,
        source_authorized=True,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
