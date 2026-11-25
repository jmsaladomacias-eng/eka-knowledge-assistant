"""Deterministic, local lexical retrieval over the approved JSONL corpus."""

from collections import Counter
import math
import re
from pathlib import Path

from app.corpus import SOURCE_FILES, read_document


DEFAULT_TOP_K = 5
_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold())


def _tfidf_vector(tokens: list[str], document_frequency: Counter, count: int) -> dict[str, float]:
    if not tokens:
        return {}
    frequencies = Counter(tokens)
    length = len(tokens)
    vector = {
        term: (frequency / length) * (math.log(1 + count / (1 + document_frequency[term])))
        for term, frequency in frequencies.items()
    }
    norm = math.sqrt(sum(weight * weight for weight in vector.values()))
    return {term: weight / norm for term, weight in vector.items()} if norm else {}


def search_corpus(
    repository_root: str | Path,
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Return the highest-scoring lexical matches from the five approved JSONL files.

    Results are copies of their corpus records with ``relevance_score`` added.
    The corpus is read only; empty questions and empty corpora return no results.
    """
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    # Iterate this fixed mapping only; no directory scanning or source-document access.
    records: list[dict] = []
    for source_path in SOURCE_FILES:
        records.extend(read_document(repository_root, source_path))

    if not records:
        return []
    query_tokens = _tokens(question)
    if not query_tokens:
        return []

    unique_records: list[dict] = []
    records_by_id: dict[str, dict] = {}
    for record in records:
        chunk_id = record.get("chunk_id")
        content = record.get("content")
        if not isinstance(chunk_id, str) or not isinstance(content, str):
            raise ValueError("Corpus record must contain string chunk_id and content fields")
        previous = records_by_id.get(chunk_id)
        if previous is not None:
            if previous != record:
                raise ValueError(f"Conflicting corpus records share chunk_id: {chunk_id}")
            continue
        records_by_id[chunk_id] = record
        unique_records.append(record)

    tokenized = [_tokens(record["content"]) for record in unique_records]
    document_frequency: Counter = Counter()
    for terms in tokenized:
        document_frequency.update(set(terms))

    count = len(unique_records)
    query_vector = _tfidf_vector(query_tokens, document_frequency, count)
    if not query_vector:
        return []

    ranked: list[tuple[float, dict]] = []
    for record, terms in zip(unique_records, tokenized):
        document_vector = _tfidf_vector(terms, document_frequency, count)
        score = sum(
            query_weight * document_vector.get(term, 0.0)
            for term, query_weight in query_vector.items()
        )
        if score > 0:
            ranked.append((score, record))

    ranked.sort(
        key=lambda item: (
            -item[0],
            item[1].get("source_path", ""),
            item[1].get("section", ""),
            item[1]["chunk_id"],
        )
    )
    return [{**record, "relevance_score": score} for score, record in ranked[:top_k]]
