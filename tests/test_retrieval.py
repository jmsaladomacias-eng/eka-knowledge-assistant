import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.corpus import SOURCE_FILES
from app.retrieval import search_corpus


def make_record(chunk_id, content, source_path="docs/architecture.md", section="Overview"):
    return {
        "chunk_id": chunk_id,
        "content": content,
        "source_document": source_path,
        "source_path": source_path,
        "section": section,
        "evidence_status": "HISTORICAL",
        "operational_validity": "NOT_ASSESSED",
        "confidentiality": "INTERNAL",
        "source_date": "UNKNOWN",
        "review_date": "NOT_REVIEWED",
        "external_llm_approved": False,
        "rag_eligibility": "ELIGIBLE",
        "document_sha256": "synthetic-document-digest",
    }


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        corpus = self.root / "data" / "corpus"
        corpus.mkdir(parents=True)
        self.records = {
            "architecture.jsonl": [
                make_record("arch-1", "Plex media services use a documented storage mount."),
                make_record("arch-2", "Engineering workspace documentation describes project structure."),
            ],
            "storage.jsonl": [
                make_record("storage-1", "MergerFS category.create=epmfs combines storage branches.", "docs/storage.md", "MergerFS"),
            ],
            "operations.jsonl": [],
            "current-state.jsonl": [],
            "documentation-reconciliation.jsonl": [],
        }
        for name, records in self.records.items():
            path = corpus / name
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_technical_terms_retrieve_matching_chunk(self):
        results = search_corpus(self.root, "category.create=epmfs")
        self.assertEqual([result["chunk_id"] for result in results], ["storage-1"])

    def test_results_are_sorted_by_relevance(self):
        results = search_corpus(self.root, "Plex media media storage mount")
        scores = [result["relevance_score"] for result in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(results[0]["chunk_id"], "arch-1")

    def test_no_lexical_matches_returns_empty_list(self):
        self.assertEqual(search_corpus(self.root, "unrelatedxyzterm"), [])

    def test_top_k_limits_result_count(self):
        results = search_corpus(self.root, "media storage documentation", top_k=1)
        self.assertEqual(len(results), 1)

    def test_source_section_and_evidence_metadata_are_preserved(self):
        result = search_corpus(self.root, "category.create=epmfs")[0]
        self.assertEqual(result["chunk_id"], "storage-1")
        self.assertEqual(result["source_document"], "docs/storage.md")
        self.assertEqual(result["source_path"], "docs/storage.md")
        self.assertEqual(result["section"], "MergerFS")
        self.assertEqual(result["evidence_status"], "HISTORICAL")
        self.assertEqual(result["operational_validity"], "NOT_ASSESSED")
        self.assertEqual(result["confidentiality"], "INTERNAL")
        self.assertFalse(result["external_llm_approved"])
        self.assertEqual(result["rag_eligibility"], "ELIGIBLE")
        self.assertEqual(result["document_sha256"], "synthetic-document-digest")
        self.assertGreater(result["relevance_score"], 0)

    def test_only_the_five_allowlisted_files_are_read(self):
        from app import retrieval

        original = retrieval.read_document
        calls = []

        def recording_read(root, source_path):
            calls.append(source_path)
            return original(root, source_path)

        with patch("app.retrieval.read_document", side_effect=recording_read):
            search_corpus(self.root, "media")
        self.assertEqual(set(calls), set(SOURCE_FILES))
        self.assertEqual(len(calls), 5)

    def test_missing_or_empty_corpus_returns_no_results(self):
        empty_root = Path(self.temp.name) / "empty"
        empty_root.mkdir()
        self.assertEqual(search_corpus(empty_root, "anything"), [])

    def test_empty_or_whitespace_question_returns_no_results(self):
        self.assertEqual(search_corpus(self.root, ""), [])
        self.assertEqual(search_corpus(self.root, "   \t\n  "), [])

    def test_results_are_deterministic(self):
        first = search_corpus(self.root, "media storage documentation")
        second = search_corpus(self.root, "media storage documentation")
        self.assertEqual(first, second)

    def test_retrieval_does_not_modify_corpus(self):
        corpus = self.root / "data" / "corpus"
        before = {path.name: path.read_bytes() for path in corpus.iterdir()}
        search_corpus(self.root, "media")
        after = {path.name: path.read_bytes() for path in corpus.iterdir()}
        self.assertEqual(after, before)

    def test_invalid_top_k_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            search_corpus(self.root, "media", top_k=0)


if __name__ == "__main__":
    unittest.main()
