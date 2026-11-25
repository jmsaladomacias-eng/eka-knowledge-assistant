import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from app.chunking import MarkdownChunk
from app.corpus import (
    ADMISSION_FIELDS,
    RAG_AUTHORIZED_AUDIENCE,
    RAG_TARGET_CORPUS,
    SOURCE_FILES,
    CorpusRejected,
    read_document,
    remove_source,
    write_chunks,
)
from app.ingestion import SourceApproval


SOURCE = "docs/architecture.md"
APPROVAL = SourceApproval("internal", "INTERNAL", True)
SYNTHETIC_SOURCE = "# Synthetic source\n\nSynthetic document body.\n"
SYNTHETIC_DIGEST = hashlib.sha256(SYNTHETIC_SOURCE.encode("utf-8")).hexdigest()


def prepare_synthetic_repo(root, *, admission_status="APPROVED", eligibility="ELIGIBLE", audience=None, corpus=None):
    for source_path in SOURCE_FILES:
        path = root / source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SYNTHETIC_SOURCE, encoding="utf-8")
    rows = []
    for source_path in SOURCE_FILES:
        values = {
            "source_path": source_path,
            "admission_status": admission_status,
            "rag_eligibility": eligibility,
            "confidentiality": "INTERNAL",
            "evidence_status": "HISTORICAL",
            "operational_validity": "NOT_ASSESSED",
            "owner": "Synthetic owner",
            "reviewed_at": "2026-11-23" if admission_status == "APPROVED" else "NOT_REVIEWED",
            "reviewer_role": "Synthetic reviewer",
            "review_scope": "Synthetic test scope" if admission_status == "APPROVED" else "NOT_REVIEWED",
            "target_corpus": corpus or RAG_TARGET_CORPUS,
            "authorized_audience": audience or RAG_AUTHORIZED_AUDIENCE,
            "reviewed_sha256": SYNTHETIC_DIGEST if admission_status == "APPROVED" else "NOT_REVIEWED",
        }
        rows.append("| " + " | ".join(values[field] for field in ADMISSION_FIELDS) + " |")
    header = "| " + " | ".join(ADMISSION_FIELDS) + " |"
    separator = "|" + "|".join("---" for _ in ADMISSION_FIELDS) + "|"
    catalog = root / "docs" / "knowledge-catalog.md"
    catalog.write_text(
        "# Synthetic catalog\n\n## Repository RAG admission decisions\n\n"
        + "\n".join((header, separator, *rows))
        + "\n",
        encoding="utf-8",
    )


def sample_chunk(chunk_id="synthetic-1", source_path=SOURCE, **overrides):
    values = {
        "content": "Synthetic local content",
        "source_document": "Synthetic document",
        "source_path": source_path,
        "section": "Overview",
        "chunk_id": chunk_id,
        "evidence_status": "HISTORICAL",
        "operational_validity": "NOT_ASSESSED",
        "confidentiality": "INTERNAL",
        "source_date": "UNKNOWN",
        "review_date": "NOT_REVIEWED",
        "external_llm_approved": False,
        "estimated_tokens": 4,
        "rag_eligibility": None,
        "document_sha256": SYNTHETIC_DIGEST,
    }
    values.update(overrides)
    return MarkdownChunk(**values)


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        prepare_synthetic_repo(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_write_and_read_jsonl(self):
        path = write_chunks(self.root, SOURCE, [sample_chunk()], APPROVAL)
        self.assertEqual(path.name, "architecture.jsonl")
        with path.open(encoding="utf-8") as stream:
            lines = stream.readlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(read_document(self.root, SOURCE)[0]["chunk_id"], "synthetic-1")

    def test_all_metadata_fields_are_preserved(self):
        chunk = sample_chunk()
        write_chunks(self.root, SOURCE, [chunk], APPROVAL)
        record = read_document(self.root, SOURCE)[0]
        expected = {
            "source_document", "source_path", "section", "chunk_id", "evidence_status",
            "operational_validity", "confidentiality", "source_date", "review_date",
            "external_llm_approved", "content", "rag_eligibility", "document_sha256",
        }
        self.assertTrue(expected.issubset(record))
        for key in expected:
            self.assertEqual(record[key], getattr(chunk, key))

    def test_all_rag_eligibility_states_round_trip(self):
        states = ("ELIGIBLE", "ELIGIBLE_AFTER_REVIEW", "EXCLUDED", None)
        chunks = [
            sample_chunk(f"synthetic-{index}", rag_eligibility=state)
            for index, state in enumerate(states)
        ]

        write_chunks(self.root, SOURCE, chunks, APPROVAL)
        records = read_document(self.root, SOURCE)

        self.assertEqual(
            {record["chunk_id"]: record["rag_eligibility"] for record in records},
            {chunk.chunk_id: chunk.rag_eligibility for chunk in chunks},
        )

    def test_legacy_record_without_rag_eligibility_remains_readable_unclassified(self):
        legacy_record = {
            "chunk_id": "legacy-synthetic",
            "content": "Synthetic legacy record",
            "source_document": "Synthetic document",
            "source_path": SOURCE,
            "section": "Overview",
            "evidence_status": "HISTORICAL",
            "operational_validity": "NOT_ASSESSED",
            "confidentiality": "INTERNAL",
            "source_date": "UNKNOWN",
            "review_date": "NOT_REVIEWED",
            "external_llm_approved": False,
        }
        path = self.root / "data" / "corpus" / "architecture.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(legacy_record) + "\n", encoding="utf-8")

        records = read_document(self.root, SOURCE)

        self.assertEqual(records, [legacy_record])
        self.assertNotIn("rag_eligibility", records[0])

    def test_unknown_rag_eligibility_value_is_rejected(self):
        with self.assertRaisesRegex(CorpusRejected, "Unsupported rag_eligibility"):
            write_chunks(
                self.root,
                SOURCE,
                [sample_chunk(rag_eligibility="UNRECOGNIZED")],
                APPROVAL,
            )

    def test_pending_catalog_cannot_persist_an_eligible_chunk(self):
        prepare_synthetic_repo(self.root, admission_status="PENDING", eligibility="ELIGIBLE_AFTER_REVIEW")
        with self.assertRaisesRegex(CorpusRejected, "no formal ELIGIBLE"):
            write_chunks(
                self.root,
                SOURCE,
                [sample_chunk(rag_eligibility="ELIGIBLE")],
                APPROVAL,
            )

    def test_source_without_catalog_entry_is_rejected(self):
        (self.root / "docs" / "knowledge-catalog.md").write_text("# No admission table\n", encoding="utf-8")
        with self.assertRaisesRegex(CorpusRejected, "section is missing"):
            write_chunks(self.root, SOURCE, [sample_chunk()], APPROVAL)

    def test_modified_document_cannot_be_persisted_as_eligible(self):
        (self.root / SOURCE).write_text("# Changed synthetic source\n", encoding="utf-8")
        with self.assertRaisesRegex(CorpusRejected, "digest does not match"):
            write_chunks(self.root, SOURCE, [sample_chunk(rag_eligibility="ELIGIBLE")], APPROVAL)

    def test_incomplete_approved_catalog_entry_is_rejected(self):
        catalog = self.root / "docs" / "knowledge-catalog.md"
        text = catalog.read_text(encoding="utf-8").replace("Synthetic owner", "NOT_ASSIGNED")
        catalog.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(CorpusRejected, "owner is missing"):
            write_chunks(self.root, SOURCE, [sample_chunk(rag_eligibility="ELIGIBLE")], APPROVAL)

    def test_unallowlisted_source_is_rejected(self):
        with self.assertRaisesRegex(CorpusRejected, "not authorized"):
            write_chunks(self.root, "README.md", [sample_chunk(source_path="README.md")], APPROVAL)

    def test_missing_internal_approval_is_rejected(self):
        with self.assertRaisesRegex(CorpusRejected, "Explicit approval"):
            write_chunks(self.root, SOURCE, [sample_chunk()], None)

    def test_confidentiality_incompatible_with_internal_corpus_is_rejected(self):
        approval = SourceApproval("internal", "CONFIDENTIAL", True)
        with self.assertRaisesRegex(CorpusRejected, "Confidentiality is incompatible"):
            write_chunks(self.root, SOURCE, [sample_chunk()], approval)

    def test_external_llm_approval_true_is_rejected(self):
        with self.assertRaisesRegex(CorpusRejected, "external LLM"):
            write_chunks(
                self.root, SOURCE,
                [sample_chunk(external_llm_approved=True)],
                APPROVAL,
            )

    def test_arbitrary_output_paths_are_not_accepted(self):
        with self.assertRaises(TypeError):
            write_chunks(self.root, SOURCE, [sample_chunk()], APPROVAL, output_path="/tmp/out.jsonl")

    def test_repeated_identical_write_is_idempotent(self):
        chunk = sample_chunk()
        write_chunks(self.root, SOURCE, [chunk], APPROVAL)
        write_chunks(self.root, SOURCE, [chunk], APPROVAL)
        self.assertEqual(len(read_document(self.root, SOURCE)), 1)

    def test_conflicting_chunk_id_is_rejected(self):
        write_chunks(self.root, SOURCE, [sample_chunk()], APPROVAL)
        with self.assertRaisesRegex(CorpusRejected, "Conflicting content or metadata"):
            write_chunks(self.root, SOURCE, [sample_chunk(content="Different synthetic text")], APPROVAL)

    def test_conflicting_chunk_ids_within_write_are_rejected(self):
        with self.assertRaisesRegex(CorpusRejected, "Conflicting content or metadata"):
            write_chunks(
                self.root, SOURCE,
                [sample_chunk(), sample_chunk(content="Different synthetic text")], APPROVAL,
            )

    def test_removing_one_source_preserves_another(self):
        storage = "docs/storage.md"
        storage_approval = SourceApproval("internal", "INTERNAL", True)
        write_chunks(self.root, SOURCE, [sample_chunk()], APPROVAL)
        write_chunks(
            self.root, storage,
            [sample_chunk("storage-1", source_path=storage)],
            storage_approval,
        )
        self.assertTrue(remove_source(self.root, SOURCE))
        self.assertEqual(read_document(self.root, SOURCE), [])
        self.assertEqual(len(read_document(self.root, storage)), 1)


    def test_data_directory_symlink_outside_repository_is_rejected_before_io(self):
        repo = self.root / "repo"
        outside = self.root / "outside"
        prepare_synthetic_repo(repo)
        outside.mkdir()
        (repo / "data").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(CorpusRejected, "Symlinked corpus directories"):
            write_chunks(repo, SOURCE, [sample_chunk()], APPROVAL)
        self.assertEqual(list(outside.iterdir()), [])

    def test_corpus_directory_symlink_outside_repository_is_rejected_before_io(self):
        repo = self.root / "repo"
        outside = self.root / "outside"
        prepare_synthetic_repo(repo)
        outside.mkdir()
        (repo / "data").mkdir()
        (repo / "data" / "corpus").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(CorpusRejected, "Symlinked corpus directories"):
            write_chunks(repo, SOURCE, [sample_chunk()], APPROVAL)
        self.assertEqual(list(outside.iterdir()), [])

    def test_jsonl_destination_symlink_is_rejected_for_read_write_and_remove(self):
        corpus = self.root / "data" / "corpus"
        corpus.mkdir(parents=True)
        outside = self.root / "outside.jsonl"
        outside.write_text("synthetic sentinel\n", encoding="utf-8")
        target = corpus / "architecture.jsonl"
        target.symlink_to(outside)
        for operation in (
            lambda: read_document(self.root, SOURCE),
            lambda: write_chunks(self.root, SOURCE, [sample_chunk()], APPROVAL),
            lambda: remove_source(self.root, SOURCE),
        ):
            with self.assertRaisesRegex(CorpusRejected, "Symlinked JSONL files"):
                operation()
        self.assertEqual(outside.read_text(encoding="utf-8"), "synthetic sentinel\n")

    def test_missing_or_empty_corpus_returns_empty_records(self):
        self.assertEqual(read_document(self.root, SOURCE), [])
        corpus_dir = self.root / "data" / "corpus"
        corpus_dir.mkdir(parents=True)
        (corpus_dir / "architecture.jsonl").touch()
        self.assertEqual(read_document(self.root, SOURCE), [])


if __name__ == "__main__":
    unittest.main()
