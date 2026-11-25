import tempfile
import unittest
import hashlib
from pathlib import Path

from app.chunking import ChunkMetadata, chunk_markdown, chunk_source
from app.corpus import ADMISSION_FIELDS, RAG_AUTHORIZED_AUDIENCE, RAG_TARGET_CORPUS
from app.ingestion import SourceApproval


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.metadata = ChunkMetadata(
            source_document="Synthetic test source",
            source_path="docs/architecture.md",
            evidence_status="HISTORICAL",
            operational_validity="NOT_ASSESSED",
            confidentiality="INTERNAL",
            source_date="UNKNOWN",
            review_date="NOT_REVIEWED",
        )

    def test_short_document_produces_one_chunk(self):
        chunks = chunk_markdown(
            "# Short\n\nA short synthetic paragraph.", self.metadata, source_authorized=True
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("# Short", chunks[0].content)

    def test_long_document_is_split(self):
        paragraphs = [f"Paragraph {i}. " + ("synthetic words " * 90) for i in range(12)]
        markdown = "# Long\n\n" + "\n\n".join(paragraphs)
        chunks = chunk_markdown(
            markdown, self.metadata, source_authorized=True, max_tokens=450, overlap_tokens=80
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.estimated_tokens > 0 for chunk in chunks))

    def test_headers_and_hierarchy_are_preserved(self):
        markdown = "# Root\n\nIntro.\n\n## Child\n\nBody.\n\n### Leaf\n\nDetails."
        chunks = chunk_markdown(markdown, self.metadata, source_authorized=True)
        by_section = {chunk.section: chunk.content for chunk in chunks}
        self.assertIn("Root", by_section["Root"])
        self.assertEqual(next(c for c in chunks if c.section == "Root > Child").section, "Root > Child")
        self.assertIn("# Root", by_section["Root > Child"])
        self.assertIn("## Child", by_section["Root > Child"])
        self.assertIn("# Root", by_section["Root > Child > Leaf"])
        self.assertIn("### Leaf", by_section["Root > Child > Leaf"])

    def test_skipped_heading_levels_are_preserved(self):
        markdown = "# Root\n\nRoot text.\n\n### Leaf\n\nLeaf text."
        chunks = chunk_markdown(markdown, self.metadata, source_authorized=True)
        leaf = next(chunk for chunk in chunks if chunk.section == "Root > Leaf")
        self.assertIn("# Root", leaf.content)
        self.assertIn("### Leaf", leaf.content)
        self.assertNotRegex(leaf.content, r"(?m)^## Leaf$")

    def test_unpunctuated_long_paragraph_is_split_by_words(self):
        markdown = "# Long\n\n" + ("word " * 1000)
        chunks = chunk_markdown(
            markdown, self.metadata, source_authorized=True, max_tokens=100, overlap_tokens=0
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.estimated_tokens <= 100 for chunk in chunks))

    def test_fenced_code_block_is_kept_intact(self):
        code = "```python\n" + "print('synthetic')\n" * 30 + "```"
        chunks = chunk_markdown(
            "# Code\n\nBefore.\n\n" + code + "\n\nAfter.",
            self.metadata,
            source_authorized=True,
            max_tokens=20,
        )
        self.assertTrue(any(code in chunk.content for chunk in chunks))

    def test_metadata_and_provenance_are_present_on_every_chunk(self):
        markdown = "# A\n\n" + ("Synthetic text. " * 300)
        chunks = chunk_markdown(
            markdown, self.metadata, source_authorized=True, max_tokens=200, overlap_tokens=0
        )
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.source_document, self.metadata.source_document)
            self.assertEqual(chunk.source_path, self.metadata.source_path)
            self.assertTrue(chunk.section)
            self.assertEqual(chunk.evidence_status, "HISTORICAL")
            self.assertEqual(chunk.operational_validity, "NOT_ASSESSED")
            self.assertEqual(chunk.confidentiality, "INTERNAL")
            self.assertEqual(chunk.source_date, "UNKNOWN")
            self.assertEqual(chunk.review_date, "NOT_REVIEWED")
            self.assertFalse(chunk.external_llm_approved)
            self.assertTrue(chunk.chunk_id)

    def test_rag_eligibility_values_are_propagated_without_inference(self):
        for eligibility in (
            "ELIGIBLE",
            "ELIGIBLE_AFTER_REVIEW",
            "EXCLUDED",
            None,
        ):
            with self.subTest(rag_eligibility=eligibility):
                metadata = ChunkMetadata(
                    source_document="Synthetic source",
                    source_path="docs/architecture.md",
                    confidentiality="INTERNAL",
                    rag_eligibility=eligibility,
                )
                chunks = chunk_markdown(
                    "# Synthetic\n\nSynthetic body.",
                    metadata,
                    source_authorized=True,
                )
                self.assertEqual(len(chunks), 1)
                self.assertEqual(chunks[0].rag_eligibility, eligibility)

    def test_identifiers_are_stable(self):
        markdown = "# Stable\n\nSame synthetic content."
        first = chunk_markdown(markdown, self.metadata, source_authorized=True)
        second = chunk_markdown(markdown, self.metadata, source_authorized=True)
        self.assertEqual([c.chunk_id for c in first], [c.chunk_id for c in second])

    def test_empty_document_returns_no_chunks(self):
        self.assertEqual(chunk_markdown(" \n\n", self.metadata, source_authorized=True), [])

    def test_document_is_not_modified_during_authorized_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "docs/architecture.md"
            path.parent.mkdir(parents=True)
            original = "# Synthetic\n\nUnchanged source content.\n"
            path.write_text(original, encoding="utf-8")
            self._write_admission_catalog(
                root,
                reviewed_sha256=hashlib.sha256(original.encode("utf-8")).hexdigest(),
            )
            approval = SourceApproval(
                corpus="internal",
                confidentiality="INTERNAL",
                security_review_complete=True,
            )
            chunks = chunk_source(root, "docs/architecture.md", approval, self.metadata)
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(
                {chunk.document_sha256 for chunk in chunks},
                {hashlib.sha256(original.encode("utf-8")).hexdigest()},
            )

    @staticmethod
    def _write_admission_catalog(
        root,
        *,
        status="APPROVED",
        eligibility="ELIGIBLE",
        reviewed_sha256=None,
        include_source=True,
    ):
        values = {
            "source_path": "docs/architecture.md",
            "admission_status": status,
            "rag_eligibility": eligibility,
            "confidentiality": "INTERNAL",
            "evidence_status": "HISTORICAL",
            "operational_validity": "NOT_ASSESSED",
            "owner": "Synthetic owner" if status == "APPROVED" else "NOT_ASSIGNED",
            "reviewed_at": "2026-11-23" if status == "APPROVED" else "NOT_REVIEWED",
            "reviewer_role": "Synthetic reviewer" if status == "APPROVED" else "NOT_ASSIGNED",
            "review_scope": "Synthetic test scope" if status == "APPROVED" else "NOT_REVIEWED",
            "target_corpus": RAG_TARGET_CORPUS,
            "authorized_audience": RAG_AUTHORIZED_AUDIENCE,
            "reviewed_sha256": reviewed_sha256 or "NOT_REVIEWED",
        }
        rows = []
        if include_source:
            rows.append("| " + " | ".join(values[field] for field in ADMISSION_FIELDS) + " |")
        header = "| " + " | ".join(ADMISSION_FIELDS) + " |"
        separator = "|" + "|".join("---" for _ in ADMISSION_FIELDS) + "|"
        (root / "docs" / "knowledge-catalog.md").write_text(
            "# Synthetic catalog\n\n## Repository RAG admission decisions\n\n"
            + "\n".join((header, separator, *rows))
            + "\n",
            encoding="utf-8",
        )

    def _synthetic_chunk_source(self, root, *, eligibility=None):
        source = "# Synthetic\n\nSynthetic source content.\n"
        source_path = root / "docs" / "architecture.md"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(source, encoding="utf-8")
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        self._write_admission_catalog(
            root,
            reviewed_sha256=digest,
            eligibility=eligibility or "ELIGIBLE",
        )
        metadata = ChunkMetadata(
            source_document="Synthetic source",
            source_path="docs/architecture.md",
            confidentiality="INTERNAL",
            rag_eligibility=eligibility,
        )
        approval = SourceApproval("internal", "INTERNAL", True)
        return source, metadata, approval

    def test_chunk_source_requires_a_catalogue_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, metadata, approval = self._synthetic_chunk_source(root)
            self._write_admission_catalog(root, include_source=False)
            with self.assertRaisesRegex(ValueError, "No formal RAG admission record"):
                chunk_source(root, "docs/architecture.md", approval, metadata)

    def test_pending_source_produces_only_noneligible_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, metadata, approval = self._synthetic_chunk_source(
                root, eligibility="ELIGIBLE_AFTER_REVIEW"
            )
            self._write_admission_catalog(
                root,
                status="PENDING",
                eligibility="ELIGIBLE_AFTER_REVIEW",
            )
            chunks = chunk_source(root, "docs/architecture.md", approval, metadata)
            self.assertTrue(chunks)
            self.assertTrue(all(chunk.rag_eligibility != "ELIGIBLE" for chunk in chunks))
            self.assertTrue(all(not chunk.external_llm_approved for chunk in chunks))

    def test_valid_eligible_source_with_matching_hash_is_chunked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, metadata, approval = self._synthetic_chunk_source(
                root, eligibility="ELIGIBLE"
            )
            chunks = chunk_source(root, "docs/architecture.md", approval, metadata)
            self.assertTrue(chunks)
            self.assertTrue(all(chunk.rag_eligibility == "ELIGIBLE" for chunk in chunks))
            self.assertTrue(all(
                chunk.document_sha256 == hashlib.sha256(source.encode("utf-8")).hexdigest()
                for chunk in chunks
            ))
            self.assertTrue(all(not chunk.external_llm_approved for chunk in chunks))

    def test_excluded_source_cannot_produce_eligible_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, metadata, approval = self._synthetic_chunk_source(
                root, eligibility="ELIGIBLE"
            )
            self._write_admission_catalog(
                root,
                status="APPROVED",
                eligibility="EXCLUDED",
            )
            with self.assertRaisesRegex(ValueError, "no formal ELIGIBLE"):
                chunk_source(root, "docs/architecture.md", approval, metadata)

    def test_missing_or_mismatched_reviewed_hash_fails_closed(self):
        for reviewed_sha256 in ("NOT_REVIEWED", "0" * 64):
            with self.subTest(reviewed_sha256=reviewed_sha256):
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    source, metadata, approval = self._synthetic_chunk_source(
                        root, eligibility="ELIGIBLE"
                    )
                    self._write_admission_catalog(
                        root,
                        reviewed_sha256=reviewed_sha256,
                        eligibility="ELIGIBLE",
                    )
                    with self.assertRaises(ValueError):
                        chunk_source(root, "docs/architecture.md", approval, metadata)

    def test_chunk_source_rejects_mismatched_source_path_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "docs/architecture.md"
            path.parent.mkdir(parents=True)
            path.write_text("# Synthetic\n\nText.", encoding="utf-8")
            approval = SourceApproval("internal", "INTERNAL", True)
            metadata = ChunkMetadata(source_path="docs/storage.md", confidentiality="INTERNAL")
            with self.assertRaisesRegex(ValueError, "source_path does not match"):
                chunk_source(root, "docs/architecture.md", approval, metadata)

    def test_chunk_source_rejects_mismatched_confidentiality_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "docs/architecture.md"
            path.parent.mkdir(parents=True)
            path.write_text("# Synthetic\n\nText.", encoding="utf-8")
            approval = SourceApproval("internal", "INTERNAL", True)
            metadata = ChunkMetadata(source_path="docs/architecture.md", confidentiality="PUBLIC")
            with self.assertRaisesRegex(ValueError, "confidentiality does not match"):
                chunk_source(root, "docs/architecture.md", approval, metadata)

    def test_chunking_requires_explicit_source_authorization(self):
        with self.assertRaisesRegex(ValueError, "explicitly authorized"):
            chunk_markdown("Synthetic content", self.metadata)


if __name__ == "__main__":
    unittest.main()
