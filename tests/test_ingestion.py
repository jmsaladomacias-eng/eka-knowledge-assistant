import tempfile
import unittest
from pathlib import Path

from app.ingestion import (
    ALLOWED_SOURCE_PATHS,
    IngestionRejected,
    SourceApproval,
    extract_markdown,
    validate_source,
)


VALID_SOURCE = "docs/architecture.md"
VALID_TEXT = "# Temporary test document\n\nLocal test content.\n"
APPROVED = SourceApproval(
    corpus="internal",
    confidentiality="INTERNAL",
    security_review_complete=True,
)


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / VALID_SOURCE
        self.source.parent.mkdir(parents=True)
        self.source.write_text(VALID_TEXT, encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_source_not_in_allowlist_is_rejected(self):
        with self.assertRaisesRegex(IngestionRejected, "not in the ingestion allowlist"):
            validate_source(self.root, "README.md", APPROVED)

    def test_allowlisted_source_without_approval_is_rejected(self):
        with self.assertRaisesRegex(IngestionRejected, "Explicit approval"):
            extract_markdown(self.root, VALID_SOURCE)

    def test_approved_valid_source_is_accepted_and_extracted(self):
        self.assertIn(VALID_SOURCE, ALLOWED_SOURCE_PATHS)
        extracted = extract_markdown(self.root, VALID_SOURCE, APPROVED)
        self.assertEqual(extracted, VALID_TEXT)
        self.assertEqual(self.source.read_text(encoding="utf-8"), VALID_TEXT)

    def test_missing_source_is_rejected(self):
        self.source.unlink()
        with self.assertRaisesRegex(IngestionRejected, "does not exist"):
            validate_source(self.root, VALID_SOURCE, APPROVED)

    def test_non_markdown_extension_is_rejected(self):
        with self.assertRaisesRegex(IngestionRejected, "Only .md sources are permitted"):
            validate_source(self.root, "docs/architecture.txt", APPROVED)

    def test_traversal_path_is_rejected(self):
        with self.assertRaisesRegex(IngestionRejected, "not in the ingestion allowlist"):
            validate_source(self.root, "../outside.md", APPROVED)

    def test_symlink_source_is_rejected(self):
        self.source.unlink()
        outside = self.root.parent / f"{self.root.name}-outside.md"
        outside.write_text(VALID_TEXT, encoding="utf-8")
        try:
            self.source.symlink_to(outside)
            with self.assertRaisesRegex(IngestionRejected, "Symlinks are not permitted"):
                validate_source(self.root, VALID_SOURCE, APPROVED)
        finally:
            outside.unlink(missing_ok=True)

    def test_confidential_source_is_rejected(self):
        approval = SourceApproval(
            corpus="internal",
            confidentiality="CONFIDENTIAL",
            security_review_complete=True,
        )
        with self.assertRaisesRegex(IngestionRejected, "Confidentiality is not compatible"):
            validate_source(self.root, VALID_SOURCE, approval)

    def test_pending_security_review_is_rejected(self):
        approval = SourceApproval(corpus="internal", confidentiality="INTERNAL")
        with self.assertRaisesRegex(IngestionRejected, "Security review is pending"):
            validate_source(self.root, VALID_SOURCE, approval)


if __name__ == "__main__":
    unittest.main()
