import json
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.answer_engine import AnswerEngine
from app.corpus import ADMISSION_FIELDS, RAG_AUTHORIZED_AUDIENCE, RAG_TARGET_CORPUS, SOURCE_FILES
from app.llm import LLMProvider, LLMRequest, LLMResponse
from app.llm.anthropic import AnthropicLLMProvider
from app.llm.mock import MockLLMProvider


SYNTHETIC_SOURCE = "# Synthetic source\n\nSynthetic test evidence.\n"
SYNTHETIC_DIGEST = hashlib.sha256(SYNTHETIC_SOURCE.encode("utf-8")).hexdigest()


def record(
    chunk_id,
    content,
    *,
    source_path="docs/architecture.md",
    section="Synthetic section",
    external_approved=True,
    evidence_status="VERIFIED",
    operational_validity="CURRENTLY_VALIDATED",
    confidentiality="INTERNAL",
    source_date="UNKNOWN",
    review_date="NOT_REVIEWED",
    rag_eligibility="ELIGIBLE",
):
    result = {
        "chunk_id": chunk_id,
        "content": content,
        "source_document": source_path.rsplit("/", 1)[-1],
        "source_path": source_path,
        "section": section,
        "evidence_status": evidence_status,
        "operational_validity": operational_validity,
        "confidentiality": confidentiality,
        "source_date": source_date,
        "review_date": review_date,
        "external_llm_approved": external_approved,
        "rag_eligibility": rag_eligibility,
        "document_sha256": SYNTHETIC_DIGEST,
    }
    return result


def output(answer, citations=(), abstain=False, **extra):
    payload = {
        "answer": answer,
        "citations": list(citations),
        "abstain": abstain,
    }
    payload.update(extra)
    return json.dumps(payload)


def fake_anthropic_provider(response_text):
    fake_client = Mock()
    fake_client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=response_text)]
    )
    with patch.dict(
        os.environ,
        {"ANTHROPIC_API_KEY": "synthetic-key", "ANTHROPIC_MODEL": "synthetic-model"},
    ), patch("app.llm.anthropic.AnthropicClient", return_value=fake_client):
        provider = AnthropicLLMProvider()
    return provider, fake_client


class RecordingMockProvider(MockLLMProvider):
    def __init__(self, response_text):
        super().__init__(response_text)
        self.calls = 0
        self.last_request = None

    def generate(self, request):
        self.calls += 1
        self.last_request = request
        return super().generate(request)


class AnswerEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for source_path in SOURCE_FILES:
            path = self.root / source_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(SYNTHETIC_SOURCE, encoding="utf-8")
        self._write_catalog()
        self.corpus_dir = self.root / "data" / "corpus"
        self.corpus_dir.mkdir(parents=True)
        self.architecture_path = self.corpus_dir / "architecture.jsonl"
        self.storage_path = self.corpus_dir / "storage.jsonl"
        self.architecture_records = [
            record(
                "synthetic-a",
                "Synthetic appliance alpha has documented state: enabled.",
                section="Alpha state",
                evidence_status="HISTORICAL",
                operational_validity="REVALIDATION_REQUIRED",
            ),
        ]
        self.storage_records = [
            record(
                "synthetic-b",
                "Synthetic appliance alpha has documented state: disabled.",
                source_path="docs/storage.md",
                section="Alpha storage state",
                evidence_status="HYPOTHESIS",
                operational_validity="UNKNOWN",
            )
        ]
        self._write_jsonl(self.architecture_path, self.architecture_records)
        self._write_jsonl(self.storage_path, self.storage_records)
        self.question = "What is the synthetic appliance alpha state?"

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _write_jsonl(path, records):
        path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
            encoding="utf-8",
        )

    def _engine(self, provider):
        return AnswerEngine(self.root, provider, top_k=5)

    def _write_catalog(
        self,
        *,
        admission_status="APPROVED",
        eligibility="ELIGIBLE",
        target_corpus=RAG_TARGET_CORPUS,
        audience=RAG_AUTHORIZED_AUDIENCE,
        reviewed_digest=SYNTHETIC_DIGEST,
        include_sources=None,
    ):
        rows = []
        for source_path in SOURCE_FILES:
            if include_sources is not None and source_path not in include_sources:
                continue
            values = {
                "source_path": source_path,
                "admission_status": admission_status,
                "rag_eligibility": eligibility,
                "confidentiality": "INTERNAL",
                "evidence_status": "HISTORICAL" if source_path.endswith(("architecture.md", "storage.md")) else "VERIFIED",
                "operational_validity": "NOT_ASSESSED",
                "owner": "Synthetic owner",
                "reviewed_at": "2026-11-23" if admission_status == "APPROVED" else "NOT_REVIEWED",
                "reviewer_role": "Synthetic reviewer",
                "review_scope": "Synthetic test scope" if admission_status == "APPROVED" else "NOT_REVIEWED",
                "target_corpus": target_corpus,
                "authorized_audience": audience,
                "reviewed_sha256": reviewed_digest if admission_status == "APPROVED" else "NOT_REVIEWED",
            }
            rows.append("| " + " | ".join(values[field] for field in ADMISSION_FIELDS) + " |")
        header = "| " + " | ".join(ADMISSION_FIELDS) + " |"
        separator = "|" + "|".join("---" for _ in ADMISSION_FIELDS) + "|"
        catalog = self.root / "docs" / "knowledge-catalog.md"
        catalog.write_text(
            "# Synthetic catalog\n\n## Repository RAG admission decisions\n\n"
            + "\n".join((header, separator, *rows))
            + "\n",
            encoding="utf-8",
        )

    def test_answer_with_sufficient_evidence_and_verified_citation(self):
        provider = RecordingMockProvider(output("The record says enabled.", ["synthetic-a"]))
        result = self._engine(provider).answer(
            self.question, question_external_llm_approved=True
        )

        self.assertFalse(result.abstained)
        self.assertEqual(result.answer, "The record says enabled.")
        self.assertEqual(len(result.citations), 1)
        citation = result.citations[0]
        self.assertEqual(citation.chunk_id, "synthetic-a")
        self.assertEqual(citation.source_path, "docs/architecture.md")
        self.assertEqual(citation.section, "Alpha state")
        self.assertEqual(citation.evidence_status, "HISTORICAL")
        self.assertEqual(citation.operational_validity, "REVALIDATION_REQUIRED")
        self.assertEqual(citation.source_date, "UNKNOWN")
        self.assertEqual(citation.review_date, "NOT_REVIEWED")
        self.assertNotIn("relevance_score", provider.last_request.user_prompt)

    def test_multiple_sources_are_sent_and_cited(self):
        provider = RecordingMockProvider(
            output("The synthetic records conflict.", ["synthetic-a", "synthetic-b"], abstain=True)
        )
        result = self._engine(provider).answer(
            self.question, question_external_llm_approved=True
        )

        self.assertTrue(result.abstained)
        self.assertEqual(
            {citation.source_path for citation in result.citations},
            {"docs/architecture.md", "docs/storage.md"},
        )
        self.assertIn("Synthetic appliance alpha has documented state: enabled.", provider.last_request.user_prompt)
        self.assertIn("Synthetic appliance alpha has documented state: disabled.", provider.last_request.user_prompt)

    def test_historical_status_and_temporal_limits_are_preserved(self):
        provider = RecordingMockProvider(output("This is historical, not confirmed current.", ["synthetic-a"]))
        self._engine(provider).answer(self.question, question_external_llm_approved=True)

        prompt = provider.last_request.system_prompt + provider.last_request.user_prompt
        self.assertIn("HISTORICAL", prompt)
        self.assertIn("REVALIDATION_REQUIRED", prompt)
        self.assertIn('"source_date": "UNKNOWN"', prompt)
        self.assertIn('"review_date": "NOT_REVIEWED"', prompt)
        self.assertIn("Do not use outside knowledge or infer", provider.last_request.system_prompt)
        self.assertIn("current state from historical records.", provider.last_request.system_prompt)

    def test_known_source_and_review_dates_are_preserved(self):
        dated = record(
            "synthetic-dated",
            "Synthetic date-bound fact about the appliance.",
            source_date="SYNTHETIC_SOURCE_DATE_2026_01_10",
            review_date="SYNTHETIC_REVIEW_DATE_2026_01_12",
        )
        self._write_jsonl(self.architecture_path, [dated])
        self.storage_path.unlink()
        provider = RecordingMockProvider(output("Synthetic dated answer.", ["synthetic-dated"]))

        result = self._engine(provider).answer(
            "What is the synthetic date-bound appliance fact?",
            question_external_llm_approved=True,
        )

        self.assertIn("SYNTHETIC_SOURCE_DATE_2026_01_10", provider.last_request.user_prompt)
        self.assertIn("SYNTHETIC_REVIEW_DATE_2026_01_12", provider.last_request.user_prompt)
        self.assertEqual(result.citations[0].source_date, "SYNTHETIC_SOURCE_DATE_2026_01_10")
        self.assertEqual(result.citations[0].review_date, "SYNTHETIC_REVIEW_DATE_2026_01_12")

    def test_insufficient_evidence_can_return_provider_abstention(self):
        provider = RecordingMockProvider(output("The supplied evidence is insufficient.", abstain=True))
        result = self._engine(provider).answer(
            self.question, question_external_llm_approved=True
        )

        self.assertTrue(result.abstained)
        self.assertEqual(result.reason, "insufficient_or_conflicting_evidence")
        self.assertEqual(result.citations, ())

    def test_unsupported_question_without_retrieved_evidence_skips_provider(self):
        provider = RecordingMockProvider(output("Should not be called.", ["synthetic-a"]))
        result = self._engine(provider).answer(
            "What is the flarnigibble value?",
            question_external_llm_approved=True,
        )

        self.assertTrue(result.abstained)
        self.assertEqual(result.reason, "no_externally_authorized_evidence")
        self.assertEqual(provider.calls, 0)

    def test_prompt_forbids_using_knowledge_outside_supplied_evidence(self):
        provider = RecordingMockProvider(output("The supplied record says enabled.", ["synthetic-a"]))
        self._engine(provider).answer(
            self.question, question_external_llm_approved=True
        )

        self.assertIn("Answer only from the supplied evidence", provider.last_request.system_prompt)
        self.assertIn("Do not use outside knowledge", provider.last_request.system_prompt)

    def test_contradictory_evidence_is_preserved_and_abstained(self):
        provider = RecordingMockProvider(
            output("The evidence conflicts; I abstain.", ["synthetic-a", "synthetic-b"], abstain=True)
        )
        result = self._engine(provider).answer(
            self.question, question_external_llm_approved=True
        )

        self.assertTrue(result.abstained)
        self.assertIn("abstain", provider.last_request.system_prompt.lower())
        self.assertEqual(len(result.citations), 2)

    def test_citation_metadata_is_resolved_from_authorized_chunk(self):
        provider = RecordingMockProvider(output("A synthetic answer.", ["synthetic-a"]))
        result = self._engine(provider).answer(
            self.question, question_external_llm_approved=True
        )
        self.assertEqual(result.citations[0].source_document, "architecture.md")
        self.assertEqual(result.citations[0].source_path, "docs/architecture.md")
        self.assertEqual(result.citations[0].section, "Alpha state")

    def test_unknown_evidence_status_is_preserved(self):
        unknown = record(
            "synthetic-unknown",
            "Synthetic baseline value is unknown.",
            evidence_status="UNKNOWN",
            operational_validity="NOT_ASSESSED",
        )
        self._write_jsonl(self.architecture_path, [unknown])
        self.storage_path.unlink()
        provider = RecordingMockProvider(
            output("The value is unknown in the supplied record.", ["synthetic-unknown"])
        )

        result = self._engine(provider).answer(
            "What is the synthetic baseline value?", question_external_llm_approved=True
        )

        self.assertEqual(result.citations[0].evidence_status, "UNKNOWN")
        self.assertEqual(result.citations[0].operational_validity, "NOT_ASSESSED")

    def test_invented_chunk_id_is_rejected_as_safe_abstention(self):
        provider = RecordingMockProvider(output("An answer.", ["invented-chunk-id"]))
        result = self._engine(provider).answer(
            self.question, question_external_llm_approved=True
        )
        self.assertTrue(result.abstained)
        self.assertEqual(result.answer, "I cannot answer reliably from the authorized evidence available.")
        self.assertEqual(result.reason, "invalid_citations")
        self.assertEqual(result.citations, ())

    def test_model_supplied_source_metadata_is_rejected(self):
        provider = RecordingMockProvider(
            output("An answer.", ["synthetic-a"], source_path="docs/invented.md", section="Invented")
        )
        result = self._engine(provider).answer(
            self.question, question_external_llm_approved=True
        )
        self.assertTrue(result.abstained)
        self.assertEqual(result.reason, "invalid_provider_response_schema")

    def test_inline_invented_path_or_chunk_citation_is_rejected(self):
        for answer in (
            "See docs/invented.md for details.",
            "See [invented-section].",
            "Source chunk 0123456789abcdef0123.",
        ):
            with self.subTest(answer=answer):
                provider = RecordingMockProvider(output(answer, ["synthetic-a"]))
                result = self._engine(provider).answer(
                    self.question, question_external_llm_approved=True
                )
                self.assertTrue(result.abstained)
                self.assertEqual(result.reason, "invalid_citations")

    def test_question_authorization_defaults_to_false_and_skips_provider(self):
        provider, fake_client = fake_anthropic_provider(
            output("Should not be called.", ["synthetic-a"])
        )
        result = self._engine(provider).answer(self.question)

        self.assertTrue(result.abstained)
        self.assertEqual(result.reason, "question_external_processing_not_approved")
        fake_client.messages.create.assert_not_called()

    def test_no_authorized_chunk_skips_provider_and_unapproved_content(self):
        provider, fake_client = fake_anthropic_provider(
            output("Should not be called.", ["synthetic-a"])
        )
        unapproved = record(
            "synthetic-unapproved",
            "Synthetic private marker: DO_NOT_SEND_UNAPPROVED.",
            external_approved=False,
        )
        self._write_jsonl(self.architecture_path, [unapproved])
        self.storage_path.unlink()
        result = self._engine(provider).answer(
            self.question, question_external_llm_approved=True
        )

        self.assertTrue(result.abstained)
        self.assertEqual(result.reason, "no_externally_authorized_evidence")
        fake_client.messages.create.assert_not_called()

    def test_invalid_provider_response_type_fails_closed(self):
        class InvalidProvider:
            def generate(self, request):
                return "not an LLMResponse"

        result = self._engine(InvalidProvider()).answer(
            self.question, question_external_llm_approved=True
        )
        self.assertTrue(result.abstained)
        self.assertEqual(result.reason, "invalid_provider_response")

    def test_any_unapproved_retrieved_chunk_blocks_provider_call(self):
        unapproved = record(
            "synthetic-unapproved",
            "Synthetic private marker: DO_NOT_SEND_UNAPPROVED.",
            external_approved=False,
        )
        self._write_jsonl(
            self.architecture_path,
            [*self.architecture_records, unapproved],
        )
        provider, fake_client = fake_anthropic_provider(
            output("Enabled.", ["synthetic-a"])
        )
        result = self._engine(provider).answer(
            self.question, question_external_llm_approved=True
        )

        self.assertTrue(result.abstained)
        self.assertEqual(result.reason, "no_externally_authorized_evidence")
        fake_client.messages.create.assert_not_called()

    def test_mock_answers_from_internal_chunk_without_external_approval(self):
        provider = RecordingMockProvider(output("The record says enabled.", ["synthetic-a"]))
        self._write_jsonl(
            self.architecture_path,
            [
                record(
                    "synthetic-a",
                    "Synthetic appliance alpha is enabled.",
                    external_approved=False,
                    rag_eligibility="ELIGIBLE",
                )
            ],
        )
        self.storage_path.unlink()

        result = self._engine(provider).answer(self.question)

        self.assertFalse(result.abstained)
        self.assertEqual(result.answer, "The record says enabled.")
        self.assertEqual(result.citations[0].chunk_id, "synthetic-a")
        self.assertEqual(result.citations[0].evidence_status, "VERIFIED")
        self.assertIn('"rag_eligibility": "ELIGIBLE"', provider.last_request.user_prompt)
        self.assertEqual(provider.calls, 1)

    def test_anthropic_rejects_same_internal_chunk_without_external_approval(self):
        provider, fake_client = fake_anthropic_provider(
            output("Should not be called.", ["synthetic-a"])
        )
        self._write_jsonl(
            self.architecture_path,
            [
                record(
                    "synthetic-a",
                    "Synthetic appliance alpha is enabled.",
                    external_approved=False,
                    rag_eligibility="ELIGIBLE",
                )
            ],
        )
        self.storage_path.unlink()

        result = self._engine(provider).answer(
            self.question, question_external_llm_approved=True
        )

        self.assertTrue(result.abstained)
        self.assertEqual(result.reason, "no_externally_authorized_evidence")
        fake_client.messages.create.assert_not_called()

    def test_mock_blocks_unclassified_pending_excluded_and_unknown_eligibility(self):
        cases = (
            ("None", None),
            ("ELIGIBLE_AFTER_REVIEW", "ELIGIBLE_AFTER_REVIEW"),
            ("EXCLUDED", "EXCLUDED"),
            ("unknown value", "UNRECOGNIZED"),
        )
        for label, eligibility in cases:
            with self.subTest(eligibility=label):
                item = record(
                    "synthetic-status",
                    "Synthetic appliance alpha is enabled.",
                    external_approved=False,
                    rag_eligibility=eligibility,
                )
                self._write_jsonl(self.architecture_path, [item])
                self.storage_path.unlink(missing_ok=True)
                provider = RecordingMockProvider(
                    output("Should not be called.", ["synthetic-status"])
                )

                result = self._engine(provider).answer(self.question)

                self.assertTrue(result.abstained)
                self.assertEqual(result.reason, "no_externally_authorized_evidence")
                self.assertEqual(provider.calls, 0)

        missing = record(
            "synthetic-missing-eligibility",
            "Synthetic appliance alpha is enabled.",
            external_approved=False,
        )
        missing.pop("rag_eligibility")
        self._write_jsonl(self.architecture_path, [missing])
        provider = RecordingMockProvider(
            output("Should not be called.", ["synthetic-missing-eligibility"])
        )

        result = self._engine(provider).answer(self.question)

        self.assertTrue(result.abstained)
        self.assertEqual(result.reason, "no_externally_authorized_evidence")
        self.assertEqual(provider.calls, 0)

    def test_mock_provider_implements_provider_contract(self):
        provider = MockLLMProvider(output("Synthetic answer.", ["synthetic-a"]))
        self.assertIsInstance(provider, LLMProvider)
        result = self._engine(provider).answer(
            self.question, question_external_llm_approved=True
        )
        self.assertFalse(result.abstained)

    def test_anthropic_provider_compatibility_uses_mock_client_only(self):
        fake_client = Mock()
        fake_client.messages.create.return_value = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="text",
                    text=output("Synthetic Claude-compatible answer.", ["synthetic-a"]),
                )
            ]
        )
        with patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "synthetic-key", "ANTHROPIC_MODEL": "synthetic-model"},
        ), patch("app.llm.anthropic.AnthropicClient", return_value=fake_client):
            provider = AnthropicLLMProvider()
            self.assertIsInstance(provider, LLMProvider)
            result = self._engine(provider).answer(
                self.question, question_external_llm_approved=True
            )

        self.assertFalse(result.abstained)
        self.assertEqual(result.answer, "Synthetic Claude-compatible answer.")
        fake_client.messages.create.assert_called_once()

    def test_non_approved_confidentiality_is_never_sent(self):
        restricted = record(
            "synthetic-restricted",
            "Synthetic restricted evidence.",
            confidentiality="RESTRICTED",
        )
        self._write_jsonl(self.architecture_path, [restricted])
        self.storage_path.unlink()
        provider = RecordingMockProvider(output("Should not be called.", ["synthetic-restricted"]))

        result = self._engine(provider).answer(
            "Synthetic restricted evidence?", question_external_llm_approved=True
        )

        self.assertTrue(result.abstained)
        self.assertEqual(provider.calls, 0)

    def test_rag_excluded_record_is_never_sent_even_if_external_approved(self):
        excluded = record(
            "synthetic-excluded",
            "Synthetic excluded knowledge marker.",
            external_approved=True,
            rag_eligibility="EXCLUDED",
        )
        self._write_jsonl(self.architecture_path, [excluded])
        self.storage_path.unlink()
        provider = RecordingMockProvider(output("Should not be called.", ["synthetic-excluded"]))

        result = self._engine(provider).answer(
            "Synthetic excluded knowledge marker?", question_external_llm_approved=True
        )

        self.assertTrue(result.abstained)
        self.assertEqual(result.reason, "no_externally_authorized_evidence")
        self.assertEqual(provider.calls, 0)


    def test_formal_admission_allows_matching_synthetic_chunk_for_mock(self):
        provider = RecordingMockProvider(output("Formal synthetic answer.", ["synthetic-a"]))
        result = self._engine(provider).answer(self.question)
        self.assertFalse(result.abstained)
        self.assertEqual(result.answer, "Formal synthetic answer.")
        self.assertEqual(provider.calls, 1)

    def test_missing_catalog_record_blocks_eligible_chunk_before_provider(self):
        self._write_catalog(include_sources={"docs/storage.md"})
        provider = RecordingMockProvider(output("Should not run.", ["synthetic-a"]))
        result = self._engine(provider).answer(self.question)
        self.assertTrue(result.abstained)
        self.assertEqual(result.reason, "formal_rag_admission_missing_or_invalid")
        self.assertEqual(provider.calls, 0)

    def test_pending_and_revoked_admissions_block_before_provider(self):
        for status, eligibility in (("PENDING", "ELIGIBLE_AFTER_REVIEW"), ("REVOKED", "EXCLUDED")):
            with self.subTest(status=status):
                self._write_catalog(admission_status=status, eligibility=eligibility)
                provider = RecordingMockProvider(output("Should not run.", ["synthetic-a"]))
                result = self._engine(provider).answer(self.question)
                self.assertTrue(result.abstained)
                self.assertEqual(provider.calls, 0)

    def test_wrong_corpus_or_audience_blocks_before_provider(self):
        for kwargs in (
            {"target_corpus": "different synthetic corpus"},
            {"audience": "different synthetic audience"},
        ):
            with self.subTest(kwargs=kwargs):
                self._write_catalog(**kwargs)
                provider = RecordingMockProvider(output("Should not run.", ["synthetic-a"]))
                result = self._engine(provider).answer(self.question)
                self.assertTrue(result.abstained)
                self.assertEqual(provider.calls, 0)

    def test_changed_source_after_review_blocks_before_provider(self):
        (self.root / "docs" / "architecture.md").write_text(
            "# Changed synthetic document\n", encoding="utf-8"
        )
        provider = RecordingMockProvider(output("Should not run.", ["synthetic-a"]))
        result = self._engine(provider).answer(self.question)
        self.assertTrue(result.abstained)
        self.assertEqual(result.reason, "formal_rag_admission_missing_or_invalid")
        self.assertEqual(provider.calls, 0)

    def test_historical_chunk_without_admission_digest_is_blocked(self):
        legacy = self.architecture_records[0].copy()
        legacy.pop("document_sha256")
        self._write_jsonl(self.architecture_path, [legacy])
        self.storage_path.unlink()
        provider = RecordingMockProvider(output("Should not run.", ["synthetic-a"]))
        result = self._engine(provider).answer(self.question)
        self.assertTrue(result.abstained)
        self.assertEqual(provider.calls, 0)

    def test_chunk_provenance_mismatch_with_catalog_is_blocked(self):
        changed = record("synthetic-mismatch", "Synthetic appliance alpha mismatched.")
        changed["document_sha256"] = "0" * 64
        self._write_jsonl(self.architecture_path, [changed])
        self.storage_path.unlink()
        provider = RecordingMockProvider(output("Should not run.", ["synthetic-mismatch"]))
        result = self._engine(provider).answer(self.question)
        self.assertTrue(result.abstained)
        self.assertEqual(provider.calls, 0)

    def test_external_chunk_authorization_and_question_gate_remain_independent(self):
        unapproved = record(
            "synthetic-external-pending",
            "Synthetic appliance alpha is enabled.",
            external_approved=False,
        )
        self._write_jsonl(self.architecture_path, [unapproved])
        self.storage_path.unlink()
        provider, fake_client = fake_anthropic_provider(
            output("Should not run.", ["synthetic-external-pending"])
        )
        result = self._engine(provider).answer(
            self.question, question_external_llm_approved=True
        )
        self.assertTrue(result.abstained)
        fake_client.messages.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
