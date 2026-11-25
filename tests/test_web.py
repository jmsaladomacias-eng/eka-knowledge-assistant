import http.client
import hashlib
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path

from app.answer_engine import AnswerEngine
from app.corpus import ADMISSION_FIELDS, RAG_AUTHORIZED_AUDIENCE, RAG_TARGET_CORPUS
from app.llm.mock import MockLLMProvider
from app.web import HOST, MAX_QUESTION_CHARS, make_server


SYNTHETIC_SOURCE = "# Synthetic status\n\nSynthetic web fixture states that the sample system is ready.\n"
SYNTHETIC_DIGEST = hashlib.sha256(SYNTHETIC_SOURCE.encode("utf-8")).hexdigest()


def synthetic_record(*, approved=True):
    return {
        "chunk_id": "synthetic-web-1",
        "content": "Synthetic web fixture states that the sample system is ready.",
        "source_document": "architecture.md",
        "source_path": "docs/architecture.md",
        "section": "Synthetic status",
        "evidence_status": "VERIFIED",
        "operational_validity": "NOT_ASSESSED",
        "confidentiality": "INTERNAL",
        "source_date": "UNKNOWN",
        "review_date": "NOT_REVIEWED",
        "external_llm_approved": approved,
        "rag_eligibility": "ELIGIBLE",
        "document_sha256": SYNTHETIC_DIGEST,
    }


def mock_response(*, answer="Synthetic test answer.", citations=None, abstain=False):
    return json.dumps(
        {"answer": answer, "citations": citations or [], "abstain": abstain}
    )


class CountingMockProvider(MockLLMProvider):
    def __init__(self, response_text):
        super().__init__(response_text)
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return super().generate(request)


class WebInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        source = self.root / "docs" / "architecture.md"
        source.parent.mkdir(parents=True)
        source.write_text(SYNTHETIC_SOURCE, encoding="utf-8")
        admission = {
            "source_path": "docs/architecture.md",
            "admission_status": "APPROVED",
            "rag_eligibility": "ELIGIBLE",
            "confidentiality": "INTERNAL",
            "evidence_status": "VERIFIED",
            "operational_validity": "NOT_ASSESSED",
            "owner": "Synthetic owner",
            "reviewed_at": "2026-11-23",
            "reviewer_role": "Synthetic reviewer",
            "review_scope": "Synthetic test scope",
            "target_corpus": RAG_TARGET_CORPUS,
            "authorized_audience": RAG_AUTHORIZED_AUDIENCE,
            "reviewed_sha256": SYNTHETIC_DIGEST,
        }
        header = "| " + " | ".join(ADMISSION_FIELDS) + " |"
        separator = "|" + "|".join("---" for _ in ADMISSION_FIELDS) + "|"
        row = "| " + " | ".join(admission[field] for field in ADMISSION_FIELDS) + " |"
        (self.root / "docs" / "knowledge-catalog.md").write_text(
            "# Synthetic catalog\n\n## Repository RAG admission decisions\n\n"
            + "\n".join((header, separator, row, "")),
            encoding="utf-8",
        )
        corpus_dir = self.root / "data" / "corpus"
        corpus_dir.mkdir(parents=True)
        self.corpus_file = corpus_dir / "architecture.jsonl"
        self._write_record(synthetic_record())
        self.provider = CountingMockProvider(
            mock_response(citations=["synthetic-web-1"])
        )
        self.engine = AnswerEngine(self.root, self.provider)
        self._start_server(self.engine)

    def tearDown(self):
        self._stop_server()
        self.temp.cleanup()

    def _write_record(self, item):
        self.corpus_file.write_text(json.dumps(item) + "\n", encoding="utf-8")

    def _start_server(self, engine, *, provider_status="Synthetic Mock Provider", external_provider=False):
        self.server = make_server(
            0,
            engine=engine,
            provider_status=provider_status,
            external_provider=external_provider,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def _stop_server(self):
        if getattr(self, "server", None) is not None:
            self.server.shutdown()
            self.thread.join(timeout=3)
            self.server.server_close()
            self.server = None

    def _request(self, method="GET", body=None, headers=None, path="/"):
        connection = http.client.HTTPConnection(HOST, self.server.server_port, timeout=3)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        status = response.status
        content = response.read().decode("utf-8")
        connection.close()
        return status, content

    def _post(self, question, *, approved=False):
        from urllib.parse import urlencode

        values = {"question": question}
        if approved:
            values["question_external_llm_approved"] = "yes"
        return self._request(
            "POST",
            urlencode(values),
            {"Content-Type": "application/x-www-form-urlencoded"},
        )

    def test_get_returns_question_form(self):
        status, page = self._request()
        self.assertEqual(status, 200)
        self.assertIn("Engineering Knowledge &amp; Operations Assistant", page)
        self.assertIn('<textarea id="question" name="question"', page)
        self.assertIn('type="checkbox"', page)
        self.assertIn('type="submit"', page)

    def test_post_valid_question_displays_answer(self):
        status, page = self._post("What does the synthetic web fixture state?", approved=True)
        self.assertEqual(status, 200)
        self.assertIn("Synthetic test answer.", page)
        self.assertIn("Answer returned", page)
        self.assertEqual(self.provider.calls, 1)

    def test_empty_question_is_rejected(self):
        status, page = self._post("   ", approved=True)
        self.assertEqual(status, 400)
        self.assertIn("Enter a question.", page)
        self.assertEqual(self.provider.calls, 0)

    def test_excessively_long_question_is_rejected(self):
        status, page = self._post("x" * (MAX_QUESTION_CHARS + 1), approved=True)
        self.assertEqual(status, 413)
        self.assertIn("character limit", page)
        self.assertEqual(self.provider.calls, 0)

    def test_question_and_generated_html_are_escaped(self):
        self.provider = CountingMockProvider(
            mock_response(
                answer='<script>alert("x")</script>',
                citations=["synthetic-web-1"],
            )
        )
        self.engine = AnswerEngine(self.root, self.provider)
        self._stop_server()
        self._start_server(self.engine)

        question = 'Synthetic fixture <img src=x onerror="alert(1)">'
        status, page = self._post(question, approved=True)

        self.assertEqual(status, 200)
        self.assertIn('&lt;img src=x onerror=&quot;alert(1)&quot;&gt;', page)
        self.assertIn('&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;', page)
        self.assertNotIn('<script>alert("x")</script>', page)

    def test_citations_and_metadata_are_rendered(self):
        status, page = self._post("What does the synthetic web fixture state?", approved=True)
        self.assertEqual(status, 200)
        for value in (
            "architecture.md",
            "Synthetic status",
            "VERIFIED",
            "NOT_ASSESSED",
            "UNKNOWN",
            "NOT_REVIEWED",
        ):
            self.assertIn(value, page)

    def test_abstention_and_reason_are_rendered(self):
        abstaining_provider = CountingMockProvider(
            mock_response(answer="Synthetic evidence is insufficient.", abstain=True)
        )
        engine = AnswerEngine(self.root, abstaining_provider)
        self._stop_server()
        self._start_server(engine)

        status, page = self._post("What does the synthetic web fixture state?", approved=True)

        self.assertEqual(status, 200)
        self.assertIn("Abstained", page)
        self.assertIn("insufficient_or_conflicting_evidence", page)
        self.assertIn("Synthetic evidence is insufficient.", page)

    def test_authorization_checkbox_defaults_to_false(self):
        status, page = self._request()
        self.assertEqual(status, 200)
        checkbox = page.split('name="question_external_llm_approved"', 1)[1].split(">", 1)[0]
        self.assertNotIn("checked", checkbox)

        status, page = self._post("What does the synthetic web fixture state?")
        self.assertEqual(status, 200)
        self.assertIn("Synthetic test answer.", page)
        self.assertIn("No external request: question authorization was not granted.", page)
        self.assertEqual(self.provider.calls, 1)

    def test_checkbox_cannot_bypass_chunk_authorization(self):
        restricted = synthetic_record()
        restricted["confidentiality"] = "RESTRICTED"
        self._write_record(restricted)

        status, page = self._post("What does the synthetic web fixture state?", approved=True)

        self.assertEqual(status, 200)
        self.assertIn("no_externally_authorized_evidence", page)
        self.assertIn("Answer Engine authorization gates blocked it", page)
        self.assertEqual(self.provider.calls, 0)

    def test_server_binds_exclusively_to_loopback(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_server_handles_second_request_while_first_connection_is_stalled(self):
        first_connection_accepted = threading.Event()
        original_get_request = self.server.get_request
        accept_lock = threading.Lock()
        accepted_count = 0

        def signal_first_accept():
            nonlocal accepted_count
            request, address = original_get_request()
            with accept_lock:
                accepted_count += 1
                if accepted_count == 1:
                    first_connection_accepted.set()
            return request, address

        self.server.get_request = signal_first_accept
        stalled_client = socket.create_connection(
            (HOST, self.server.server_port), timeout=3
        )
        try:
            # Send no request bytes so the first handler waits for a request line
            # while the client connection remains open.
            self.assertTrue(first_connection_accepted.wait(timeout=3))

            status, page = self._request()

            self.assertEqual(status, 200)
            self.assertIn("Engineering Knowledge &amp; Operations Assistant", page)
        finally:
            stalled_client.close()

    def test_unavailable_provider_is_reported_without_startup_call(self):
        self._stop_server()
        self._start_server(
            None,
            provider_status="External provider unavailable; no provider request will be made.",
            external_provider=True,
        )
        status, page = self._post("A synthetic question", approved=True)
        self.assertEqual(status, 503)
        self.assertIn("No provider is available", page)
        self.assertIn("no provider request will be made", page)


if __name__ == "__main__":
    unittest.main()
