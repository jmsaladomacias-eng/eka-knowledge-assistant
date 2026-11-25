"""Minimal loopback-only web interface for the Engineering Knowledge Assistant."""

import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html
import json
import os
from pathlib import Path
from urllib.parse import parse_qs

from app.answer_engine import AnswerEngine, AnswerResult
from app.llm.mock import MockLLMProvider


HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_QUESTION_CHARS = 2000
MAX_REQUEST_BYTES = 16 * 1024
_ROOT = Path(__file__).resolve().parent.parent
_MOCK_RESPONSE = json.dumps(
    {
        "answer": "Mock Provider is active. Its fixed output is for interface testing, not a factual answer.",
        "citations": [],
        "abstain": True,
    }
)
_NO_PROVIDER_CALL_REASONS = {
    "question_external_processing_not_approved",
    "no_externally_authorized_evidence",
    "authorized_evidence_metadata_incomplete",
}


def _build_runtime() -> tuple[AnswerEngine | None, str, bool]:
    """Select a local Mock by default or Anthropic only by explicit configuration."""
    mode = os.environ.get("EKA_LLM_PROVIDER", "mock").strip().lower()
    if mode == "mock":
        return (
            AnswerEngine(_ROOT, MockLLMProvider(_MOCK_RESPONSE)),
            "Mock mode: fixed local test output; it is not a factual answer.",
            False,
        )

    if mode == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip() or not os.environ.get(
            "ANTHROPIC_MODEL", ""
        ).strip():
            return None, "External provider unavailable; no provider request will be made.", True
        try:
            # Import and construct only after explicit mode selection. Construction does not call the API.
            from app.llm.anthropic import AnthropicLLMProvider

            return (
                AnswerEngine(_ROOT, AnthropicLLMProvider()),
                "External provider configured; question and chunk authorization are still required.",
                True,
            )
        except Exception:
            return None, "External provider unavailable; no provider request will be made.", True

    return None, "No supported provider is configured; no provider request will be made.", False


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _citation_html(result: AnswerResult) -> str:
    if not result.citations:
        return "<p>No supporting citations were returned.</p>"
    entries = []
    for citation in result.citations:
        fields = (
            ("Document", citation.source_document),
            ("Section", citation.section),
            ("Evidence status", citation.evidence_status),
            ("Operational validity", citation.operational_validity),
            ("Source date", citation.source_date),
            ("Review date", citation.review_date),
        )
        values = "".join(
            f"<dt>{_escape(label)}</dt><dd>{_escape(value)}</dd>"
            for label, value in fields
        )
        entries.append(f"<article class=\"citation\"><dl>{values}</dl></article>")
    return "".join(entries)


def _page(
    provider_status: str,
    *,
    question: str = "",
    authorization_checked: bool = False,
    result: AnswerResult | None = None,
    message: str = "",
    external_status: str = "",
) -> str:
    checked = " checked" if authorization_checked else ""
    error_block = f'<p class="message" role="alert">{_escape(message)}</p>' if message else ""
    result_block = ""
    if result is not None:
        result_block = (
            '<section class="result" aria-live="polite">'
            "<h2>Result</h2>"
            f"<p><strong>Status:</strong> {'Abstained' if result.abstained else 'Answer returned'}</p>"
            + (f"<p><strong>Reason:</strong> {_escape(result.reason)}</p>" if result.reason else "")
            + f"<div class=\"answer\">{_escape(result.answer)}</div>"
            "<h3>Supporting citations</h3>"
            f"{_citation_html(result)}"
            f"<p><strong>External-processing status:</strong> {_escape(external_status)}</p>"
            "</section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Engineering Knowledge &amp; Operations Assistant</title>
  <style>
    body {{ max-width: 52rem; margin: 2rem auto; padding: 0 1rem; font: 1rem/1.5 sans-serif; color: #18212b; }}
    textarea {{ box-sizing: border-box; width: 100%; min-height: 9rem; }}
    button {{ margin-top: 1rem; padding: .55rem 1rem; }}
    .notice, .result, .message {{ margin: 1.25rem 0; padding: 1rem; border: 1px solid #aab6c2; border-radius: .35rem; }}
    .answer {{ white-space: pre-wrap; }}
    .citation {{ border-top: 1px solid #ccd3da; }}
    dt {{ font-weight: bold; }}
    dd {{ margin: 0 0 .5rem; }}
  </style>
</head>
<body>
  <header>
    <h1>Engineering Knowledge &amp; Operations Assistant</h1>
    <p>Local, read-only interface for questions grounded in approved evidence.</p>
    <p class="notice">{_escape(provider_status)}</p>
  </header>
  <main>
    {error_block}
    <form method="post" action="/">
      <label for="question"><strong>Technical question</strong></label>
      <textarea id="question" name="question" maxlength="{MAX_QUESTION_CHARS}" required>{_escape(question)}</textarea>
      <div>
        <input id="question_external_llm_approved" name="question_external_llm_approved" type="checkbox" value="yes"{checked}>
        <label for="question_external_llm_approved">I authorize this question and any separately approved evidence to be processed by the configured external provider.</label>
      </div>
      <p>The Answer Engine still blocks provider calls unless every retrieved chunk passes its external authorization and confidentiality checks. This checkbox does not approve corpus content.</p>
      <button type="submit">Submit question</button>
    </form>
    <aside class="notice"><strong>Limitations:</strong> Answers may be incomplete. Historical evidence is not confirmation of current operation. Check citations, evidence status, operational validity, and dates.</aside>
    {result_block}
  </main>
</body>
</html>"""


def make_server(
    port: int = DEFAULT_PORT,
    *,
    engine: AnswerEngine | None = None,
    provider_status: str | None = None,
    external_provider: bool | None = None,
) -> ThreadingHTTPServer:
    """Create an HTTP server bound only to IPv4 loopback."""
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer from 0 through 65535")
    if provider_status is None:
        engine, provider_status, external_provider = _build_runtime()
    if external_provider is None:
        external_provider = False

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            # Avoid logging submitted content or request details.
            return

        def _send(self, status: int, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            if self.path != "/":
                self._send(404, _page(provider_status, message="Page not found."))
                return
            self._send(200, _page(provider_status))

        def do_POST(self) -> None:
            if self.path != "/":
                self._send(404, _page(provider_status, message="Page not found."))
                return
            if self.headers.get_content_type() != "application/x-www-form-urlencoded":
                self._send(415, _page(provider_status, message="Use the question form to submit."))
                return
            try:
                content_length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send(400, _page(provider_status, message="Invalid form submission."))
                return
            if content_length < 0:
                self._send(400, _page(provider_status, message="Invalid form submission."))
                return
            if content_length > MAX_REQUEST_BYTES:
                self._send(413, _page(provider_status, message="Form submission is too large."))
                return
            try:
                form = parse_qs(
                    self.rfile.read(content_length).decode("utf-8"),
                    keep_blank_values=True,
                    max_num_fields=4,
                )
            except (UnicodeDecodeError, ValueError):
                self._send(400, _page(provider_status, message="Invalid form submission."))
                return

            questions = form.get("question", [])
            if len(questions) != 1 or not questions[0].strip():
                self._send(400, _page(provider_status, message="Enter a question."))
                return
            question = questions[0]
            if len(question) > MAX_QUESTION_CHARS:
                self._send(
                    413,
                    _page(
                        provider_status,
                        question=question[:MAX_QUESTION_CHARS],
                        message=f"Question exceeds the {MAX_QUESTION_CHARS}-character limit.",
                    ),
                )
                return

            approved = form.get("question_external_llm_approved") == ["yes"]
            if engine is None:
                self._send(
                    503,
                    _page(
                        provider_status,
                        question=question,
                        authorization_checked=approved,
                        message="No provider is available; no answer was generated.",
                        external_status="No provider request was made.",
                    ),
                )
                return

            try:
                result = engine.answer(
                    question,
                    question_external_llm_approved=approved,
                )
            except Exception:
                # Do not expose provider exceptions, headers, keys, or request contents.
                status = (
                    "Mock Provider is local; no external API was used."
                    if not external_provider
                    else "External processing failed; a provider request may have been attempted."
                )
                self._send(
                    502,
                    _page(
                        provider_status,
                        question=question,
                        authorization_checked=approved,
                        message="The request could not be completed. Details are withheld.",
                        external_status=status,
                    ),
                )
                return

            if not approved:
                external_status = "No external request: question authorization was not granted."
            elif result.reason in _NO_PROVIDER_CALL_REASONS:
                external_status = (
                    "No external request: Answer Engine authorization gates blocked it. "
                    + ("Mock Provider was not called." if not external_provider else "")
                )
            elif not external_provider:
                external_status = "Mock Provider is local; no external API was used."
            else:
                external_status = "The configured external provider was invoked after the Answer Engine gates passed."

            self._send(
                200,
                _page(
                    provider_status,
                    question=question,
                    authorization_checked=approved,
                    result=result,
                    external_status=external_status,
                ),
            )

    return ThreadingHTTPServer((HOST, port), partial(Handler))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Engineering Knowledge Assistant")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    server = make_server(args.port)
    print(f"Assistant listening at http://{HOST}:{server.server_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
