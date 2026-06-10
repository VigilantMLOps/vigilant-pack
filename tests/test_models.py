import json
import pytest
import httpx
from unittest.mock import MagicMock, patch

from vigilantpack.models import list_present, pull, warmup, _classify_warmup_failure, _strip_latest
from vigilantpack.services import StageResult


def _stream_ctx(status_code, lines):
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=ctx)
    ctx.__exit__ = MagicMock(return_value=False)
    ctx.status_code = status_code
    ctx.iter_lines = MagicMock(return_value=iter(lines))
    ctx.read = MagicMock(return_value=b"error body")
    return ctx


class TestStripLatest:
    def test_strips_latest_tag(self):
        assert _strip_latest("nomic-embed-text:latest") == "nomic-embed-text"

    def test_leaves_non_latest_tag(self):
        assert _strip_latest("llama3.2:3b") == "llama3.2:3b"

    def test_leaves_bare_name(self):
        assert _strip_latest("nomic-embed-text") == "nomic-embed-text"


class TestListPresent:
    def test_returns_model_names(self):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"models": [{"name": "llama3.2"}, {"name": "nomic-embed-text"}]}
        with patch("httpx.get", return_value=resp):
            result = list_present("http://localhost:11434")
        assert result == {"llama3.2", "nomic-embed-text"}

    def test_strips_latest_suffix(self):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"models": [
            {"name": "nomic-embed-text:latest"},
            {"name": "llama3.2:3b"},
        ]}
        with patch("httpx.get", return_value=resp):
            result = list_present("http://localhost:11434")
        assert result == {"nomic-embed-text", "llama3.2:3b"}

    def test_empty_list_when_no_models(self):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"models": []}
        with patch("httpx.get", return_value=resp):
            assert list_present("http://localhost:11434") == set()

    def test_returns_empty_set_on_connect_error(self):
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            assert list_present("http://localhost:11434") == set()

    def test_returns_empty_set_on_any_exception(self):
        with patch("httpx.get", side_effect=RuntimeError("unexpected")):
            assert list_present("http://localhost:11434") == set()


class TestPull:
    def test_success_via_status_line(self):
        lines = [
            json.dumps({"status": "downloading", "total": 100, "completed": 100}),
            json.dumps({"status": "success"}),
        ]
        with patch("httpx.stream", return_value=_stream_ctx(200, lines)):
            result = pull("llama3.2", "http://localhost:11434")
        assert result.success

    def test_success_when_stream_ends_without_status_line(self):
        lines = [json.dumps({"status": "downloading", "total": 100, "completed": 100})]
        with patch("httpx.stream", return_value=_stream_ctx(200, lines)):
            result = pull("llama3.2", "http://localhost:11434")
        assert result.success

    def test_model_not_found_is_deterministic(self):
        lines = [json.dumps({"error": "model not found"})]
        with patch("httpx.stream", return_value=_stream_ctx(200, lines)):
            result = pull("badmodel", "http://localhost:11434")
        assert not result.success
        assert result.error_type == "deterministic"

    def test_generic_api_error(self):
        lines = [json.dumps({"error": "server is overloaded"})]
        with patch("httpx.stream", return_value=_stream_ctx(200, lines)):
            result = pull("llama3.2", "http://localhost:11434")
        assert not result.success
        assert result.error_type == "unknown"

    def test_non_200_response(self):
        with patch("httpx.stream", return_value=_stream_ctx(500, [])):
            result = pull("llama3.2", "http://localhost:11434")
        assert not result.success
        assert "500" in result.message

    def test_connect_error_is_transient(self):
        with patch("httpx.stream", side_effect=httpx.ConnectError("refused")):
            result = pull("llama3.2", "http://localhost:11434")
        assert not result.success
        assert result.error_type == "transient"
        assert result.recoverable

    def test_progress_callback_called_for_each_progress_line(self):
        # pull() calls progress_cb on every line with a total; pct filtering is the caller's job
        calls = []
        lines = [
            json.dumps({"status": "pulling", "total": 100, "completed": 10}),
            json.dumps({"status": "pulling", "total": 100, "completed": 15}),
            json.dumps({"status": "pulling", "total": 100, "completed": 20}),
            json.dumps({"status": "success"}),
        ]
        with patch("httpx.stream", return_value=_stream_ctx(200, lines)):
            pull("llama3.2", "http://localhost:11434", progress_cb=lambda s, p: calls.append(p))
        assert calls == [10, 15, 20]

    def test_malformed_json_lines_skipped(self):
        lines = ["not json at all", json.dumps({"status": "success"})]
        with patch("httpx.stream", return_value=_stream_ctx(200, lines)):
            result = pull("llama3.2", "http://localhost:11434")
        assert result.success


class TestWarmup:
    def test_embed_success(self):
        with patch("httpx.post", return_value=MagicMock(status_code=200)):
            result = warmup("nomic-embed-text", "http://localhost:11434", required=True)
        assert result.success

    def test_falls_back_to_generate_when_not_embed_model(self):
        embed_resp = MagicMock(status_code=400)
        embed_resp.json.return_value = {"error": "not an embedding model"}
        generate_resp = MagicMock(status_code=200)
        with patch("httpx.post", side_effect=[embed_resp, generate_resp]):
            result = warmup("llama3.2", "http://localhost:11434", required=True)
        assert result.success

    def test_timeout_always_gives_warn_severity(self):
        with patch("httpx.post", side_effect=httpx.TimeoutException("timed out")):
            result = warmup("llama3.2", "http://localhost:11434", required=True)
        assert not result.success
        assert result.suggested_severity == "warn"

    def test_required_api_error_gives_soft_severity(self):
        embed_resp = MagicMock(status_code=400)
        embed_resp.json.return_value = {"error": "something generic"}
        with patch("httpx.post", return_value=embed_resp):
            result = warmup("llama3.2", "http://localhost:11434", required=True)
        assert not result.success
        assert result.suggested_severity == "soft"

    def test_optional_api_error_gives_warn_severity(self):
        embed_resp = MagicMock(status_code=400)
        embed_resp.json.return_value = {"error": "something generic"}
        with patch("httpx.post", return_value=embed_resp):
            result = warmup("llama3.2", "http://localhost:11434", required=False)
        assert not result.success
        assert result.suggested_severity == "warn"

    def test_ollama_unreachable_is_classified_as_warn(self):
        # ConnectError in _warmup_embed → transient → _classify_warmup_failure → warn
        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            result = warmup("llama3.2", "http://localhost:11434", required=True)
        assert not result.success
        assert result.suggested_severity == "warn"


class TestClassifyWarmupFailure:
    def test_transient_always_produces_warn(self):
        r = StageResult(False, "hard", "timed out", True, "transient")
        out = _classify_warmup_failure(r, required=True)
        assert out.suggested_severity == "warn"

    def test_required_non_transient_produces_soft(self):
        r = StageResult(False, "hard", "api error", True, "unknown")
        out = _classify_warmup_failure(r, required=True)
        assert out.suggested_severity == "soft"

    def test_optional_non_transient_produces_warn(self):
        r = StageResult(False, "hard", "api error", True, "unknown")
        out = _classify_warmup_failure(r, required=False)
        assert out.suggested_severity == "warn"

    def test_message_is_preserved(self):
        r = StageResult(False, "hard", "original message", True, "unknown")
        out = _classify_warmup_failure(r, required=True)
        assert out.message == "original message"
