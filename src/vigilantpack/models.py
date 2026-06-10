"""
models.py — Ollama model lifecycle.

Responsibilities: check presence, pull with progress, warmup.
Does NOT know about Docker, compose files, or stage ordering.

Warmup severity rules (from the finalized design):
  - Timeout during warmup           → always WARN (model is still loading async)
  - API error + model required      → SOFT FAIL (first query on this model path may fail)
  - API error + model not required  → WARN
"""
from __future__ import annotations

import json
from typing import Callable

import httpx

from .services import StageResult


# ── Presence ──────────────────────────────────────────────────────────────────

def list_present(ollama_url: str) -> set[str]:
    """Return model names known to Ollama with ':latest' stripped.

    Ollama tags untagged pulls as ':latest', so 'nomic-embed-text' and
    'nomic-embed-text:latest' are the same model.  Non-latest tags (e.g.
    'llama3.2:3b') are kept as-is.
    """
    try:
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        return {_strip_latest(m["name"]) for m in resp.json().get("models", [])}
    except Exception:
        return set()


def _strip_latest(name: str) -> str:
    return name[: -len(":latest")] if name.endswith(":latest") else name


# ── Pull ──────────────────────────────────────────────────────────────────────

def pull(
    name: str,
    ollama_url: str,
    progress_cb: Callable[[str, int], None] | None = None,
) -> StageResult:
    """
    Pull a model with streaming progress.
    progress_cb(status_str, pct_int) is called on each progress line.
    Retries are NOT done here — transient vs deterministic is signalled via error_type
    so the orchestrator can decide the retry policy.
    """
    try:
        with httpx.stream(
            "POST",
            f"{ollama_url}/api/pull",
            json={"name": name, "stream": True},
            timeout=None,     # global deadline is managed by orchestrator
        ) as resp:
            if resp.status_code != 200:
                body = resp.read().decode(errors="replace")
                return StageResult(
                    False, "hard",
                    f"{name}: pull request failed HTTP {resp.status_code} — {body}",
                    False, "unknown",
                )

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    line = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                if "error" in line:
                    err = line["error"]
                    etype = "deterministic" if "not found" in err else "unknown"
                    return StageResult(False, "hard", f"{name}: {err}", False, etype)

                status = line.get("status", "")

                if status == "success":
                    return StageResult(True, "hard", f"{name}: downloaded", True)

                if progress_cb and line.get("total"):
                    completed = line.get("completed", 0)
                    total = line["total"]
                    pct = int(completed / total * 100) if total else 0
                    progress_cb(status, pct)

        return StageResult(True, "hard", f"{name}: ready", True)

    except httpx.ConnectError:
        return StageResult(
            False, "hard", f"{name}: Ollama not reachable", True, "transient"
        )
    except httpx.StreamError as exc:
        return StageResult(
            False, "hard", f"{name}: stream interrupted — {exc}", True, "transient"
        )
    except Exception as exc:
        return StageResult(False, "hard", f"{name}: {exc}", False, "unknown")


# ── Warmup ────────────────────────────────────────────────────────────────────

def warmup(name: str, ollama_url: str, required: bool, timeout: int = 30) -> StageResult:
    """
    Pre-load model into memory with a single inference call.
    Tries /api/embeddings first (embed models); falls back to /api/generate (LLMs).

    Severity of failure depends on:
      - Timeout: always WARN (model is loading asynchronously)
      - API error + required: SOFT FAIL (first query path may fail)
      - API error + not required: WARN
    """
    result = _warmup_embed(name, ollama_url, timeout)

    if result.success:
        return result

    # _warmup_embed signals deterministic "not an embed model" via error_type
    if result.error_type == "deterministic":
        result = _warmup_generate(name, ollama_url, timeout)

    # Reclassify based on required flag and failure reason
    if not result.success:
        result = _classify_warmup_failure(result, required)

    return result


def _warmup_embed(name: str, ollama_url: str, timeout: int) -> StageResult:
    try:
        resp = httpx.post(
            f"{ollama_url}/api/embeddings",
            json={"model": name, "prompt": "warmup"},
            timeout=float(timeout),
        )
        if resp.status_code == 200:
            return StageResult(True, "warn", f"{name}: warm (embed)", True)

        err = resp.json().get("error", f"HTTP {resp.status_code}")
        if any(phrase in err for phrase in ("not an embedding", "does not support", "no embedding")):
            return StageResult(False, "warn", "not an embed model", True, "deterministic")
        return StageResult(False, "warn", f"embed error: {err}", True, "unknown")

    except httpx.TimeoutException:
        return StageResult(False, "warn", f"warmup timed out ({timeout}s)", True, "transient")
    except httpx.ConnectError:
        return StageResult(False, "hard", "Ollama unreachable", True, "transient")
    except Exception as exc:
        return StageResult(False, "warn", f"embed warmup: {exc}", True, "unknown")


def _warmup_generate(name: str, ollama_url: str, timeout: int) -> StageResult:
    try:
        resp = httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": name, "prompt": "hi", "stream": False},
            timeout=float(timeout),
        )
        if resp.status_code == 200:
            return StageResult(True, "warn", f"{name}: warm (generate)", True)
        err = resp.json().get("error", f"HTTP {resp.status_code}")
        return StageResult(False, "warn", f"generate error: {err}", True, "unknown")

    except httpx.TimeoutException:
        return StageResult(False, "warn", f"warmup timed out ({timeout}s)", True, "transient")
    except Exception as exc:
        return StageResult(False, "warn", f"generate warmup: {exc}", True, "unknown")


def _classify_warmup_failure(result: StageResult, required: bool) -> StageResult:
    """
    Apply the design's warmup failure semantics:
      timeout (transient)   → always WARN
      API error + required  → SOFT FAIL
      API error + optional  → WARN
    """
    if result.error_type == "transient":
        return StageResult(False, "warn", result.message, True, "transient")

    severity = "soft" if required else "warn"
    return StageResult(False, severity, result.message, True, result.error_type)
