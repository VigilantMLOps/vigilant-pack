import pytest
from unittest.mock import MagicMock, patch

from vigilantpack.manifest import ManifestError
from vigilantpack.orchestrator import StageEvent, run
from vigilantpack.preflight import CheckResult
from vigilantpack.services import StageResult


_MANIFEST = {
    "vigilantpack": "1",
    "app": {"name": "test-app", "version": "0.1.0"},
    "compose": "docker-compose.yml",
    "services": {"ollama": {"health": "http://localhost:11434/api/tags", "timeout": 30}},
    "models": [],
    "runtime": {"service": "app", "health": "http://localhost:8000/health", "timeout": 30},
    "startup": {"timeout": 300},
}

_PASS_CHECKS = [CheckResult("docker", "PASS", "running")]


def _mock_happy_path(manifest=None):
    m = manifest or _MANIFEST
    return [
        patch("vigilantpack.orchestrator.manifest_mod.load", return_value=m),
        patch("vigilantpack.orchestrator.preflight_mod.run_checks", return_value=_PASS_CHECKS),
        patch("vigilantpack.orchestrator.preflight_mod.has_hard_failure", return_value=False),
        patch("vigilantpack.orchestrator.services_mod.start",
              return_value=StageResult(True, "hard", "started", True)),
        patch("vigilantpack.orchestrator.services_mod.poll_health",
              return_value=StageResult(True, "hard", "healthy", True)),
        patch("vigilantpack.orchestrator.models_mod.list_present", return_value=set()),
    ]


def _apply_patches(patches):
    for p in patches:
        p.start()
    return patches


def _stop_patches(patches):
    for p in patches:
        p.stop()


class TestHappyPath:
    def test_emits_ready_event(self):
        events = []
        patches = _apply_patches(_mock_happy_path())
        try:
            run("vigilant.yaml", emit=events.append)
        finally:
            _stop_patches(patches)

        stages = [e.stage for e in events]
        assert "READY" in stages

    def test_ready_event_is_succeeded(self):
        events = []
        patches = _apply_patches(_mock_happy_path())
        try:
            run("vigilant.yaml", emit=events.append)
        finally:
            _stop_patches(patches)

        ready = next(e for e in events if e.stage == "READY")
        assert ready.event == "succeeded"

    def test_ready_metadata_has_app_info(self):
        events = []
        patches = _apply_patches(_mock_happy_path())
        try:
            run("vigilant.yaml", emit=events.append)
        finally:
            _stop_patches(patches)

        ready = next(e for e in events if e.stage == "READY")
        assert ready.metadata["app_name"] == "test-app"
        assert ready.metadata["app_version"] == "0.1.0"

    def test_all_six_stages_emitted(self):
        events = []
        patches = _apply_patches(_mock_happy_path())
        try:
            run("vigilant.yaml", emit=events.append)
        finally:
            _stop_patches(patches)

        stage_names = {e.stage for e in events}
        assert stage_names == {"VALIDATE", "PREFLIGHT", "INFRA", "MODELS", "RUNTIME", "READY"}

    def test_default_emit_is_noop(self):
        patches = _apply_patches(_mock_happy_path())
        try:
            run("vigilant.yaml")   # no emit= argument, should not raise
        finally:
            _stop_patches(patches)


class TestValidateStage:
    def test_manifest_load_error_emits_failed_and_exits(self):
        events = []
        with patch("vigilantpack.orchestrator.manifest_mod.load",
                   side_effect=ManifestError("bad manifest")):
            with pytest.raises(SystemExit) as exc_info:
                run("vigilant.yaml", emit=events.append)
        assert exc_info.value.code == 1
        assert any(e.stage == "VALIDATE" and e.event == "failed" for e in events)

    def test_validate_succeeded_event_contains_timestamp(self):
        events = []
        patches = _apply_patches(_mock_happy_path())
        try:
            run("vigilant.yaml", emit=events.append)
        finally:
            _stop_patches(patches)

        validated = next(e for e in events if e.stage == "VALIDATE" and e.event == "succeeded")
        assert validated.timestamp


class TestPreflightStage:
    def test_hard_failure_aborts(self):
        fail_checks = [CheckResult("docker", "FAIL", "not running", "start docker")]
        with patch("vigilantpack.orchestrator.manifest_mod.load", return_value=_MANIFEST):
            with patch("vigilantpack.orchestrator.preflight_mod.run_checks", return_value=fail_checks):
                with patch("vigilantpack.orchestrator.preflight_mod.has_hard_failure", return_value=True):
                    with pytest.raises(SystemExit) as exc_info:
                        run("vigilant.yaml", emit=lambda e: None)
        assert exc_info.value.code == 1

    def test_preflight_failed_event_includes_results(self):
        events = []
        fail_checks = [CheckResult("docker", "FAIL", "not running", "start docker")]
        with patch("vigilantpack.orchestrator.manifest_mod.load", return_value=_MANIFEST):
            with patch("vigilantpack.orchestrator.preflight_mod.run_checks", return_value=fail_checks):
                with patch("vigilantpack.orchestrator.preflight_mod.has_hard_failure", return_value=True):
                    with pytest.raises(SystemExit):
                        run("vigilant.yaml", emit=events.append)

        failed = next(e for e in events if e.stage == "PREFLIGHT" and e.event == "failed")
        assert "results" in failed.metadata


class TestInfraStage:
    def test_service_health_failure_aborts_after_retry(self):
        events = []
        patches = [
            patch("vigilantpack.orchestrator.manifest_mod.load", return_value=_MANIFEST),
            patch("vigilantpack.orchestrator.preflight_mod.run_checks", return_value=_PASS_CHECKS),
            patch("vigilantpack.orchestrator.preflight_mod.has_hard_failure", return_value=False),
            patch("vigilantpack.orchestrator.services_mod.start",
                  return_value=StageResult(True, "hard", "started", True)),
            patch("vigilantpack.orchestrator.services_mod.poll_health",
                  return_value=StageResult(False, "hard", "timeout", False)),
            patch("vigilantpack.orchestrator.services_mod.restart"),
            patch("vigilantpack.orchestrator.services_mod.tail_logs", return_value=""),
        ]
        _apply_patches(patches)
        try:
            with pytest.raises(SystemExit) as exc_info:
                run("vigilant.yaml", emit=events.append)
        finally:
            _stop_patches(patches)

        assert exc_info.value.code == 1
        assert any(e.stage == "INFRA" and e.event == "failed" for e in events)

    def test_deterministic_start_failure_aborts(self):
        patches = [
            patch("vigilantpack.orchestrator.manifest_mod.load", return_value=_MANIFEST),
            patch("vigilantpack.orchestrator.preflight_mod.run_checks", return_value=_PASS_CHECKS),
            patch("vigilantpack.orchestrator.preflight_mod.has_hard_failure", return_value=False),
            patch("vigilantpack.orchestrator.services_mod.start",
                  return_value=StageResult(False, "hard", "port conflict", False, "deterministic")),
        ]
        _apply_patches(patches)
        try:
            with pytest.raises(SystemExit) as exc_info:
                run("vigilant.yaml", emit=lambda e: None)
        finally:
            _stop_patches(patches)
        assert exc_info.value.code == 1


class TestModelsStage:
    def test_no_models_emits_succeeded(self):
        events = []
        manifest = {**_MANIFEST, "models": []}
        patches = _apply_patches(_mock_happy_path(manifest))
        try:
            run("vigilant.yaml", emit=events.append)
        finally:
            _stop_patches(patches)

        models_events = [e for e in events if e.stage == "MODELS"]
        assert any(e.event == "succeeded" for e in models_events)

    def test_required_model_pull_failure_aborts(self):
        manifest = {**_MANIFEST, "models": [{"name": "llama3.2", "required": True, "warmup": False}]}
        patches = [
            patch("vigilantpack.orchestrator.manifest_mod.load", return_value=manifest),
            patch("vigilantpack.orchestrator.preflight_mod.run_checks", return_value=_PASS_CHECKS),
            patch("vigilantpack.orchestrator.preflight_mod.has_hard_failure", return_value=False),
            patch("vigilantpack.orchestrator.services_mod.start",
                  return_value=StageResult(True, "hard", "started", True)),
            patch("vigilantpack.orchestrator.services_mod.poll_health",
                  return_value=StageResult(True, "hard", "healthy", True)),
            patch("vigilantpack.orchestrator.models_mod.list_present", return_value=set()),
            patch("vigilantpack.orchestrator.models_mod.pull",
                  return_value=StageResult(False, "hard", "not found", False, "deterministic")),
        ]
        _apply_patches(patches)
        try:
            with pytest.raises(SystemExit) as exc_info:
                run("vigilant.yaml", emit=lambda e: None)
        finally:
            _stop_patches(patches)
        assert exc_info.value.code == 1

    def test_optional_model_pull_failure_is_soft(self):
        events = []
        manifest = {**_MANIFEST, "models": [{"name": "opt-model", "required": False, "warmup": False}]}
        patches = [
            patch("vigilantpack.orchestrator.manifest_mod.load", return_value=manifest),
            patch("vigilantpack.orchestrator.preflight_mod.run_checks", return_value=_PASS_CHECKS),
            patch("vigilantpack.orchestrator.preflight_mod.has_hard_failure", return_value=False),
            patch("vigilantpack.orchestrator.services_mod.start",
                  return_value=StageResult(True, "hard", "started", True)),
            patch("vigilantpack.orchestrator.services_mod.poll_health",
                  return_value=StageResult(True, "hard", "healthy", True)),
            patch("vigilantpack.orchestrator.models_mod.list_present", return_value=set()),
            patch("vigilantpack.orchestrator.models_mod.pull",
                  return_value=StageResult(False, "hard", "network error", True, "transient")),
        ]
        _apply_patches(patches)
        try:
            run("vigilant.yaml", emit=events.append)
        finally:
            _stop_patches(patches)

        ready = next(e for e in events if e.stage == "READY")
        assert "opt-model" in ready.metadata["degraded_models"]

    def test_already_present_model_is_skipped(self):
        events = []
        manifest = {**_MANIFEST, "models": [{"name": "llama3.2", "required": True, "warmup": False}]}
        patches = [
            patch("vigilantpack.orchestrator.manifest_mod.load", return_value=manifest),
            patch("vigilantpack.orchestrator.preflight_mod.run_checks", return_value=_PASS_CHECKS),
            patch("vigilantpack.orchestrator.preflight_mod.has_hard_failure", return_value=False),
            patch("vigilantpack.orchestrator.services_mod.start",
                  return_value=StageResult(True, "hard", "started", True)),
            patch("vigilantpack.orchestrator.services_mod.poll_health",
                  return_value=StageResult(True, "hard", "healthy", True)),
            patch("vigilantpack.orchestrator.models_mod.list_present", return_value={"llama3.2"}),
        ]
        _apply_patches(patches)
        try:
            run("vigilant.yaml", emit=events.append)
        finally:
            _stop_patches(patches)

        skip_events = [
            e for e in events
            if e.stage == "MODELS" and e.metadata.get("skipped")
        ]
        assert len(skip_events) == 1


class TestRuntimeStage:
    def test_runtime_health_failure_aborts(self):
        # poll_health: first call for INFRA succeeds, next two for RUNTIME fail
        health_responses = [
            StageResult(True, "hard", "healthy", True),   # INFRA ollama
            StageResult(False, "hard", "timeout", False), # RUNTIME first attempt
            StageResult(False, "hard", "timeout", False), # RUNTIME after restart
        ]
        patches = [
            patch("vigilantpack.orchestrator.manifest_mod.load", return_value=_MANIFEST),
            patch("vigilantpack.orchestrator.preflight_mod.run_checks", return_value=_PASS_CHECKS),
            patch("vigilantpack.orchestrator.preflight_mod.has_hard_failure", return_value=False),
            patch("vigilantpack.orchestrator.services_mod.start",
                  return_value=StageResult(True, "hard", "started", True)),
            patch("vigilantpack.orchestrator.services_mod.poll_health",
                  side_effect=health_responses),
            patch("vigilantpack.orchestrator.services_mod.restart"),
            patch("vigilantpack.orchestrator.services_mod.tail_logs", return_value=""),
            patch("vigilantpack.orchestrator.models_mod.list_present", return_value=set()),
        ]
        _apply_patches(patches)
        try:
            with pytest.raises(SystemExit) as exc_info:
                run("vigilant.yaml", emit=lambda e: None)
        finally:
            _stop_patches(patches)
        assert exc_info.value.code == 1
