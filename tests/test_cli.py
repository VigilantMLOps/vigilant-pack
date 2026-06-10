import pytest
from click.testing import CliRunner
from unittest.mock import patch

from vigilantpack.cli import cli
from vigilantpack.manifest import ManifestError
from vigilantpack.preflight import CheckResult
from vigilantpack.services import StageResult


@pytest.fixture
def runner():
    return CliRunner()


_MANIFEST = {
    "compose": "docker-compose.yml",
    "services": {"ollama": {"health": "http://localhost:11434/api/tags"}},
    "models": [],
    "runtime": {"service": "app", "health": "http://localhost:8000/health"},
}


class TestDoctorCommand:
    def test_all_pass_exits_0(self, runner):
        checks = [CheckResult("docker", "PASS", "Docker version 24.0.0")]
        with patch("vigilantpack.cli.manifest_mod.load", return_value=_MANIFEST):
            with patch("vigilantpack.cli.preflight_mod.run_checks", return_value=checks):
                with patch("vigilantpack.cli.preflight_mod.has_hard_failure", return_value=False):
                    result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0
        assert "All checks passed" in result.output

    def test_fail_exits_1(self, runner):
        checks = [CheckResult("docker", "FAIL", "daemon not running", "start Docker")]
        with patch("vigilantpack.cli.manifest_mod.load", return_value=_MANIFEST):
            with patch("vigilantpack.cli.preflight_mod.run_checks", return_value=checks):
                with patch("vigilantpack.cli.preflight_mod.has_hard_failure", return_value=True):
                    result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 1

    def test_fix_hint_shown_on_fail(self, runner):
        checks = [CheckResult("docker", "FAIL", "not running", "start Docker Desktop")]
        with patch("vigilantpack.cli.manifest_mod.load", return_value=_MANIFEST):
            with patch("vigilantpack.cli.preflight_mod.run_checks", return_value=checks):
                with patch("vigilantpack.cli.preflight_mod.has_hard_failure", return_value=True):
                    result = runner.invoke(cli, ["doctor"])
        assert "start Docker Desktop" in result.output

    def test_manifest_error_exits_1(self, runner):
        with patch("vigilantpack.cli.manifest_mod.load", side_effect=ManifestError("not found")):
            result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 1
        assert "Manifest error" in result.output

    def test_warn_does_not_abort(self, runner):
        checks = [
            CheckResult("disk", "WARN", "7.0 GB free", "free up space"),
            CheckResult("docker", "PASS", "running"),
        ]
        with patch("vigilantpack.cli.manifest_mod.load", return_value=_MANIFEST):
            with patch("vigilantpack.cli.preflight_mod.run_checks", return_value=checks):
                with patch("vigilantpack.cli.preflight_mod.has_hard_failure", return_value=False):
                    result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0


class TestStopCommand:
    def test_calls_stop_all(self, runner):
        with patch("vigilantpack.cli.manifest_mod.load", return_value=_MANIFEST):
            with patch("vigilantpack.cli.services_mod.stop_all") as mock_stop:
                result = runner.invoke(cli, ["stop"])
        mock_stop.assert_called_once_with("docker-compose.yml")
        assert result.exit_code == 0

    def test_manifest_error_exits_1(self, runner):
        with patch("vigilantpack.cli.manifest_mod.load", side_effect=ManifestError("bad")):
            result = runner.invoke(cli, ["stop"])
        assert result.exit_code == 1


class TestLogsCommand:
    def test_all_services_no_follow(self, runner):
        with patch("vigilantpack.cli.manifest_mod.load", return_value=_MANIFEST):
            with patch("vigilantpack.cli.services_mod.stream_logs") as mock_logs:
                runner.invoke(cli, ["logs"])
        mock_logs.assert_called_once_with("docker-compose.yml", None, False)

    def test_specific_service_with_follow(self, runner):
        with patch("vigilantpack.cli.manifest_mod.load", return_value=_MANIFEST):
            with patch("vigilantpack.cli.services_mod.stream_logs") as mock_logs:
                runner.invoke(cli, ["logs", "ollama", "--follow"])
        mock_logs.assert_called_once_with("docker-compose.yml", "ollama", True)

    def test_manifest_error_exits_1(self, runner):
        with patch("vigilantpack.cli.manifest_mod.load", side_effect=ManifestError("bad")):
            result = runner.invoke(cli, ["logs"])
        assert result.exit_code == 1


class TestRunCommand:
    def test_delegates_to_orchestrator(self, runner):
        with patch("vigilantpack.cli.orchestrator.run") as mock_run:
            runner.invoke(cli, ["run"])
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "vigilant.yaml"

    def test_custom_manifest_path_passed_through(self, runner):
        with patch("vigilantpack.cli.orchestrator.run") as mock_run:
            runner.invoke(cli, ["run", "--file", "my-app/vigilant.yaml"])
        assert mock_run.call_args[0][0] == "my-app/vigilant.yaml"


class TestStatusCommand:
    def test_shows_table(self, runner):
        with patch("vigilantpack.cli.manifest_mod.load", return_value=_MANIFEST):
            with patch("vigilantpack.cli.models_mod.list_present", return_value=set()):
                with patch("vigilantpack.cli._health_check", return_value=(True, "HTTP 200")):
                    result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "ollama" in result.output

    def test_manifest_error_exits_1(self, runner):
        with patch("vigilantpack.cli.manifest_mod.load", side_effect=ManifestError("bad")):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 1
