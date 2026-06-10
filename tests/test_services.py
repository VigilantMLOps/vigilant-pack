import pytest
import httpx
from unittest.mock import MagicMock, patch

from vigilantpack.services import (
    StageResult,
    poll_health,
    restart,
    start,
    stop_all,
    stream_logs,
    tail_logs,
)


class TestStart:
    def test_success(self):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            result = start("docker-compose.yml", "ollama")
        assert result.success

    def test_port_conflict_is_deterministic(self):
        with patch("subprocess.run", return_value=MagicMock(
            returncode=1, stderr="port is already allocated on the host"
        )):
            result = start("docker-compose.yml", "ollama")
        assert not result.success
        assert result.error_type == "deterministic"

    def test_image_pull_failure_is_transient(self):
        with patch("subprocess.run", return_value=MagicMock(
            returncode=1, stderr="Unable to find image"
        )):
            result = start("docker-compose.yml", "ollama")
        assert not result.success
        assert result.error_type == "transient"
        assert result.recoverable

    def test_unknown_failure(self):
        with patch("subprocess.run", return_value=MagicMock(
            returncode=1, stderr="something unexpected happened"
        )):
            result = start("docker-compose.yml", "ollama")
        assert not result.success
        assert result.error_type == "unknown"

    def test_passes_correct_docker_command(self):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)) as mock_run:
            start("my-compose.yml", "redis")
        cmd = mock_run.call_args[0][0]
        assert "my-compose.yml" in cmd
        assert "redis" in cmd
        assert "up" in cmd
        assert "-d" in cmd


class TestPollHealth:
    def test_healthy_on_first_try(self):
        with patch("httpx.get", return_value=MagicMock(status_code=200)):
            with patch("time.monotonic", side_effect=[0, 0]):
                result = poll_health("http://localhost:8000/health", timeout=10)
        assert result.success

    def test_4xx_is_healthy(self):
        with patch("httpx.get", return_value=MagicMock(status_code=404)):
            with patch("time.monotonic", side_effect=[0, 0]):
                result = poll_health("http://localhost:8000/health", timeout=10)
        assert result.success

    def test_5xx_is_not_healthy(self):
        # monotonic: deadline=0+5=5, loop-check=0 (enter), httpx fails, sleep, loop-check=100 (exit)
        with patch("httpx.get", return_value=MagicMock(status_code=503)):
            with patch("time.sleep"):
                with patch("time.monotonic", side_effect=[0, 0, 100]):
                    result = poll_health("http://localhost:8000/health", timeout=5)
        assert not result.success
        assert "timed out" in result.message

    def test_connect_error_retries_until_timeout(self):
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            with patch("time.sleep"):
                with patch("time.monotonic", side_effect=[0, 0, 100]):
                    result = poll_health("http://localhost:8000/health", timeout=5)
        assert not result.success
        assert "timed out" in result.message


class TestRestart:
    def test_runs_docker_restart(self):
        with patch("subprocess.run") as mock_run:
            restart("docker-compose.yml", "ollama")
        cmd = mock_run.call_args[0][0]
        assert "restart" in cmd
        assert "ollama" in cmd


class TestStopAll:
    def test_runs_docker_compose_down(self):
        with patch("subprocess.run") as mock_run:
            stop_all("docker-compose.yml")
        cmd = mock_run.call_args[0][0]
        assert "down" in cmd
        assert "--volumes" not in cmd

    def test_uses_correct_compose_file(self):
        with patch("subprocess.run") as mock_run:
            stop_all("my-stack.yml")
        cmd = mock_run.call_args[0][0]
        assert "my-stack.yml" in cmd


class TestTailLogs:
    def test_with_service(self):
        with patch("subprocess.run", return_value=MagicMock(stdout="log output\n", stderr="")):
            result = tail_logs("docker-compose.yml", "ollama", lines=20)
        assert result == "log output"

    def test_without_service(self):
        with patch("subprocess.run", return_value=MagicMock(stdout="all logs\n", stderr="")) as mock_run:
            tail_logs("docker-compose.yml", None)
        cmd = mock_run.call_args[0][0]
        assert "ollama" not in cmd

    def test_combines_stdout_and_stderr(self):
        with patch("subprocess.run", return_value=MagicMock(stdout="out\n", stderr="err\n")):
            result = tail_logs("docker-compose.yml", "svc")
        assert "out" in result
        assert "err" in result

    def test_default_lines_is_30(self):
        with patch("subprocess.run", return_value=MagicMock(stdout="", stderr="")) as mock_run:
            tail_logs("docker-compose.yml", "svc")
        cmd = mock_run.call_args[0][0]
        assert "30" in cmd


class TestStreamLogs:
    def test_passes_follow_flag(self):
        with patch("subprocess.run") as mock_run:
            stream_logs("docker-compose.yml", None, follow=True)
        cmd = mock_run.call_args[0][0]
        assert "--follow" in cmd

    def test_no_follow_flag_when_false(self):
        with patch("subprocess.run") as mock_run:
            stream_logs("docker-compose.yml", None, follow=False)
        cmd = mock_run.call_args[0][0]
        assert "--follow" not in cmd

    def test_keyboard_interrupt_is_swallowed(self):
        with patch("subprocess.run", side_effect=KeyboardInterrupt):
            stream_logs("docker-compose.yml", None, follow=True)
