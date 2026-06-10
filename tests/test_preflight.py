import subprocess
import pytest
from unittest.mock import MagicMock, patch

from vigilantpack.preflight import (
    CheckResult,
    has_hard_failure,
    run_checks,
    _check_docker,
    _check_disk,
    _check_env,
    _check_ports,
    _check_vault,
)


class TestCheckDocker:
    def test_docker_not_in_path(self):
        with patch("shutil.which", return_value=None):
            results = _check_docker()
        assert results[0].status == "FAIL"
        assert "not found" in results[0].message

    def test_docker_daemon_not_running(self):
        with patch("shutil.which", return_value="/usr/bin/docker"):
            with patch("subprocess.run", return_value=MagicMock(returncode=1, stderr=b"")):
                results = _check_docker()
        assert results[0].status == "FAIL"
        assert results[0].fix is not None

    def test_docker_daemon_running(self):
        with patch("shutil.which", return_value="/usr/bin/docker"):
            info = MagicMock(returncode=0)
            ver = MagicMock(returncode=0, stdout="Docker version 24.0.0")
            with patch("subprocess.run", side_effect=[info, ver]):
                results = _check_docker()
        assert results[0].status == "PASS"
        assert "24.0.0" in results[0].message

    def test_docker_info_timeout(self):
        with patch("shutil.which", return_value="/usr/bin/docker"):
            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("docker", 10)):
                results = _check_docker()
        assert results[0].status == "FAIL"
        assert "timed out" in results[0].message


class TestCheckEnv:
    def test_required_var_present(self, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "secret")
        results = _check_env({"env": {"require": ["MY_API_KEY"], "file": ".env"}})
        assert results[0].status == "PASS"

    def test_required_var_missing(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        results = _check_env({"env": {"require": ["MISSING_VAR"], "file": ".env"}})
        assert results[0].status == "FAIL"
        assert "MISSING_VAR" in results[0].fix

    def test_no_required_vars(self):
        assert _check_env({"env": {"require": []}}) == []

    def test_no_env_section(self):
        assert _check_env({}) == []

    def test_multiple_vars_mixed(self, monkeypatch):
        monkeypatch.setenv("PRESENT", "yes")
        monkeypatch.delenv("ABSENT", raising=False)
        results = _check_env({"env": {"require": ["PRESENT", "ABSENT"]}})
        by_name = {r.name.split()[-1]: r for r in results}
        assert by_name["PRESENT"].status == "PASS"
        assert by_name["ABSENT"].status == "FAIL"


class TestCheckVault:
    def test_vault_path_not_set(self, monkeypatch):
        monkeypatch.delenv("VAULT_PATH", raising=False)
        assert _check_vault() == []

    def test_vault_dir_exists(self, tmp_path, monkeypatch):
        (tmp_path / "note.md").write_text("# hello")
        monkeypatch.setenv("VAULT_PATH", str(tmp_path))
        results = _check_vault()
        assert results[0].status == "PASS"
        assert "1 markdown" in results[0].message

    def test_vault_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VAULT_PATH", str(tmp_path / "nonexistent"))
        results = _check_vault()
        assert results[0].status == "FAIL"
        assert results[0].fix is not None


class TestCheckPorts:
    def _manifest(self):
        return {
            "services": {"ollama": {"health": "http://localhost:11434/api/tags"}},
            "runtime": {"service": "app", "health": "http://localhost:8000/health"},
        }

    def test_ports_free(self):
        with patch("vigilantpack.preflight._port_in_use", return_value=False):
            results = _check_ports(self._manifest())
        assert all(r.status == "PASS" for r in results)
        assert len(results) == 2

    def test_port_in_use_and_health_responds(self):
        manifest = {"services": {"svc": {"health": "http://localhost:5432/health"}}, "runtime": {}}
        with patch("vigilantpack.preflight._port_in_use", return_value=True):
            with patch("vigilantpack.preflight._url_responds", return_value=True):
                results = _check_ports(manifest)
        assert results[0].status == "PASS"
        assert "running" in results[0].message

    def test_port_in_use_foreign_process(self):
        manifest = {"services": {"svc": {"health": "http://localhost:5432/health"}}, "runtime": {}}
        with patch("vigilantpack.preflight._port_in_use", return_value=True):
            with patch("vigilantpack.preflight._url_responds", return_value=False):
                results = _check_ports(manifest)
        assert results[0].status == "FAIL"
        assert results[0].fix is not None

    def test_no_port_in_health_url_skipped(self):
        manifest = {"services": {"svc": {"health": "http://localhost/health"}}, "runtime": {}}
        with patch("vigilantpack.preflight._port_in_use", return_value=False):
            results = _check_ports(manifest)
        assert len(results) == 1


class TestCheckDisk:
    def test_plenty_of_space(self):
        with patch("shutil.disk_usage", return_value=MagicMock(free=50 * 1024**3)):
            results = _check_disk()
        assert results[0].status == "PASS"

    def test_low_space_warns(self):
        with patch("shutil.disk_usage", return_value=MagicMock(free=7 * 1024**3)):
            results = _check_disk()
        assert results[0].status == "WARN"

    def test_critically_low_space_warns(self):
        with patch("shutil.disk_usage", return_value=MagicMock(free=2 * 1024**3)):
            results = _check_disk()
        assert results[0].status == "WARN"


class TestHasHardFailure:
    def test_no_failures(self):
        results = [CheckResult("a", "PASS", "ok"), CheckResult("b", "WARN", "advisory")]
        assert not has_hard_failure(results)

    def test_one_failure(self):
        results = [CheckResult("a", "PASS", "ok"), CheckResult("b", "FAIL", "broken")]
        assert has_hard_failure(results)

    def test_empty_list(self):
        assert not has_hard_failure([])
