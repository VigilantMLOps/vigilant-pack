import os
import pytest
import yaml
from pathlib import Path

from vigilantpack.manifest import load, ManifestError, ollama_url


def _write_manifest(tmp_path, data):
    p = tmp_path / "vigilant.yaml"
    p.write_text(yaml.dump(data))
    return p


def _base(tmp_path, **overrides):
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("services:\n")
    data = {
        "vigilantpack": "1",
        "app": {"name": "test-app", "version": "0.1.0"},
        "compose": str(compose),
        "services": {"ollama": {"health": "http://localhost:11434/api/tags"}},
        "runtime": {"service": "app", "health": "http://localhost:8000/health"},
    }
    data.update(overrides)
    return data


class TestLoad:
    def test_valid_manifest(self, tmp_path):
        m = load(_write_manifest(tmp_path, _base(tmp_path)))
        assert m["app"]["name"] == "test-app"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ManifestError, match="not found"):
            load(tmp_path / "missing.yaml")

    def test_invalid_yaml_raises(self, tmp_path):
        p = tmp_path / "vigilant.yaml"
        p.write_text(":\nbad: yaml: {[}")
        with pytest.raises(ManifestError, match="YAML parse"):
            load(p)

    def test_non_mapping_yaml_raises(self, tmp_path):
        p = tmp_path / "vigilant.yaml"
        p.write_text("- item1\n- item2\n")
        with pytest.raises(ManifestError, match="mapping"):
            load(p)

    def test_missing_app_name_raises(self, tmp_path):
        data = _base(tmp_path)
        data["app"] = {"version": "1.0"}
        with pytest.raises(ManifestError, match="app.name"):
            load(_write_manifest(tmp_path, data))

    def test_missing_app_version_raises(self, tmp_path):
        data = _base(tmp_path)
        data["app"] = {"name": "x"}
        with pytest.raises(ManifestError, match="app.version"):
            load(_write_manifest(tmp_path, data))

    def test_missing_compose_key_raises(self, tmp_path):
        data = _base(tmp_path)
        del data["compose"]
        with pytest.raises(ManifestError, match="compose"):
            load(_write_manifest(tmp_path, data))

    def test_compose_file_not_found_raises(self, tmp_path):
        data = _base(tmp_path)
        data["compose"] = str(tmp_path / "nonexistent.yml")
        with pytest.raises(ManifestError, match="compose file not found"):
            load(_write_manifest(tmp_path, data))

    def test_empty_services_raises(self, tmp_path):
        data = _base(tmp_path)
        data["services"] = {}
        with pytest.raises(ManifestError, match="at least one"):
            load(_write_manifest(tmp_path, data))

    def test_service_missing_health_raises(self, tmp_path):
        data = _base(tmp_path)
        data["services"] = {"svc": {"timeout": 30}}
        with pytest.raises(ManifestError, match="services.svc.health"):
            load(_write_manifest(tmp_path, data))

    def test_missing_runtime_service_raises(self, tmp_path):
        data = _base(tmp_path)
        data["runtime"] = {"health": "http://localhost:8000/health"}
        with pytest.raises(ManifestError, match="runtime.service"):
            load(_write_manifest(tmp_path, data))

    def test_missing_runtime_health_raises(self, tmp_path):
        data = _base(tmp_path)
        data["runtime"] = {"service": "app"}
        with pytest.raises(ManifestError, match="runtime.health"):
            load(_write_manifest(tmp_path, data))


class TestEnvSubstitution:
    def test_var_substituted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_HOST", "real-host:9999")
        data = _base(tmp_path)
        data["services"]["ollama"]["health"] = "http://${MY_HOST}/health"
        m = load(_write_manifest(tmp_path, data))
        assert m["services"]["ollama"]["health"] == "http://real-host:9999/health"

    def test_unresolved_var_left_as_is(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        data = _base(tmp_path)
        data["label"] = "${MISSING_VAR}"
        m = load(_write_manifest(tmp_path, data))
        assert m["label"] == "${MISSING_VAR}"

    def test_system_env_takes_precedence_over_env_file(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("API_KEY=from-file\n")
        monkeypatch.setenv("API_KEY", "from-system")
        data = _base(tmp_path)
        data["env"] = {"file": str(env_file), "require": []}
        data["key"] = "${API_KEY}"
        m = load(_write_manifest(tmp_path, data))
        assert m["key"] == "from-system"

    def test_env_file_loaded(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FROM_FILE_VAR", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("FROM_FILE_VAR=hello\n")
        data = _base(tmp_path)
        data["env"] = {"file": str(env_file), "require": []}
        data["label"] = "${FROM_FILE_VAR}"
        m = load(_write_manifest(tmp_path, data))
        assert m["label"] == "hello"

    def test_missing_env_file_is_ignored(self, tmp_path):
        data = _base(tmp_path)
        data["env"] = {"file": str(tmp_path / "nonexistent.env"), "require": []}
        m = load(_write_manifest(tmp_path, data))
        assert m["app"]["name"] == "test-app"

    def test_substitution_in_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MODEL_NAME", "llama3.2")
        data = _base(tmp_path)
        data["models"] = [{"name": "${MODEL_NAME}"}]
        m = load(_write_manifest(tmp_path, data))
        assert m["models"][0]["name"] == "llama3.2"


class TestOllamaUrl:
    def test_extracts_base_from_health_url(self):
        m = {"services": {"ollama": {"health": "http://localhost:11434/api/tags"}}}
        assert ollama_url(m) == "http://localhost:11434"

    def test_default_when_no_ollama_service(self):
        assert ollama_url({}) == "http://localhost:11434"

    def test_default_when_services_has_no_ollama(self):
        m = {"services": {"redis": {"health": "http://localhost:6379/ping"}}}
        assert ollama_url(m) == "http://localhost:11434"
