# VigilantPack

Deterministic startup runner for ML/LLM applications. Brings up Docker services, pulls Ollama models, warms up inference, and waits for your app to be healthy — in a single command.

```
vigilantpack run
```

## How it works

VigilantPack reads a `vigilant.yaml` manifest and executes a 6-stage lifecycle:

```
VALIDATE → PREFLIGHT → INFRA → MODELS → RUNTIME → READY
```

| Stage | What happens |
|-------|-------------|
| **VALIDATE** | Parse and validate `vigilant.yaml` |
| **PREFLIGHT** | Check Docker, env vars, ports, disk space |
| **INFRA** | `docker compose up` each infrastructure service and poll health |
| **MODELS** | Pull missing Ollama models; optionally warm them up |
| **RUNTIME** | Start your application service and wait for its health endpoint |
| **READY** | Print the ready banner with elapsed time and API URL |

Any hard failure (Docker not running, required model can't be pulled, app won't start) aborts immediately with a clear error message.

## Installation

```bash
pip install vigilantpack
```

Requires Python 3.12+ and Docker.

## Quick start

Create `vigilant.yaml` in your project root:

```yaml
vigilantpack: "1"

app:
  name: my-llm-app
  version: 1.0.0

compose: docker-compose.yml

env:
  file: .env
  require:
    - OPENAI_API_KEY

services:
  ollama:
    health: http://localhost:11434/api/tags
    timeout: 60

models:
  - name: llama3.2
    required: true
    warmup: true
  - name: nomic-embed-text
    required: true
    warmup: false

runtime:
  service: my-app
  health: http://localhost:8080/health
  timeout: 45

startup:
  timeout: 300
```

Then run:

```bash
vigilantpack run
```

## Commands

### `vigilantpack run`

Start the full stack. Runs all 6 stages and prints live progress.

```bash
vigilantpack run
vigilantpack run --file path/to/vigilant.yaml
```

On success:

```
─────────────── my-llm-app  v1.0.0  ready  (42s) ───────────────

  API   http://localhost:8080
```

### `vigilantpack doctor`

Check prerequisites without starting anything. Exits 0 if all checks pass, 1 if any hard failure is found.

```bash
vigilantpack doctor
```

Checks: Docker daemon, required env vars, vault path, port availability, disk space.

### `vigilantpack status`

Show the current health of all services and models.

```bash
vigilantpack status
```

```
  Name                Type       Status
  ──────────────────────────────────────────────────────────
  ✓  qdrant           service    HTTP 404
  ✓  ollama           service    HTTP 200
  ✓  nomic-embed-text model      downloaded
  ✓  llama3.2         model      downloaded
  ✓  my-app           runtime    HTTP 200
```

### `vigilantpack stop`

Stop all services. Data volumes are preserved.

```bash
vigilantpack stop
```

### `vigilantpack logs`

Stream or tail logs for all services or a specific one.

```bash
vigilantpack logs                    # all services
vigilantpack logs ollama             # one service
vigilantpack logs ollama --follow    # follow live
```

## `vigilant.yaml` reference

```yaml
# Required — schema version
vigilantpack: "1"

# Required — your application metadata
app:
  name: my-app
  version: 1.0.0

# Required — path to your docker-compose.yml
compose: docker-compose.yml

# Optional — environment configuration
env:
  file: .env           # loaded before ${VAR} substitution
  require:             # variable names that must be set (fail if missing)
    - MY_API_KEY

# Required — infrastructure services (at least one)
# Each entry must have a health URL; VigilantPack polls it until healthy.
services:
  ollama:
    health: http://localhost:11434/api/tags
    timeout: 60        # seconds to wait for health (default: 60)
  qdrant:
    health: http://localhost:6333/health
    timeout: 30

# Optional — Ollama models to manage
models:
  - name: llama3.2          # model name as used in ollama pull
    required: true          # if true, failure aborts the run
    warmup: true            # send a test inference to pre-load into memory
  - name: nomic-embed-text
    required: true
    warmup: false

# Required — your application service
runtime:
  service: my-app           # service name in docker-compose.yml
  health: http://localhost:8080/health
  timeout: 45               # seconds to wait for health (default: 45)

# Optional — global deadline
startup:
  timeout: 300              # total seconds allowed for all stages (default: 300)
```

`${VAR}` references in any string value are resolved against env vars (after loading the `.env` file).

## Preflight checks

`doctor` and `run` both execute the same checks before touching any service:

| Check | PASS | WARN | FAIL |
|-------|------|------|------|
| Docker daemon | running | — | not found / not running |
| Required env vars | all set | — | any missing |
| `VAULT_PATH` | directory exists | — | directory missing |
| Ports | free, or service already healthy on port | — | occupied by unknown process |
| Disk space | ≥ 10 GB free | < 10 GB free | — |

WARN checks are advisory — the run continues. FAIL checks abort immediately.

## Development

```bash
git clone https://github.com/your-org/vigilant-pack
cd vigilant-pack
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

### Project layout

```
src/vigilantpack/
  cli.py          # Click commands and Rich terminal output
  orchestrator.py # 6-stage state machine
  manifest.py     # YAML loading, ${VAR} substitution, validation
  preflight.py    # Environment checks (docker, env, ports, disk)
  models.py       # Ollama model presence, pull, warmup
  services.py     # docker compose start / health poll / logs
tests/
  test_manifest.py
  test_preflight.py
  test_services.py
  test_models.py
  test_orchestrator.py
  test_cli.py
```

### Running tests

```bash
pytest
pytest -v                    # verbose
pytest tests/test_models.py  # single file
```

All 118 tests run in under a second — no Docker or Ollama required (fully mocked).

## Requirements

- Python 3.12+
- Docker Desktop (or Docker Engine + Compose plugin)
- Ollama — only if your manifest includes models
