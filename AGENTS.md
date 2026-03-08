# AGENTS.md

## Cursor Cloud specific instructions

### Overview

SERF (Semantic Entity Resolution Framework) is a Python 3.12 CLI tool for semantic entity resolution.
It is a pure Python library with no external service dependencies (no databases, Redis, etc.).

See `CLAUDE.md` for comprehensive code style rules, development guidelines, and commands.
See `README.md` for project overview and quick start.

### Key commands

All commands use `poetry run` prefix since dependencies are managed by Poetry:

| Task         | Command                                 |
| ------------ | --------------------------------------- |
| Install deps | `poetry install`                        |
| CLI          | `poetry run serf --help`                |
| Lint (all)   | `poetry run pre-commit run --all-files` |
| Black        | `poetry run black --check src tests`    |
| Flake8       | `poetry run flake8 src tests`           |
| Isort        | `poetry run isort --check src tests`    |
| Type check   | `poetry run zuban check src/serf tests` |
| Tests        | `poetry run pytest tests/ -v`           |

### Environment notes

- **Poetry binary** is installed at `~/.local/bin/poetry`. The PATH is configured in `~/.bashrc`.
- **Java 21** (OpenJDK) is pre-installed and works with PySpark 3.5.5+. Ignore the `WARN NativeCodeLoader` message from Spark — it is harmless.
- **GEMINI_API_KEY** environment variable is required to run the test suite (`tests/test_dspy.py`). Without it, `pytest` will error on setup. All other tooling (lint, type check, CLI) works without it. The `GEMINI_API_KEY` secret is configured in the Cursor Cloud Secrets panel.
- **Pydantic serialization warnings** during `pytest` (e.g. `PydanticSerializationUnexpectedValue`) are harmless — they come from DSPy/LiteLLM internals and do not affect test results.
- **Pre-commit hooks** run black, flake8, isort, zuban, pytest, and prettier. The prettier hook requires Node.js (pre-installed). Run `poetry run pre-commit install` after a fresh clone to activate git hooks.
- The `core.hooksPath` git config may need to be unset before `pre-commit install` works: `git config --unset-all core.hooksPath`.
