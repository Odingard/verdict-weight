# AGENTS.md

## Cursor Cloud specific instructions

This is a pure Python library (`verdict-weight`) with no external services, databases, or Docker dependencies.

### Quick reference

- **Install (dev):** `pip install -e ".[dev]"` — installs pytest, pytest-cov, scikit-learn, xgboost plus core deps (numpy, scipy).
- **Tests:** `python3 -m pytest tests/` — 197 tests across unit, integration, property, regression, and performance suites.
- **Lint:** No linter is configured in this project.
- **Build check:** `pip install -e .` (the package uses setuptools via `pyproject.toml`).

### Running benchmarks/validation (optional)

These reproduce the paper results and are all deterministic (seed=42):

```
python3 -m validation.synthetic_validation --n 10000 --seed 42
python3 -m benchmarks.ieee_head_to_head --n 2000 --seed 42
python3 -m benchmarks.learned_head_to_head --n 10000 --seed 42
python3 -m benchmarks.cve_validation --n 120 --seed 42
```

### Gotchas

- Use `python3` not `python` — the VM does not alias `python` to `python3`.
- After `pip install`, `pytest` lands in `~/.local/bin`. Ensure `PATH` includes `$HOME/.local/bin` or invoke via `python3 -m pytest`.
- The project has both `pyproject.toml` (v1.2.0) and a legacy `setup.py` (v1.1.0). The `pyproject.toml` is the source of truth for build config and dependencies.
