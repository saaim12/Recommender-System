# Contributing

## Development Workflow

1. Fork and clone the repository.
2. Create a feature branch from `main`.
3. Use Python 3.12.3 virtual environment.
4. Install dependencies with `pip install -r requirements.txt`.
5. Run checks before opening a pull request:

```bash
python -m compileall src scripts tests airflow
pytest -q
```

## Coding Standards

- Keep modules focused and small.
- Use explicit type hints in public functions.
- Add tests for behavior changes.
- Avoid hard-coded paths and credentials.

## Pull Request Checklist

- Tests added or updated.
- Documentation updated.
- No sensitive data committed.
- CI workflow passes.
