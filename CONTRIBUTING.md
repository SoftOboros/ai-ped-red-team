# Contributing to ai-ped-red-team

## Development workflow
1. Fork and clone the repository.
2. Create a feature branch: `git checkout -b feature/your-idea`.
3. Activate the project virtual environment: `source venv/bin/activate` (or create your own).
4. Install dependencies in editable mode: `pip install -e ".[dev,docs]"`.
5. Run `pre-commit install` to enable formatting and linting checks.
6. Add tests for new features and ensure `pytest` passes.
7. Update docs as needed.
8. Submit a pull request with a clear description of the change.

## Pull request checklist
- [ ] Tests added or updated.
- [ ] `pytest` passes locally.
- [ ] Linters (`ruff`, `black`) pass via `pre-commit run --all-files`.
- [ ] Documentation or examples updated if behaviour changes.
- [ ] Added entry to `CHANGELOG.md` under **Unreleased**.

## Questions?
Reach the maintainers at **dev@softoboros.com**.
