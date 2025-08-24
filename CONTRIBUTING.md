# Contributing to the project

Thank you for contributing. This file explains the recommended workflow and rules for collaborating in this repository.

Quick summary

- Follow the Git Flow documented in [gitflow.md](./docs/gitflow.md).
- Use short-lived branches: `feature/*`, `release/*`, `hotfix/*`.
- Run `pre-commit` and `pytest` before opening a PR.

1. Setup

- Clone the repository and create a virtual environment. Recommended using `uv` (if available):
  - `uv venv --python 3.11`
  - `uv sync` (installs dependencies from `pyproject.toml` and `uv.lock`)
  - Install pre-commit hooks locally: `pre-commit install`


2. Branches and naming

- Integration branch: `develop`.
- Create feature branches from `develop`: `feature/brief-description`.
- For urgent fixes on production, create `hotfix/brief-description` from `main`.
- See `docs/gitflow.md` for the branching diagram and flow.

3. Commit messages

- Use imperative, clear commit messages. Conventional Commits are recommended (optional):
  - `feat(scope): add inference endpoint`
  - `fix(train): correct early stopping bug`
  - `chore(deps): update dev dependencies`

4. Pre-commit and quality checks

- Install hooks locally: `make precommit-install` or `pre-commit install`.
- Hooks will run automatically on `git commit` for staged files. To run all hooks across the repo:
  - `pre-commit run --all-files` or `make check`
- Format: `make format` (runs isort and black)
- Lint: `make lint` (ruff, isort --check, black --check, mypy)

5. Tests

- Run tests locally: `make test` (runs `pytest --cov=animal_classification`).
- Add unit tests for new functions and integration tests for pipeline changes.

6. Opening a Pull Request

- Push your branch and open a PR targeting `develop`.
- Include a clear description, reproduction steps (if applicable), and a concise list of changes.
- Ensure:
  - Pre-commit hooks pass (or the PR applies suggested fixes).
  - Relevant tests pass.
  - Documentation updated if needed.

7. PR checklist (before merge)

- [ ] At least one reviewer approved the changes.
- [ ] All pre-commit hooks pass and there are no lint errors.
- [ ] Tests relevant to the change pass.
- [ ] Documentation updated when applicable.

8. Adding dependencies

- Add dev tools / extras to `pyproject.toml` under `[project.optional-dependencies]` and use `uv` to sync.
- If `uv` is not used, document new dependencies in the PR with instructions to install them locally.

9. Notebooks

- Notebooks live in `notebooks/`. If you want formatting/linting on `.ipynb`, use `nbqa` or keep a paired `.py` version via `jupytext` and run hooks against the `.py` file.

10. Questions and communication

- For questions about workflow, dependencies, or reviews, open an issue labeled `question` or contact the team directly.

Thanks for contributing — your work improves the project.
