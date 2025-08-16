# Animal Classification

Proyecto para clasificación de imágenes de animales.

Resumen rápido

- Código de la aplicación: `animal_classification/` (paquete en la raíz).
- Configuración centralizada en `pyproject.toml` (formatters, linters, extras de dependencias).
- Hooks locales con `pre-commit` configurados en `.pre-commit-config.yaml` (black, isort, ruff, mypy, entre otros).
- Gestión de entorno/deps con `uv` (se usan `uv sync`, `uv venv`, `uv lock`).
- Tests: `tests/` (pytest + pytest-cov).

Estructura principal (resumida)

- `Makefile` — comandos de conveniencia (`make test`, `make lint`, `make format`, etc.)
- `pyproject.toml` — metadata del proyecto y configuración de herramientas (black/isort/ruff/pytest)
- `.pre-commit-config.yaml` — hooks que se ejecutan antes de commitear
- `animal_classification/` — paquete de la aplicación (entrypoints y código de app/train/serve)
- `data/` — raw / interim / processed / external (gestión con DVC para datos grandes)
- `models/` — artefactos/weights (no versionar pesos pesados en git)
- `notebooks/` — experimentos y análisis (recomendado usar `jupytext`/`nbqa` si quieres formatearlos)
- `tests/` — pruebas unitarias e integradas (pytest)

Comandos importantes

- Instalar dependencias / sincronizar (uv):

  - `uv sync` (usa el lock / pyproject para instalar deps)
  - `uv venv --python 3.11` (crear venv con `uv`)

- Pre-commit:

  - Instalar hooks localmente: `pre-commit install` o `make precommit-install`
  - Ejecutar todos los hooks en el repo: `pre-commit run --all-files` o `make check`

- Formateo / lint:

  - Formatear: `make format` (isort + black) o `make fmt`
  - Comprobar linters: `make lint` (ruff, isort --check, black --check, mypy)

- Tests:
  - Ejecutar pruebas: `make test` (ejecuta `pytest --cov=animal_classification`)

Configuración y dependencias

- Las herramientas de desarrollo (black, isort, ruff, pre-commit, mypy, pytest, etc.) están en `pyproject.toml` como extras (`[project.optional-dependencies].dev`).
- Dependencias de datos/ML se agrupan en un extra `data` (puedes instalar `pip install -e .[data]` o usar `uv` para sincronizar).
