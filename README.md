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
- `animal_classification/` — paquete de la aplicación; contiene el código fuente organizado por responsabilidad:
  - `animal_classification/app.py` o `animal_classification/app/` — entrada para servir la app (Gradio/serve wrappers)
  - `animal_classification/datasets/` — clases y utilidades para cargar, preprocesar y transformar datasets (data loaders)
  - `animal_classification/models/` — definiciones de modelos, arquitecturas y utilidades relacionadas (NO almacenar pesos grandes aquí)
  - `animal_classification/train/` — scripts y funciones del loop de entrenamiento y pipelines de experimentación
  - `animal_classification/serve/` — código de inferencia y adaptadores para exponer el modelo (p. ej. Gradio handlers)
  - `animal_classification/utils/` — utilidades compartidas (I/O, métricas, transforms, logging)
- `data/` — Sólo se encuentra en local, debido a que es gestionada con DVC (no versionar datos pesados en git)
- `notebooks/` — experimentos y análisis
- `tests/` — pruebas unitarias e integradas (pytest)

Comandos importantes

- Instalar dependencias / sincronizar (uv):

  - `uv venv --python 3.11` (crear venv con `uv`)
  - `uv sync` (usa el lock / pyproject para instalar deps)
    - Para instalar sólo las dependencias principales (comportamiento por defecto):
      - `uv sync`
    - Para incluir un grupo concreto (p. ej. herramientas de desarrollo):
      - `uv sync --group dev`
    - Para excluir un grupo del proceso de sincronización:
      - `uv sync --no-group notebooks`
    - Puedes combinar inclusión y exclusión:
      - `uv sync --group dev --no-group notebooks`

- Pre-commit:

  - Instalar hooks localmente: `pre-commit install` o `make precommit-install`
  - Ejecutar todos los hooks en el repo: `pre-commit run --all-files` o `make check`

- Tests:
  - Ejecutar pruebas: `make test` (ejecuta `pytest --cov=animal_classification`)

Configuración y dependencias

- Las herramientas de desarrollo (black, isort, ruff, pre-commit, mypy, pytest, etc.) están en `pyproject.toml` como extras (`[project.optional-dependencies].dev`).
- Dependencias de datos/ML se agrupan en un extra `data` (puedes instalar `pip install -e .[data]` o usar `uv` para sincronizar).
