# Git Flow — guía rápida

Usaremos Git Flow como modelo de ramificación principal del proyecto. El diagrama a continuación resume el flujo y las ramas principales.

![Diagrama GitFlow](assets/gitflow.png)

Resumen de ramas y propósito

- `main` (o `master`): versión estable y desplegable del proyecto. Solo se mergean cambios listos para producción.
- `develop`: integración de nuevas funcionalidades; rama base para las ramas `feature`.
- `feature/*`: ramas cortas para nuevas funcionalidades o tareas (se crean desde `develop` y se mergean de vuelta a `develop`).
- `release/*`: preparación de una versión para producción; permite correcciones menores y ajustes de metadata. Se mergea a `main` y `develop` al finalizar.
- `hotfix/*`: correcciones urgentes sobre `main`; se mergea a `main` y `develop`.

Flujo recomendado (breve)

1. Trabaja en `develop` como rama de integración.
2. Crea una rama `feature/<descripcion>` desde `develop` para cada tarea.
3. Push y abre Pull Request hacia `develop` cuando la feature esté lista.
4. Ejecuta revisiones y asegúrate que pasan los checks.
5. Para una nueva versión, crea `release/<version>` desde `develop`; al finalizar, mergea a `main` y etiqueta la versión.
6. Para un problema crítico en producción, crea `hotfix/<descripcion>` desde `main` y mergea a `main` y `develop`.

Comandos útiles

- Crear feature: `git checkout -b feature/mi-nueva-funcionalidad develop`
- Actualizar feature con develop: `git fetch && git rebase origin/develop` (o `git merge origin/develop` según preferencia)
- Abrir PR: push de la rama y abrir Pull Request hacia `develop`

Políticas del equipo (sugeridas)

- Cada PR debe pasar los linters y pruebas antes de merge (usar `pre-commit` localmente).
- Usar mensajes de commit con estilo imperativo y opcionalmente Conventional Commits (ej. `feat(...)`, `fix(...)`, `chore(...)`).
- Revisión de al menos una persona antes del merge.
