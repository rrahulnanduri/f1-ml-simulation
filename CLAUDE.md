# Notes for agents

- **Read [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md) first.** Its "Current state" section says
  where the project stands and what's next.
- **Keep the log current:**
  - add a `D<n>` entry for every design decision;
  - add a change-log row for every major change;
  - rewrite "Current state" before you finish a session.
- **Environment:** the venv is `.venv` (Python 3.13); install with `pip install -e ".[dev,fast,es]"`.
- **Before committing:** run `pytest -q` and `ruff check`.
- **Deeper docs:**
  - `docs/revamp-plan.md`: design and milestones.
  - `docs/data-formats.md`: file formats.
  - `docs/m1-interfaces.md`, `docs/m2-interfaces.md`, `docs/m3-interfaces.md`: module contracts.
