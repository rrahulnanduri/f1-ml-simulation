# Security

Please report a vulnerability privately through GitHub: the repository's **Security** tab →
**Report a vulnerability**. Don't open a public issue for it.

Things to know when running this project:

- **Only load files you made.** Training checkpoints, run folders, ghost and trace `.npz`
  files and FastF1's cache are data, but treat any you didn't create yourself with care. The
  app loads arrays with `allow_pickle=False` and the CMA-ES state through an allowlist
  unpickler; FastF1's own cache (`.ff1pkl`) is plain pickle, so never copy one in from
  someone else.
- **`scripts/run_latest.bat` shows new commits before applying them.** Read the list (and
  any change to `pyproject.toml` or `scripts/`) before you answer Y: updating runs that code.
- The only network access is HTTPS downloads from `raw.githubusercontent.com` (TUM track
  data at a pinned commit), `api.openf1.org`, and FastF1's sources. Nothing is uploaded.
