# Third-party data and software

## TUM racetrack-database (LGPL-3.0)

- **Source:** <https://github.com/TUMFTM/racetrack-database>, by the Institute of Automotive
  Technology (FTM), Technical University of Munich.
- **License:** GNU Lesser General Public License v3.0 (LGPL-3.0).
- **Used for:** circuit centre lines and track widths (`tracks/`) and minimum-curvature racing
  lines (`racelines/`).
- **How:** `f1sim.data.tum_source` downloads the CSV files at build time from the pinned commit
  `e59595d1f3573b30d1ded6a08984935b957688e0` into the user cache folder. The data is not
  redistributed with this repository, and tracks built from it are not committed.
- **Exception:** `tests/fixtures/tum/` holds two 20-row excerpts of the Silverstone files, used
  only for offline parser tests (see the README in that folder).

## FastF1 (MIT)

- **Source:** <https://github.com/theOehrly/Fast-F1>, by Philipp Schaefer and contributors.
- **License:** MIT.
- **Used for:** F1 event schedules, timing and position data for ghost laps and track
  alignment. It is a runtime dependency installed from PyPI and is not vendored here. Data it
  downloads is cached in the user cache folder, not in the repository.
