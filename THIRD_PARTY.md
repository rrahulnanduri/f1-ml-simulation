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

## OpenF1 (CC BY-NC-SA 4.0)

- **Source:** <https://openf1.org>, an unofficial community API that isn't affiliated with
  Formula 1.
- **License:** data under CC BY-NC-SA 4.0, intended for non-commercial and educational use.
- **Used for:** reference-lap positions and sector times from 2023 onwards
  (`f1sim.data.openf1_source`). Responses are cached in the user cache folder and not
  redistributed.

## FastF1 (MIT)

- **Source:** <https://github.com/theOehrly/Fast-F1>, by Philipp Schaefer and contributors.
- **License:** MIT.
- **Used for:** F1 event schedules, timing and position data for ghost laps and track
  alignment. It is a runtime dependency installed from PyPI and is not vendored here. Data it
  downloads is cached in the user cache folder, not in the repository.

## Archivo (SIL Open Font License 1.1)

- **Source:** <https://github.com/Omnibus-Type/Archivo> at commit
  `211127690e8ff106c36c935f7e5e697114cff103` (`fonts/ttf/`), the official repository by
  Omnibus-Type (Héctor Gatti). The repository publishes no GitHub releases, so the static TTFs
  are taken from that pinned commit. Copyright 2020 The Archivo Project Authors.
- **License:** SIL Open Font License 1.1, in `src/f1sim/app/assets/fonts/Archivo/OFL.txt`
  (the repository's `OFL.txt`, unmodified).
- **Used for:** the viewers' type: `ArchivoExpanded-Bold.ttf` for headlines, the speed and gap
  numbers and the wordmark, `ArchivoSemiExpanded-Bold.ttf` for the 10 px UPPERCASE labels and
  `Archivo-Medium.ttf` for names. Static files are bundled (unmodified) because pyglet can't
  select the width axis of the variable font.

## JetBrains Mono (SIL Open Font License 1.1)

- **Source:** <https://github.com/JetBrains/JetBrainsMono>, release tag `v2.304`
  (`fonts/ttf/`). Copyright 2020 The JetBrains Mono Project Authors.
- **License:** SIL Open Font License 1.1, in `src/f1sim/app/assets/fonts/JetBrainsMono/OFL.txt`.
- **Used for:** tabular numbers in the HUD (lap times, gaps, sector times). The static
  `JetBrainsMono-Regular.ttf` (version 2.304) is bundled unmodified as package data.

## OpenStreetMap (ODbL 1.0)

- **Source:** <https://www.openstreetmap.org>, © OpenStreetMap contributors.
- **License:** Open Data Commons Open Database License 1.0
  (<https://opendatacommons.org/licenses/odbl/1-0/>).
- **Used for:** the `highway=raceway` ways in the loader assets
  (`src/f1sim/app/assets/geo/<track_id>.npz`): the raceway outline and the circuit loop,
  projected to the track's metres. `scripts/geo/fetch.py` downloads them at build time (through
  Aleph, the Overpass API or the OSM API); `scripts/geo/bake.py` makes the asset. Each asset
  carries this attribution in its `attribution` field. The asset is a produced work with
  derived data; the raw downloads stay in git-ignored `assets_src/geo/`.

## AWS Terrain Tiles (Tilezen joerd, open elevation data)

- **Source:** <https://registry.opendata.aws/terrain-tiles/>, the Mapzen/Tilezen terrain set on
  AWS Open Data, built from several public elevation sources.
- **License and attribution:** per source, as listed in
  <https://github.com/tilezen/joerd/blob/master/docs/attribution.md> (for Great Britain, the
  Environment Agency's open LIDAR and SRTM among others).
- **Used for:** the elevation profile and the coarse terrain grid in the loader assets, sampled
  from the `geotiff/` tiles (EPSG:3857 metres) at build time.

## Aleph (MIT, optional build-time tool)

- **Source:** <https://github.com/Belluxx/Aleph> (`alephgeo`), MIT. Not a dependency of
  `f1sim` and not vendored: `scripts/geo/fetch.py` runs it if it is on PATH, and then only with
  its `osm` and `terrain` sources. Its `satellite`, `mesh` and `streetview` sources are Google
  data and must not be used for anything in this repository.
