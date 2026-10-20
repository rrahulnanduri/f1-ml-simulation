# TUM racetrack-database test excerpts

`tracks/Silverstone.csv` and `racelines/Silverstone.csv` are the header line plus the first 20
data rows of the files of the same name in the
[TUM racetrack-database](https://github.com/TUMFTM/racetrack-database) by the Institute of
Automotive Technology, Technical University of Munich, at commit
`e59595d1f3573b30d1ded6a08984935b957688e0`. They are unmodified apart from the truncation.

The database is licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0). These
excerpts are kept only so the CSV parser can be tested offline; the full files are downloaded
at build time from the pinned commit (`f1sim.data.tum_source`) and are not redistributed. See
`THIRD_PARTY.md` at the repository root.
