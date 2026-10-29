# Security audit

*Audited 2026-10-29 on `feature/leftovers`; fixes on branch `security/hardening` (D45). Method:
static review of `src/`, `scripts/`, `f1_anatomy/`, `.claude/`, packaging, and the full git
history and branch list.*

**Threat model.** A stranger finds this public repository, clones or forks it, and wants to
(a) run code on the owner's machine, (b) learn about the owner, or (c) harm the repo or its
users. The attacker has no access to the machine or the GitHub login.

**Status key:** Fixed (in code here) / Owner (needs a GitHub-side or history action only the
owner can take) / Accepted.

## Summary

| ID | Severity | Status | Finding |
|----|----------|--------|---------|
| S1 | High | Fixed, + Owner | `run_latest.bat` pulled and pip-installed whatever the remote held |
| S2 | Medium | Fixed | Unrestricted `pickle.loads` of CMA-ES state from checkpoint files |
| S3 | Medium | Owner | FastF1 `.ff1pkl` pickles in public history and two stale branches |
| S4 | Medium | Fixed, + Owner | A local Windows path with the account name in public docs |
| S5 | Low | Fixed | Hardware details and the Claude app's package ID in docs |
| S6 | Low | Fixed, + Owner | Personal email on every commit |
| S7 | Low | Fixed, + Owner | No `SECURITY.md`, Dependabot, pins or branch protection |
| S8 | Low | Fixed | Downloads had no size cap; OpenF1 JSON accepted NaN/Infinity |
| S9 | Low | Fixed | Anatomy preview server: no Host check, served scratch folders |
| S10 | Low | Fixed | Untracked `* - Copy` folders one `git add .` from being published |
| S11 | Info | Fixed | `track_id` became a file name unchecked |
| S12 | Info | Fixed | One `np.load` relied on NumPy's default for `allow_pickle` |

## Findings and fixes

### S1. Auto-update launcher (High)
**Was:** `scripts/run_latest.bat` ran `git pull` then `pip install -e` on any `pyproject.toml`
change, then the app. Anyone able to push to the branch (stolen token, compromised account) got
code execution on the next double-click; pip also runs build hooks of new dependencies.
**Fix:** the launcher now fetches, lists the incoming commits (and flags changes to
`pyproject.toml` or `scripts/`), and only fast-forwards on an explicit **Y**. pip installs
with `-c constraints.txt`, which pins every dependency. `.gitattributes` keeps `.bat` files
CRLF, which `cmd` needs for `goto`. Tested in a scratch clone (N keeps the old commit, Y
fast-forwards).
**Owner:** turn on 2FA with a passkey, and branch protection on `main` (and the branch you run
from): require a PR and block force-pushes.

### S2. Pickle in checkpoints (Medium)
**Was:** `CMAES.load_state_dict` ran `pickle.loads` on the `es_pickle` array, so a run folder
from someone else could run code on load.
**Fix:** `_CMAUnpickler` resolves only the exact 39 cma / numpy / f1sim globals a real
`CMAEvolutionStrategy` pickle uses (recorded from cma 4.5, numpy 2.x); anything else raises
`UnpicklingError`. A module-prefix allowlist was rejected: `numpy.load(..., allow_pickle=True)`
would be a gadget. Tests: round trip, and a crafted `os.system` payload is refused.

### S3. Pickles in history (Medium, owner)
`cache/**/*.ff1pkl` were committed in `9651923`/`0696560`, removed in `f90f34a`, and are still
on `origin/feature/fixed-map` (2 unmerged commits) and `origin/feature/telemetry-walls`
(3 unmerged commits). FastF1 loads them with `pickle`, so they are a lure: a tampered copy in
your FastF1 cache runs code. Not done here because it needs force-pushing or deleting remote
branches with unmerged work while another agent pushes to this repo.
**Owner:** delete the two branches if their work is dead
(`git push origin --delete feature/fixed-map feature/telemetry-walls`); optionally
`git filter-repo --path cache --invert-paths` on all branches and force-push, at a moment no
one else is working. Never copy `.ff1pkl`/`.npz` files from others into your cache.

### S4. Account name in docs (Medium)
**Fix:** the absolute venv path in `docs/m1-interfaces.md` is now `<main checkout>\.venv\...`.
**Owner:** the old path stays in history; treat the Windows account name as public, or
rewrite history (as in S3).

### S5. Machine details in docs (Low)
**Fix:** `PROJECT_LOG.md` and `m3-interfaces.md` now say "integrated GPU", "dual-GPU machine"
and `<Claude package>` instead of the exact GPU models and the app's package family name.

### S6. Email on commits (Low)
**Fix:** this repository's `user.email` is now the GitHub noreply address, so new commits
(including the other agent's) no longer carry the personal address. The global git config is
unchanged.
**Owner:** GitHub → Settings → Emails: tick "Keep my email addresses private" and "Block
command line pushes that expose my email". Old commits keep the old address.

### S7. Repo hardening (Low)
**Fix:** `SECURITY.md` (private reporting, safe-use notes), `.github/dependabot.yml` (weekly
pip updates), `constraints.txt` (pinned versions the launcher installs with).
**Owner:** in the repo's Settings → Security: enable private vulnerability reporting, secret
scanning with push protection, and Dependabot alerts; set branch protection (S1). Your local
`origin/HEAD` points at `feature/leftovers` while GitHub's default is `main`; that's harmless.

### S8. Remote data (Low)
**Fix:** TUM downloads are capped at 8 MB and OpenF1 responses at 256 MB (memory DoS);
OpenF1 JSON with NaN/Infinity, or that doesn't decode, raises `OpenF1Error`. TUM files are
already fetched by full commit SHA over HTTPS, so their content can't change; no hashes added.

### S9. F1 Anatomy preview server (Low)
**Fix:** `f1_anatomy/serve.mjs` answers only requests whose `Host` is `127.0.0.1`,
`localhost` or `[::1]` on its port (blocks DNS rebinding), and refuses `tmp/` and `shots*`.
Checked by hand: `/site/` 200, a foreign Host 404, `shots - Copy/` 404.
`vendor/scripts/session.mjs` (reads `~/.claude/projects` transcripts) is called by nothing in
the build and makes no network request; leave it dormant.

### S10. `Copy` folders (Low)
**Fix:** `.gitignore` ignores `* - Copy/` and `* - Copy (*)/`. The folders themselves were
left in place (they are yours); delete them when you no longer need them.

### S11. `track_id` (Info)
**Fix:** `paths.track_file` raises `ValueError` unless the id is letters, digits, `_` or `-`.

### S12. `np.load` (Info)
**Fix:** `metrics.read_generation` passes `allow_pickle=False` explicitly. `subprocess` use
was already safe (list argv, no shell); no `eval`, `exec`, `os.system`, `yaml.load` or
archive extraction exists.

## What an outsider can still learn

Your first name, GitHub handle, the old commit email and the old Windows path (all in
history until rewritten). No API keys, tokens, passwords or `.env` files exist in the tree
or in any branch's history.
