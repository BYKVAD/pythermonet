# Contributing to pythermonetII

## Branch model

- **`main`** — stable/LTS branch. Other repos depend on this being releasable at all
  times. Protected via a GitHub ruleset.
- **`dev`** — integration branch. All new work forks from here.
- **`feature/*`, `fix/*`** — short-lived, fork from `dev`, PR back into `dev`.
- **`hotfix/*`** — urgent fixes only. Fork from `main`, PR into `main`, then merge back
  into `dev` afterward so the branches don't drift.

## Merge strategy

- **Squash merge** feature/fix branches into `dev` — keeps one commit per change,
  hides WIP/typo-fix noise.
- **Regular merge commit** for `dev` → `main` — preserves a visible release boundary,
  makes reverting an entire release trivial. Do **not** squash this direction.
- GitHub can't restrict merge method by target branch, so this convention is enforced
  by habit, not tooling — please follow it manually.

## Versioning

- SemVer (`MAJOR.MINOR.PATCH`).
- Version string lives in `pyproject.toml` (`[project] version`).
- Tags are cut on `main` only, at the moment a `dev` → `main` PR merges. Never tag
  `dev`.
- Before pushing a tag, verify it points at a commit where `pyproject.toml`'s version
  already matches the tag (`git show <tag>:pyproject.toml`) — a tag pointing at the
  wrong commit is a real failure mode, not a hypothetical one.
- Manual for now — version bumps and tagging are owned by a single maintainer. This
  may move to an automated tool (e.g. Conventional Commits + `python-semantic-release`)
  once there's a test suite to gate it on.
- **Dependent repos must pin to a version tag** (`@v0.2.0`), never to `@main`/`@dev`.

## Pull requests

- `main` and `dev` both require changes to land via PR (see repo rulesets under
  Settings → Rules).
- On `dev`, required approvals are set to **0** — GitHub cannot allow a contributor to
  approve their own PR, and with a small/solo team that would otherwise deadlock every
  merge. "Require PR" stays on regardless, so every change to `dev` still has a
  diff/discussion trail even without a second reviewer.
- On `main`, PRs require review as normal — `main` → PRs are typically release PRs from
  `dev`, or `hotfix/*` branches.
- Repo/org admins are on the bypass list for both rulesets, so the version-bump/tag
  owner can push release commits and tags directly when needed.

## Tags

- Tags matching `v*` are protected (deletion and updates restricted) — once pushed, a
  tag is immutable. If a tag is wrong, it must be caught and fixed *before* it's
  pushed.
