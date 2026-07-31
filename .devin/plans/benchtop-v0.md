# Benchtop v0 — sim-only robot policy evaluation harness

MuJoCo + Franka Panda. Goal: prove the product loop end-to-end — **bring a policy
checkpoint, get a reproducible scorecard** — on one task, with a real learned policy
trained in-repo.

## Product thesis

Evaluation, not training, is the product: versioned tasks, held-out seeds, honest
metrics with confidence intervals, videos, and a dashboard. v0 is the smallest thing
that demonstrates that loop with a genuinely learned checkpoint (not a scripted stand-in).

## Locked scope decisions (from planning Q&A)

1. **Evaluate a real learned checkpoint** — trained in-repo: scripted expert generates
   demos → LeRobot ACT trains on them → harness evaluates the checkpoint.
2. **One task**: cube pick-and-place (`pick_cube-v0`), done well (seeded randomization,
   crisp success criterion, held-out eval seeds).
3. **Surface**: CLI (`benchtop collect|train|eval|dash`) + local web dashboard.
4. **Code only** — no tutorial/docs track. README quickstart + docstrings, nothing more.

### Assumptions shaping the scope (named, not silent)

- **v0 learned policy is state-based** (proprioception + object poses via
  `observation.environment_state`), no camera input. This is what makes training
  feasible on CPU (Devin VMs and the user's Mac — no GPU anywhere). Cameras are still
  rendered for rollout videos. Image-based policies are post-v0.
- Dashboard is no-build: FastAPI + Jinja + vanilla JS. No node toolchain in v0.

## Verified facts (2026-07-30)

- LeRobot latest release **v0.6.0** (2026-07-06). Python ≥3.12 (repo already pins 3.12).
  Breaking changes in 0.6.0: training deps moved to `lerobot[training]` extra, import
  paths changed ("use canonical public entry points"), torch ≥2.7 required. **Pin
  `lerobot==0.6.0`** and isolate all lerobot imports in two modules (adapter + collect).
- ACT config: "at least one image **or** `observation.environment_state`" — state-only
  ACT is supported and is LeRobot's recommended starter policy (fast, low compute).
- `vendor/mujoco_menagerie/franka_emika_panda/` is already vendored (BSD-3, LICENSE
  retained) with `panda.xml` (position actuators) + `scene.xml`.
- Repo state: **zero commits, no remote.** mujoco≥3.11, numpy≥2.5 in pyproject; empty
  `src/benchtop/__init__.py`.

## Core design

### Task: `pick_cube-v0`

- Scene: menagerie Panda + table + 5cm cube + target zone marker. Own XML in
  `src/benchtop/envs/assets/`, including the vendored panda via `<include>`/attach.
- Physics dt 2ms; control at 50Hz (10 substeps). Episode cap: 500 control steps (10s).
- Reset randomization (seeded): cube XY in a ~20×20cm region, target XY in a disjoint
  region. Modest ranges in v0 — expand later as task versions (`pick_cube-v1`, ...).
- Success: cube center within 3cm XY of target center, resting on table (z tolerance),
  cube speed < eps, held 10 consecutive steps. Also record `lifted` sub-event.
- **Task semver is a product primitive**: env class `PickCubeV0`; semantics of a
  released version never change — behavior changes mean a new version.

### Observation / action spec (frozen in Phase 0, `core/types.py`)

- `observation.state` (proprio): 7 joint pos + 7 joint vel + gripper width → 15-dim.
- `observation.environment_state`: cube pos(3) + cube quat(4) + target pos(3) → 10-dim.
- `action`: 7 joint position targets + 1 gripper command → 8-dim (matches panda.xml
  position actuators). Scripted expert reaches Cartesian waypoints via damped-least-
  squares IK on MuJoCo jacobians, emitting the same 8-dim actions.
- Gymnasium `Env` API; obs returned as a flat dict of float32 arrays.

### Seed protocol (eval hygiene)

- Demo collection: seeds 0–99. Evaluation suites: seeds 10_000+ (held out, listed
  explicitly in the suite YAML). Never evaluate on collection seeds.

### Run artifact schema (frozen in Phase 0; dashboard + runner both build against it)

```
runs/<run_id>/
  manifest.json     # run_id, git sha, benchtop version, resolved suite config,
                    # policy descriptor (type, checkpoint path+hash), task version,
                    # platform, timestamps, durations
  metrics.json      # success_rate + 95% CI (Wilson), lift_rate, mean/median
                    # steps-to-success (successes only), mean final cube→target dist
  episodes.jsonl    # per-episode: seed, success, lifted, steps, time_to_success,
                    # final_distance
  videos/ep{NNN}.mp4
```

`runs/`, `datasets/`, `outputs/` are gitignored. A tiny fixture run is committed under
`tests/fixtures/runs/` so the dashboard work item is independent.

### Package layout

```
src/benchtop/
  core/types.py         # obs/action spec, EpisodeResult, manifest/metrics schemas
  envs/                 # PickCubeV0 + assets/*.xml
  policies/base.py      # Policy protocol: reset(seed), act(obs) -> action
  policies/simple.py    # RandomPolicy, NoopPolicy
  policies/expert.py    # scripted pick-place expert          (WI-1)
  ik.py                 # damped-least-squares IK helper      (WI-1)
  data/collect.py       # expert rollouts -> LeRobotDataset   (WI-1)
  eval/runner.py        # episode loop, seeding, video capture (WI-2)
  eval/metrics.py       # aggregation, Wilson CI              (WI-2)
  eval/artifacts.py     # run dir writer/reader               (WI-2)
  adapters/lerobot_policy.py  # checkpoint -> Policy          (WI-4)
  train/                # lerobot-train config + wrapper      (WI-4)
  server/               # FastAPI app, templates, static      (WI-3)
  commands/{collect,train,eval,dash}.py  # typer sub-apps (stubs in Phase 0)
  cli.py                # mounts the four sub-apps
suites/pick_cube_v0.yaml
tests/
```

Dependencies (all declared in Phase 0, single `uv add` — work items never touch
pyproject): gymnasium, typer, pyyaml, imageio[ffmpeg], fastapi, uvicorn, jinja2,
lerobot==0.6.0 + `[training]` extra (brings torch ≥2.7 CPU); dev: pytest, ruff.

---

## Execution plan

### Phase 0 — Foundation (sequential; one session, or local)

Everything Phase 1 depends on. Deliverables:

1. GitHub repo `benchtop` created (user confirms name/visibility), initial commit
   pushed, including `.devin/blueprint.yaml`:
   - initialize: Python 3.12 via setup-python; uv; persist PATH via `$ENVRC`;
     apt `libegl1 libosmesa6 ffmpeg` (headless rendering: `MUJOCO_GL=egl`, fallback
     `osmesa`, on Linux; macOS needs neither)
   - maintenance: `uv sync --all-extras`
   - knowledge: test (`uv run pytest`, markers `slow`/`render` excluded by default),
     lint (`uv run ruff check && uv run ruff format --check`), layout, conventions
     (spec frozen in core/types.py; lerobot imports only in adapters/ and data/;
     task versions are immutable)
2. All dependencies declared; `uv.lock` updated.
3. `core/types.py` with the frozen obs/action spec and artifact schemas.
4. **Working `PickCubeV0`**: reset/step/seeded randomization/success detection/
   offscreen camera render. Smoke-tested: deterministic given seed, success detector
   unit-tested by teleporting the cube.
5. `policies/base.py` + RandomPolicy/NoopPolicy; `cli.py` + four command stubs
   (each prints "not implemented"); suite YAML; committed dashboard fixture run.
6. CI (GitHub Actions): ruff + `pytest -m "not slow and not render"` on ubuntu.
7. ruff configured; README stub.

Done when: fresh clone → `uv sync` → `uv run pytest` green on macOS and Linux CI.

### Phase 1 — Parallel work items (4 sessions, file-disjoint)

**Collision rules:** each item owns only the files listed; nobody edits pyproject,
core/types.py, envs/, or another item's files. Schema/spec change needed → stop and
flag, don't edit.

**WI-1: Scripted expert + demo collection**
- Owns: `policies/expert.py`, `ik.py`, `data/collect.py`, `commands/collect.py`, tests.
- Expert: DLS-IK waypoint sequence (pregrasp → grasp → close → lift → move → place →
  open → retreat) emitting 8-dim actions.
- `benchtop collect --episodes N --seed-start 0 --out datasets/pick_cube_v0_expert`
  writes a LeRobotDataset (fps=50, features matching the frozen spec; only
  successful episodes kept, count reported).
- Done when: expert ≥90% success over seeds 0–49 (marked `slow`); dataset loads
  round-trip via lerobot's public API.

**WI-2: Eval runner + metrics + artifacts + `eval` CLI**
- Owns: `eval/*`, `commands/eval.py`, `suites/pick_cube_v0.yaml` finalization, tests.
- `benchtop eval --suite suites/pick_cube_v0.yaml --policy random` runs the seed
  list, writes the full artifact schema incl. videos (video capture optional per
  suite flag; `render` marker for tests that need a GL context).
- Policy selector string: `random`, `noop`, `expert` (if present), `lerobot:<path>`
  — resolved via a small registry so WI-1/WI-4 plug in without edits here.
- Done when: fixture-conformant run dir produced with RandomPolicy; metrics unit
  tests pass (Wilson CI checked against known values); manifest carries git sha +
  config + policy hash.

**WI-3: Dashboard**
- Owns: `server/*`, `commands/dash.py`, tests.
- Built against the committed fixture run only. `benchtop dash --runs-dir runs/`
  serves: runs table (policy, task, success rate ± CI, date) → run detail (metric
  cards, video grid, episodes table) → compare view (two runs side by side).
  FastAPI + Jinja + vanilla JS; API routes `/api/runs`, `/api/runs/{id}` for later
  hosted use.
- Done when: API smoke tests pass against fixtures; pages render fixture data.

**WI-4: LeRobot adapter + training entrypoint**
- Owns: `adapters/lerobot_policy.py`, `train/*`, `commands/train.py`, tests.
- Adapter: load ACT checkpoint dir → Policy; maps our obs dict to
  `observation.state`/`observation.environment_state`, handles normalization and
  action chunking, torch CPU.
- `benchtop train --dataset <path> --out outputs/train/<name>` wraps lerobot-train
  (ACT, state-only features, no vision backbone, CPU-sized: batch 64, ~30k steps,
  documented expected runtime).
- Done when: adapter round-trips a randomly-initialized ACT (train not required):
  fabricate obs → action of correct shape/dtype; one 20-step rollout in PickCubeV0
  executes without error.

### Phase 2 — Integration + baseline (sequential; one session, after Phase 1 merges)

1. `collect` 100 demos (seeds 0–99) → `train` ACT on CPU (time-boxed; also eval a
   mid-training checkpoint, keep the better) → `eval` all of {random, expert,
   lerobot:<ckpt>} on the held-out suite (100 episodes, seeds 10_000+).
2. Publish checkpoint as a **GitHub release asset** (default; HF Hub only if user
   provides a token); `benchtop eval` accepts a local checkpoint path regardless.
3. README quickstart: clone → sync → collect → train → eval → dash, with the actual
   numbers and a sample video/gif. Tag `v0.1.0`.
4. Acceptance gate: trained ACT ≥50% success on held-out seeds; expert ≥90%; random
   ~0%; dashboard shows all three runs comparably; CI green.
   If ACT <50%: first tighten randomization ranges / add demos (documented as task
   config, not silent); fallback policy: LeRobot diffusion (state-only) — still a
   real learned checkpoint.

Session budget: 1 (Phase 0) + 4 parallel (Phase 1) + 1 (Phase 2) = 6 sessions.

## Risks

- **lerobot 0.6.0 API churn** — pinned; imports quarantined in `adapters/` + `data/`.
- **CPU training quality/time** — state-only ACT is small; expert demos are clean;
  fallbacks staged above. Training runs fine on a Devin VM (no GPU needed).
- **Scripted expert brittleness** — modest v0 randomization; success-only filtering
  on collection; expert success rate is itself a tracked metric.
- **Headless rendering on Linux** — blueprint installs EGL/OSMesa and sets
  `MUJOCO_GL`; `render`-marked tests excluded from default runs. macOS: offscreen
  rendering works natively; interactive viewer would need `mjpython` (debug only).
- **Menagerie licensing** — BSD-3 with LICENSE retained in vendor dir; fine.

## Out of scope (post-v0 roadmap, do not build now)

Image-based policies (cameras already rendered); more tasks + a task registry;
adapters for external benchmarks/checkpoints (robomimic Lift, LIBERO); vectorized/
batched eval (MJX variants are already vendored); hosted dashboard + "eval as CI"
(GitHub App posting scorecards on PRs); leaderboards; domain-randomization sweeps.

## Needs user

1. Confirm GitHub repo creation: name `benchtop`, private or public? (outward-facing)
2. Approve this plan (esp. the state-based-policy assumption for v0).
3. Optional: HF Hub token if checkpoints should live on the Hub instead of GitHub
   releases (default: GitHub release asset).
