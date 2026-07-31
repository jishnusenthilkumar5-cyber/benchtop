# benchtop

Reproducible evaluation for robot manipulation policies, in simulation.

Bring a policy checkpoint, get a scorecard: success rate with a confidence
interval, per-episode records, and rollout videos — against a versioned task
on held-out seeds.

**Status: pre-v0, under construction.** The plan of record is
[`.devin/plans/benchtop-v0.md`](.devin/plans/benchtop-v0.md).

## Why

Training robot policies has good tooling. Knowing whether a policy actually
got better does not. Results get reported on seeds that were trained on, on
task variants that quietly changed between runs, with a success rate quoted
to three digits off twenty episodes.

benchtop treats the evaluation itself as the artifact: tasks are versioned and
immutable, evaluation seeds are held out from collection seeds by protocol,
every run records the git sha and resolved config that produced it, and
success rates come with Wilson intervals so you can see when a difference
isn't one.

## Planned interface

```bash
benchtop collect --episodes 100     # scripted expert -> demonstration dataset
benchtop train   --dataset <path>   # train a policy on those demos
benchtop eval    --suite suites/pick_cube_v0.yaml --policy lerobot:<ckpt>
benchtop dash                       # local dashboard: browse and compare runs
```

## v0 scope

One task — `pick_cube-v0`, a Franka Panda picking a cube and placing it in a
target zone — evaluated end to end with a genuinely learned checkpoint rather
than a scripted stand-in. Policies are state-based in v0 (proprioception plus
object poses, no camera input), which is what makes training tractable on CPU;
cameras are still rendered for rollout video. Image-based policies come after.

## Requirements

Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/). MuJoCo runs on CPU — no
GPU required for anything in v0.

```bash
uv sync --all-extras
uv run pytest
```

## Third-party

The Franka Panda model is vendored from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
under BSD-3-Clause. See [`vendor/VENDORING.md`](vendor/VENDORING.md).
