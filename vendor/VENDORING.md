# Vendored third-party assets

## mujoco_menagerie (Franka Emika Panda only)

- Upstream: https://github.com/google-deepmind/mujoco_menagerie
- Pinned commit: `71f066a`
- License: BSD-3-Clause (see `mujoco_menagerie/LICENSE`, retained verbatim)
- Citation: `mujoco_menagerie/CITATION.cff`

Only `franka_emika_panda/` is vendored. The upstream repo ships models for
several dozen robots; benchtop uses one, and carrying the rest would triple
the clone cost for every session with no benefit.

This is a **flat copy, not a submodule**. That is deliberate: cloud agent VMs
clone the repo and must find the model already present, with no init step and
no network dependency at setup time.

To refresh against a newer upstream:

```bash
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie /tmp/mm
rm -rf vendor/mujoco_menagerie/franka_emika_panda
cp -R /tmp/mm/franka_emika_panda vendor/mujoco_menagerie/
cp /tmp/mm/LICENSE /tmp/mm/CITATION.cff vendor/mujoco_menagerie/
```

Then update the pinned commit above. Model geometry changes can move success
rates, so a refresh means a new task version (`pick_cube-v1`), never a silent
edit to an existing one.
