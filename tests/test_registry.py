from __future__ import annotations

import pytest

from benchtop.eval import registry
from benchtop.policies.simple import NoopPolicy, RandomPolicy


class FakeEnv:
    class action_space:  # noqa: N801 - mimics gymnasium's attribute
        low = [0.0] * 8
        high = [1.0] * 8


def test_known_selectors():
    assert registry.registered() == ("expert", "lerobot", "noop", "random")


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("random", ("random", None)), ("lerobot:/tmp/ckpt", ("lerobot", "/tmp/ckpt"))],
)
def test_split_spec(spec, expected):
    assert registry.split_spec(spec) == expected


def test_resolves_builtin_policies():
    env = FakeEnv()
    assert isinstance(registry.resolve_policy("random", env, 0), RandomPolicy)
    assert isinstance(registry.resolve_policy("noop", env, 0), NoopPolicy)


def test_random_policy_is_reproducible_for_a_seed():
    a = registry.resolve_policy("random", FakeEnv(), 0)
    b = registry.resolve_policy("random", FakeEnv(), 0)
    a.reset(10000)
    b.reset(10000)
    obs = {"state": None}
    assert (a.act(obs) == b.act(obs)).all()


def test_unknown_selector_is_rejected():
    with pytest.raises(ValueError, match="unknown policy"):
        registry.resolve_policy("magic", FakeEnv())
    with pytest.raises(ValueError, match="unknown policy"):
        registry.describe("magic")


def test_lerobot_without_a_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="checkpoint path"):
        registry.resolve_policy("lerobot:", FakeEnv())


def test_missing_implementations_degrade_gracefully():
    """`expert` and `lerobot:` are known selectors even before their modules land."""
    for spec in ("expert", "lerobot:/nonexistent/ckpt"):
        try:
            registry.resolve_policy(spec, FakeEnv())
        except registry.PolicyUnavailableError:
            pass
        except Exception as exc:  # the module exists and failed for its own reasons
            assert not isinstance(exc, ValueError), exc


def test_third_party_policies_register_without_editing_the_registry():
    registry.register("dummy", lambda env, arg, seed: NoopPolicy())
    try:
        assert isinstance(registry.resolve_policy("dummy", FakeEnv()), NoopPolicy)
    finally:
        registry._REGISTRY.pop("dummy")


def test_descriptor_hashes_a_checkpoint(tmp_path):
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"weights")
    descriptor = registry.describe(f"lerobot:{ckpt}")
    assert descriptor.type == "lerobot"
    assert descriptor.checkpoint_path == str(ckpt)
    assert descriptor.checkpoint_sha256 == registry.sha256_of(ckpt)
    assert len(descriptor.checkpoint_sha256) == 64


def test_descriptor_hash_covers_every_file_in_a_checkpoint_dir(tmp_path):
    ckpt = tmp_path / "ckpt"
    (ckpt / "sub").mkdir(parents=True)
    (ckpt / "config.json").write_text("{}")
    (ckpt / "sub" / "model.bin").write_bytes(b"a")
    before = registry.sha256_of(ckpt)
    (ckpt / "sub" / "model.bin").write_bytes(b"b")
    assert registry.sha256_of(ckpt) != before


def test_descriptor_for_a_builtin_policy_has_no_checkpoint():
    descriptor = registry.describe("random")
    assert descriptor.to_dict() == {"spec": "random", "type": "random"}
