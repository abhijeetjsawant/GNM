#!/usr/bin/env python3
"""D1 (fix): re-derive the MPFB body asset under the repaired joint map -- a RELABEL.

WHAT THIS IS. `body_provider.DEFAULT_MPFB_JOINT_MAP` used to send AutoAnim's `Left*`
joints onto MPFB's `.R` bones. Repairing that map changes which MPFB bone answers to each
AutoAnim name and nothing else: the MakeHuman hm08 basemesh, its triangles, its skin
weights and its bone rest transforms are all exactly what they were. So an asset built
from the old map and one built from the new map differ by a PERMUTATION of the per-joint
arrays and a relabelling of the per-vertex bone indices -- which is what this module
applies, so that the repair can be measured without a Blender/MPFB regeneration.

    old_map[N'] == new_map[N]   =>   the bone that used to answer to N' answers to N

WHY A RELABEL AND NOT A MIRROR. The review proposed negating every vertex's X and swapping
each vertex's Left/Right weights. On a bilaterally symmetric bind pose that reaches the
same joint positions -- and it is a REFLECTION of the surface: it reverses every triangle's
winding and inverts every normal, so the character renders inside out. Nothing in the D1
gate can see that: not a forward-dot, not a handedness sign, not a silhouette IoU, because
an inside-out mesh has the same outline. `tests/test_facing_fix.py` demonstrates the sign
flip on the mesh's signed volume. This route moves no vertex at all.

WHAT IS ASSERTED RATHER THAN ASSUMED
  * the two maps are the same relabelling on both sides (a bijection over joint names);
  * the parent table survives the permutation unchanged -- it does, because the MPFB rig
    is structurally symmetric, and if it ever stops being so this raises rather than
    writing a skeleton whose hierarchy quietly changed;
  * `vertices_m`, `triangles` and the weight VALUES come through byte-identical.

It is deliberately NOT written into `.cache/.../detailed-hands`: the before arm of every
gate, and every other lane, still needs the delivered asset exactly as it is.

    .venv/bin/python tools/compare/d1_asset_relabel.py \
        --out artifacts/compare/d1-fix/body-run
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SOURCE = ROOT / ".cache/autoanim_gnm/body-provider/run/detailed-hands"

# The map as it stood before the repair, kept here verbatim so the derivation states both
# ends of the relabelling rather than inferring one of them from a string rule.
LEGACY_DETAILED_MPFB_JOINT_MAP: dict[str, str] = {
    "Root": "root", "Hips": "spine05", "Spine": "spine04", "Chest": "spine03",
    "UpperChest": "spine01", "Neck": "neck01", "Head": "head",
    "LeftEye": "eye.R", "RightEye": "eye.L",
    "LeftShoulder": "clavicle.R", "LeftUpperArm": "upperarm01.R",
    "LeftLowerArm": "lowerarm01.R", "LeftHand": "wrist.R",
    "RightShoulder": "clavicle.L", "RightUpperArm": "upperarm01.L",
    "RightLowerArm": "lowerarm01.L", "RightHand": "wrist.L",
    "LeftUpperLeg": "upperleg01.R", "LeftLowerLeg": "lowerleg01.R",
    "LeftFoot": "foot.R", "LeftToes": "toe1-1.R",
    "RightUpperLeg": "upperleg01.L", "RightLowerLeg": "lowerleg01.L",
    "RightFoot": "foot.L", "RightToes": "toe1-1.L",
}
for _side, _mpfb_side in (("Left", "R"), ("Right", "L")):
    for _index, _finger in enumerate(("Thumb", "Index", "Middle", "Ring", "Little"), start=1):
        _segments = (("Metacarpal", "Proximal", "Distal") if _finger == "Thumb"
                     else ("Proximal", "Intermediate", "Distal"))
        for _segment_index, _segment in enumerate(_segments, start=1):
            LEGACY_DETAILED_MPFB_JOINT_MAP[f"{_side}{_finger}{_segment}"] = (
                f"finger{_index}-{_segment_index}.{_mpfb_side}"
            )


def name_permutation(names: list[str]) -> list[int]:
    """`out[i]` is the OLD slot holding the bone that now answers to `names[i]`."""
    from autoanim_gnm.body_provider import DETAILED_MPFB_JOINT_MAP

    fixed = dict(DETAILED_MPFB_JOINT_MAP)
    legacy = dict(LEGACY_DETAILED_MPFB_JOINT_MAP)
    if set(fixed) != set(legacy) or set(fixed) != set(names):
        raise ValueError("the two joint maps do not cover the same joint names")
    if len(set(fixed.values())) != len(fixed) or len(set(legacy.values())) != len(legacy):
        raise ValueError("a joint map is not injective over MPFB bones")
    if set(fixed.values()) != set(legacy.values()):
        raise ValueError("the repair changed WHICH MPFB bones are used, not just their labels")
    legacy_slot = {legacy[name]: slot for slot, name in enumerate(names)}
    return [legacy_slot[fixed[name]] for name in names]


def relabel_asset(source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply the repaired joint map to an asset built under the old one.

    Per-joint arrays are permuted; per-vertex bone indices are remapped; every other array
    is passed through untouched. Nothing is scaled, mirrored, reordered or re-welded.
    """
    names = [str(name) for name in source["joint_names"].tolist()]
    permutation = np.asarray(name_permutation(names), dtype=np.int64)
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation))

    parents = np.asarray(source["parents"], dtype=np.int64)
    permuted_parents = np.array(
        [-1 if parents[old] == -1 else inverse[parents[old]] for old in permutation],
        dtype=parents.dtype,
    )
    if not np.array_equal(permuted_parents, parents):
        raise ValueError(
            "the MPFB rig's hierarchy is not symmetric under this relabelling; a "
            "permutation would silently change the skeleton's parent table"
        )

    out = dict(source)
    out["parents"] = parents
    out["joint_names"] = source["joint_names"]                 # names keep their slots
    for key in ("local_rest_matrices", "inverse_bind_matrices"):
        out[key] = np.asarray(source[key])[permutation]
    indices = np.asarray(source["joint_indices"])
    out["joint_indices"] = inverse[indices.astype(np.int64)].astype(indices.dtype)
    if "gnm_head_socket_matrix" in source:
        # The head socket hangs off `Head`, which is on the midline and is not relabelled.
        out["gnm_head_socket_matrix"] = source["gnm_head_socket_matrix"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "artifacts/compare/d1-fix/body-run")
    arguments = parser.parse_args()

    source_npz = arguments.source / "neutral-body.npz"
    source_json = arguments.source / "neutral-body.json"
    # This project's own npz carries object arrays; `allow_pickle` is required and the
    # file is a local build product of our pinned MPFB run, not third-party input.
    source = dict(np.load(source_npz, allow_pickle=True))
    fixed = relabel_asset(source)

    arguments.out.mkdir(parents=True, exist_ok=True)
    target = arguments.out / "neutral-body.npz"
    np.savez(target, **fixed)

    from autoanim_gnm.body_provider import DETAILED_MPFB_JOINT_MAP

    # The manifest's field set is EXACT (`validate_body_asset`), and the relabel touches
    # nothing it describes -- `joint_names`, `parents`, vertex and triangle counts and the
    # skin contract are all unchanged. Only the NPZ digest moves. The provenance of the
    # derivation goes in a sidecar rather than into the manifest, because adding a field
    # to the manifest makes the asset unloadable, which is the contract working.
    manifest = json.loads(source_json.read_text())
    digest_after = sha256(target.read_bytes()).hexdigest()
    manifest["artifact"] = dict(manifest["artifact"]) | {"npz_sha256": digest_after}
    (arguments.out / "neutral-body.json").write_text(json.dumps(manifest, indent=2))
    (arguments.out / "d1-facing-fix.json").write_text(json.dumps({
        "derivation": "tools/compare/d1_asset_relabel.py",
        "what_changed": "which AutoAnim joint name each MPFB bone answers to",
        "what_did_not": ("vertices_m, triangles, joint_weights values, parents, and every "
                         "field the manifest describes"),
        "derived_from": str(source_npz.relative_to(ROOT)),
        "derived_from_sha256": sha256(source_npz.read_bytes()).hexdigest(),
        "asset_npz_sha256": digest_after,
        "joint_map": dict(DETAILED_MPFB_JOINT_MAP),
        "equivalent_to": ("a regeneration under the repaired DETAILED_MPFB_JOINT_MAP; the "
                          "underlying MPFB rig, mesh and weights are identical, so the two "
                          "differ only by which name each bone answers to"),
        "not_a_provider_run": ("this asset was NOT produced by the pinned Blender/MPFB "
                               "worker. `artifact.request_sha256` still binds it to the "
                               "PRE-repair provider request, whose joint map differs. "
                               "Before anything ships, regenerate through the worker and "
                               "check the two agree."),
    }, indent=2))
    print(f"wrote {target}  sha256={digest_after[:16]}")


if __name__ == "__main__":
    main()
