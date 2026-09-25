"""D4: the delivered GLB read back from its own bytes and forward-kinematicked against its track.

D3's closure band, on the MHR delivery: `<= 1e-4 m`. "The delivered file must be read back from
its own bytes; a code-path instrument cannot see what the exporter wrote" (CLAUDE.md) -- the
whole point is that nothing here imports momentum. The glTF parsing primitives are
`tools/compare/d3_skeleton_gate.py`'s (`read_glb`, `accessor`, `_qmul`, `_qrot`), reused rather
than re-implemented; the forward-kinematic loop is written here because momentum's export differs
from the rig exporter's in two ways that gate's loop does not allow for:

  * most joints carry NO rotation channel (45 rotation and 52 translation channels over 127
    joints), so a joint without one must fall back to its node transform rather than KeyError;
  * there is a SCALE channel (momentum writes the centimetre-to-metre conversion as a scale on
    the root), so scale has to propagate down the chain or every position is 100x out.

Frames: glTF is Y-up, the capture world is Z-up, so a position `(x, y, z)` read here is
`(x, -z, y)` in the capture frame -- and the root scale has already put it in metres, which the
band itself verifies rather than assumes.

    .venv/bin/python tools/compare/d4_glb_closure.py --pair NAME=GLB,TRACK [...] --out OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/compare"))
import d3_skeleton_gate as d3  # noqa: E402

BAND_M = 1e-4


def glb_joint_positions(path: Path) -> tuple[list[str], np.ndarray]:
    """Forward-kinematic the GLB's own skin, as a glTF viewer would. Returns names, [f, j, 3].

    Every NODE is walked, not only the skin's joints: momentum parents the skeleton under
    non-joint nodes, so a chain that starts at a skin joint whose parent is outside the skin
    would otherwise lose that parent's transform (and the root scale with it).
    """
    document, binary = d3.read_glb(path)
    nodes = document["nodes"]
    joints = document["skins"][0]["joints"]
    parent = {index: -1 for index in range(len(nodes))}
    for index, node in enumerate(nodes):
        for child in node.get("children", ()):
            parent[child] = index
    animation = document["animations"][0]
    rotation, translation, scale = {}, {}, {}
    for channel in animation["channels"]:
        sampler = animation["samplers"][channel["sampler"]]
        data = d3.accessor(document, binary, sampler["output"]).astype(np.float64)
        {"rotation": rotation, "translation": translation, "scale": scale}[
            channel["target"]["path"]][channel["target"]["node"]] = data
    frames = max(len(v) for v in list(rotation.values()) + list(translation.values()))

    def channel_or_node(table, node, key, default):
        if node in table:
            return np.asarray(table[node], np.float64)
        value = np.asarray(nodes[node].get(key, default), np.float64)
        return np.broadcast_to(value, (frames, len(default)))

    depth = {}
    for index in range(len(nodes)):
        chain, walk = 0, index
        while parent[walk] != -1:
            walk, chain = parent[walk], chain + 1
        depth[index] = chain
    positions = np.zeros((frames, len(nodes), 3))
    world_q = np.zeros((frames, len(nodes), 4))
    world_s = np.ones((frames, len(nodes), 3))
    for index in sorted(range(len(nodes)), key=lambda n: depth[n]):
        local_t = channel_or_node(translation, index, "translation", (0.0, 0.0, 0.0))
        local_q = channel_or_node(rotation, index, "rotation", (0.0, 0.0, 0.0, 1.0))
        local_s = channel_or_node(scale, index, "scale", (1.0, 1.0, 1.0))
        up = parent[index]
        if up == -1:
            positions[:, index] = local_t
            world_q[:, index] = local_q
            world_s[:, index] = local_s
        else:
            positions[:, index] = positions[:, up] + d3._qrot(world_q[:, up],
                                                              world_s[:, up] * local_t)
            world_q[:, index] = d3._qmul(world_q[:, up], local_q)
            world_s[:, index] = world_s[:, up] * local_s
    return [nodes[node]["name"] for node in joints], positions[:, joints]


def to_capture(positions: np.ndarray) -> np.ndarray:
    """glTF Y-up -> the capture's Z-up world. The root's own scale carries the units."""
    return np.stack([positions[..., 0], -positions[..., 2], positions[..., 1]], axis=-1)


def closure(glb: Path, track: Path) -> dict:
    # D4c: the report carries the sha256 of the bytes it MEASURED, so a gate binds the residual to the files by
    # content rather than by a path or a basename (D4b's closure-provenance debt). Hashed before and after the read.
    glb_sha, track_sha = hashlib.sha256(Path(glb).read_bytes()).hexdigest(), \
        hashlib.sha256(Path(track).read_bytes()).hexdigest()
    names, positions = glb_joint_positions(glb)
    captured = to_capture(positions)
    # allow_pickle: our own delivery npz (object arrays; CLAUDE.md).
    data = np.load(track, allow_pickle=True)
    track_names = [str(n) for n in data["joint_names"]]
    expected = np.asarray(data["joint_positions_z_up_m"], np.float64)
    missing = [n for n in track_names if n not in names]
    rows = [names.index(n) for n in track_names if n in names]
    columns = [i for i, n in enumerate(track_names) if n in names]
    frames = min(captured.shape[0], expected.shape[0])
    difference = np.abs(captured[:frames][:, rows] - expected[:frames][:, columns])
    if (hashlib.sha256(Path(glb).read_bytes()).hexdigest(), hashlib.sha256(Path(track).read_bytes()).hexdigest()) \
            != (glb_sha, track_sha):
        raise SystemExit(f"{glb} or {track} changed while it was measured")
    return {"glb": str(glb), "track": str(track), "glb_sha256": glb_sha, "track_sha256": track_sha,
            "frames_in_glb": int(captured.shape[0]),
            "frames_in_track": int(expected.shape[0]),
            "joints_compared": len(rows), "joints_missing_from_the_glb": missing,
            "max_abs_m": float(difference.max()), "p95_abs_m": float(np.percentile(difference, 95)),
            "band_m": BAND_M, "within_band": bool(difference.max() <= BAND_M)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", action="append", required=True,
                        help="NAME=GLB,TRACK")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    report = {"band_m": BAND_M, "pairs": {}}
    for item in arguments.pair:
        name, paths = item.split("=", 1)
        glb, track = paths.split(",", 1)
        report["pairs"][name] = closure(Path(glb), Path(track))
        row = report["pairs"][name]
        print(f"{name}: {row['joints_compared']} joints, {row['frames_in_glb']} frames, "
              f"max {row['max_abs_m']:.3e} m, within {BAND_M:.0e}: {row['within_band']}")
    report["all_within_band"] = all(p["within_band"] for p in report["pairs"].values())
    report["worst_max_abs_m"] = max(p["max_abs_m"] for p in report["pairs"].values())
    Path(arguments.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("all within band:", report["all_within_band"],
          "worst", f"{report['worst_max_abs_m']:.3e} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
