#!/usr/bin/env python3
"""All FOUR gate arms at native detector width. Completes §6s's pre-registration.

§6u reported the candidate and oracle at 3840 but not the controls. "No gate a constant can
pass" means a published gate result carries its controls, so this runs `head_gate.main()`
itself against the native artifacts -- the same gate, repointed, never a copy.

Everything the gate reads is redirected: the tracks it takes its torso frame from, the
association it triangulates under, and the head solve it scores. The MAMMA reference is
unchanged, as it must be for the two widths to share a denominator.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import associate  # noqa: E402
import head_gate as hg  # noqa: E402
import solve_head  # noqa: E402
import triangulate_soma  # noqa: E402
from autoanim_gnm.commercial_multiview import (  # noqa: E402
    JOINT_INDEX, _thorax_frames, load_camera_rig, load_observation_jsonl,
)
from autoanim_gnm.head_orientation import solve_head_orientation  # noqa: E402

NATIVE = Path("artifacts/commercial-multiview-native")
ASSOC = Path("artifacts/head-lane/native-width_3840")
H = {"Head": 6, "HeadEnd": 7, "Jaw": 8, "LeftEye": 9, "RightEye": 10}


def load():
    rig = {c.name: c for c in load_camera_rig(NATIVE / "camera-rig.json")}
    obs = [load_observation_jsonl(NATIVE / "work" / f"{n}-soma77-observations.jsonl")
           for n in associate.CAMERAS]
    w, h = obs[0][0]["width"], obs[0][0]["height"]
    return [rig[n].scaled(w, h) for n in associate.CAMERAS], obs


def main() -> None:
    # redirect every path the gate and its helpers read
    associate.load = load
    associate.RIG, associate.WORK = NATIVE / "camera-rig.json", NATIVE / "work"
    associate.TRACKS, associate.OUT = NATIVE, ASSOC
    triangulate_soma.load, triangulate_soma.OUT = load, ASSOC
    solve_head.load = load
    for mod in (solve_head,):
        if hasattr(mod, "OUT"):
            mod.OUT = ASSOC
    hg.TRACKS = NATIVE
    hg.triangulate = triangulate_soma.triangulate
    hg.initialise = solve_head.initialise

    # the head solve on native data, in the format the gate scores
    cams, raw = load()
    head = [[[np.asarray([p["landmarks_soma77"][i] for i in H.values()], float)
              for p in r["people"]] for r in c] for c in raw]
    asg = np.load(ASSOC / "association.npz")["assignment"]
    payload = {}
    for s in (0, 1):
        sm = np.load(NATIVE / f"subject-{s:02d}.body-track.npz")["triangulated_world_positions_z_up_m"]
        obs = np.full((len(raw[0]), len(associate.CAMERAS), len(H), 3), np.nan)
        for f in range(len(raw[0])):
            for c in range(len(associate.CAMERAS)):
                p = int(asg[f, s, c])
                if p >= 0:
                    obs[f, c] = head[c][f][p]
        sol = solve_head_orientation(cams, obs, tuple(H), thorax_world=_thorax_frames(sm),
                                     neck_origin_world_m=sm[:, JOINT_INDEX["neck"]])
        payload[f"subject_{s:02d}_head_world"] = sol.rotations_world
        payload[f"subject_{s:02d}_head_position_m"] = sol.positions_world_m
        payload[f"subject_{s:02d}_template_m"] = sol.template_m
        print(f"native solve subject {s}: weight {sol.temporal_weight:g}, {sol.reprojection_px:.4f} px")
    payload["gauge_applied"] = np.asarray([1])
    ASSOC.mkdir(parents=True, exist_ok=True)
    np.savez(ASSOC / "head-solve-native.npz", **payload)

    sys.argv = ["head_gate", str(ASSOC / "head-solve-native.npz")]
    hg.main()


if __name__ == "__main__":
    main()
