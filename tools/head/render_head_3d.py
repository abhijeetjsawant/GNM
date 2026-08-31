#!/usr/bin/env python3
"""Rebuild the 3D viewer's scene data FROM THE DELIVERED ARTIFACTS.

The viewer was first written from an in-memory run, and this lane has twice published a
number that the file a user opens did not carry (§6f, §6j). So the scene it renders is
read here from the shipped track and the shipped solve, and from nothing else:

  joints       artifacts/commercial-multiview-soma77/subject-XX.body-track.npz
  head axes    artifacts/head-lane/head-solve-shipped.npz   (the solve the gate scored)
  locked axes  the torso frame, via the pipeline's own `_thorax_frames`

`turn_deg` is the head's rotation RELATIVE TO THE TORSO with the take mean removed. Both
qualifiers are load-bearing. World rotation would be dominated by the body turning, and
without mean-removal the number is the anatomical gauge -- roughly 104 deg on this rig --
rather than any head motion. An earlier caption printed exactly that constant.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoanim_gnm.commercial_multiview import _thorax_frames  # noqa: E402
from head_gate import mean_removed  # noqa: E402  -- ONE definition, imported not copied

TRACKS = Path("artifacts/commercial-multiview-soma77")
SHIPPED = Path("artifacts/head-lane/head-solve-shipped.npz")
PAGE = Path("docs/head-solve-3d.html")


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    rel = np.einsum("nji,njk->nik", a, b)
    trace = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def main() -> None:
    solved = np.load(SHIPPED)
    page = PAGE.read_text(encoding="utf-8")
    scene = json.loads(
        re.search(r'<script id="scene-data" type="application/json">(.*?)</script>', page, re.S).group(1)
    )
    for subject in range(2):
        track = np.load(TRACKS / f"subject-{subject:02d}.body-track.npz")
        joints = track["triangulated_world_positions_z_up_m"]
        thorax = _thorax_frames(joints)
        head = solved[f"subject_{subject:02d}_head_world"]
        origin = solved[f"subject_{subject:02d}_head_position_m"]
        ok = np.isfinite(joints).all(axis=(1, 2)) & np.isfinite(head).all(axis=(1, 2))
        # head relative to the torso, gauge removed on the SCORED population
        relative = np.einsum("nji,njk->nik", np.nan_to_num(thorax, nan=0.0), head)
        turn = np.full(len(head), np.nan)
        deviation = mean_removed(relative, ok)
        turn[ok] = geodesic_deg(
            deviation[ok], np.broadcast_to(np.eye(3), deviation[ok].shape)
        )
        scene["subjects"][subject] = {
            "joints": np.nan_to_num(joints, nan=0.0).round(5).tolist(),
            "head_origin": np.nan_to_num(origin, nan=0.0).round(5).tolist(),
            "head_axes": np.nan_to_num(head, nan=0.0).round(6).tolist(),
            "locked_axes": np.nan_to_num(thorax, nan=0.0).round(6).tolist(),
            "turn_deg": np.nan_to_num(turn, nan=0.0).round(3).tolist(),
        }
        print(f"subject {subject}: turn about the take mean -- median "
              f"{np.nanmedian(turn):.2f} deg, p95 {np.nanpercentile(turn, 95):.2f} deg, "
              f"max {np.nanmax(turn):.2f} deg over {int(ok.sum())} frames")
    page = re.sub(
        r'(<script id="scene-data" type="application/json">).*?(</script>)',
        lambda m: m.group(1) + json.dumps(scene, separators=(",", ":")) + m.group(2),
        page, flags=re.S,
    )
    PAGE.write_text(page, encoding="utf-8")
    print(f"wrote {PAGE} ({PAGE.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
