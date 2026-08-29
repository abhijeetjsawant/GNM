"""Skin NVIDIA's SOMA neutral body with a somaskel77 motion, in numpy alone.

Arm (i) of `docs/BATTLE2_SYNTHETIC_TRUTH_FIXTURE.md` needs a renderable human whose
joint positions are known exactly and expressed in SOMA-77's own joint definitions.
`SOMA_neutral.npz` supplies all of it: an 18,056-vertex mesh, a 78-joint rig whose
last 77 joints are somaskel77 in identical order, sparse skinning weights, and both
an A-pose bind and a T-pose.

Two conventions matter and both were measured rather than assumed:

* the mesh is bound in the **A-pose** (`bind_pose_world`), whose left arm points
  along [0.68, -0.70, 0.23], while the motion's rest is a **T-pose** along
  [1, 0, 0]. `t_pose_world` matches the motion's rest to 68 mm median where the
  bind pose is 455 mm out, so poses are composed against the T-pose and the mesh is
  unbound from the A-pose;
* the file is in **centimetres, Y-up**, and the motion is in metres, so the model is
  scaled on load and the caller applies whatever world rotation it already uses.

**Licence.** This asset ships with NVIDIA GEM-X under the NVIDIA Open Model License
and its shape basis is distilled from Triplegangers and SizeUSA scans. Using it to
*evaluate* our own detector is inside how GEM-X is already used here. Using renders
of it as *training data* for a model we ship is a different question and needs the
standing asset gate first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .soma_motion import SOMASKEL77_PARENTS, soma_forward_kinematics, _soma_world_rotations

SOMA_NEUTRAL_PATH = Path(
    ".cache/autoanim_gnm/gem-x/third_party/soma/assets/SOMA_neutral.npz"
)
CENTIMETRES_PER_METRE = 100.0
# Model joint 0 is `Root`, an extra above somaskel77's Hips; joints 1..77 are
# somaskel77 0..76 in identical order (verified by name comparison).
JOINT_OFFSET = 1


class SomaBodyMesh:
    """The neutral body, ready to be posed by a somaskel77 motion."""

    def __init__(self, path: Path | str = SOMA_NEUTRAL_PATH) -> None:
        data = np.load(Path(path), allow_pickle=True)
        self.vertices_bind_m = data["mean"].astype(np.float64) / CENTIMETRES_PER_METRE
        self.triangles = data["triangles"].astype(np.int64)
        self.bind_world = data["bind_pose_world"].astype(np.float64).copy()
        self.t_pose_world = data["t_pose_world"].astype(np.float64).copy()
        for matrices in (self.bind_world, self.t_pose_world):
            matrices[:, :3, 3] /= CENTIMETRES_PER_METRE
        from scipy.sparse import csr_matrix

        self.weights = csr_matrix(
            (data["skinning_weights_data"], data["skinning_weights_indices"],
             data["skinning_weights_indptr"]),
            shape=tuple(int(v) for v in data["skinning_weights_shape"][::-1]),
        ).T.tocsr()   # stored joints-by-vertices; we want vertices-by-joints
        self.joint_names = [str(v) for v in data["joint_names"]]

    # -- the rest pose the motion format expects -----------------------------
    @property
    def rest_positions_m(self) -> np.ndarray:
        return self.t_pose_world[JOINT_OFFSET:, :3, 3]

    @property
    def rest_world_xyzw(self) -> np.ndarray:
        return Rotation.from_matrix(self.t_pose_world[JOINT_OFFSET:, :3, :3]).as_quat()

    def pose(self, root_translation_m: np.ndarray, local_rotations_xyzw: np.ndarray):
        """Return (vertices, joint positions) per frame, both in metres."""

        positions = soma_forward_kinematics(
            root_translation_m, local_rotations_xyzw,
            self.rest_positions_m, self.rest_world_xyzw,
        ).astype(np.float64)
        world = _soma_world_rotations(local_rotations_xyzw, self.rest_world_xyzw)
        frames = positions.shape[0]
        inverse_bind = np.linalg.inv(self.bind_world)
        homogeneous = np.hstack((self.vertices_bind_m, np.ones((len(self.vertices_bind_m), 1))))
        out = np.empty((frames, len(self.vertices_bind_m), 3), dtype=np.float64)
        for frame in range(frames):
            posed = np.tile(np.eye(4), (len(self.bind_world), 1, 1))
            posed[JOINT_OFFSET:, :3, :3] = Rotation.from_quat(world[frame]).as_matrix()
            posed[JOINT_OFFSET:, :3, 3] = positions[frame]
            # The extra Root joint carries no motion; give it the pelvis transform
            # so any vertices weighted to it travel with the body rather than
            # staying at the origin.
            posed[0] = posed[JOINT_OFFSET]
            skinning = posed @ inverse_bind                       # (78,4,4)
            # sum_j w_j (M_j B_j^-1) v, done as a sparse product over joints
            transformed = np.einsum("jab,vb->jva", skinning, homogeneous)[:, :, :3]
            out[frame] = np.asarray(
                sum(self.weights[:, j].toarray() * transformed[j] for j in range(len(posed)))
            )
        return out, positions
