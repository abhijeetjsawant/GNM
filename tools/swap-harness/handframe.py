"""Wrist-local hand frame + gauge-invariant articulation metrics.

Built identically for both systems from *positions only*, so no solved rotation
from either estimator leaks into the comparison.
"""
import numpy as np

def local_frame(wrist, index1, middle1, ring1, pinky1):
    """(F,3) each -> (F,3,3) rotation whose columns are the local axes."""
    knuckles = (index1 + middle1 + ring1 + pinky1) / 4.0
    e1 = knuckles - wrist                       # down the hand
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    across = index1 - pinky1                    # across the palm
    e3 = np.cross(e1, across)                   # palm normal
    e3 /= np.linalg.norm(e3, axis=1, keepdims=True)
    e2 = np.cross(e3, e1)
    return np.stack([e1, e2, e3], axis=2)

def to_local(points, wrist, basis):
    """points (F,N,3) -> wrist-local (F,N,3)."""
    return np.einsum("fij,fnj->fni", np.transpose(basis, (0, 2, 1)), points - wrist[:, None, :])

def metrics(local_tips, hand_length_m):
    """local_tips (F,N,3) in metres. Returns amplitude and jitter, mm and %."""
    sd = np.sqrt((local_tips.std(axis=0) ** 2).sum(axis=1)) * 1000.0     # (N,) mm
    d2 = local_tips[2:] - 2.0 * local_tips[1:-1] + local_tips[:-2]
    jitter = np.median(np.linalg.norm(d2, axis=2), axis=0) * 1000.0      # (N,) mm
    scale = hand_length_m * 1000.0
    return {
        "amplitude_mm": float(sd.mean()),
        "jitter_mm": float(jitter.mean()),
        "amplitude_pct": float(sd.mean() / scale * 100.0),
        "jitter_pct": float(jitter.mean() / scale * 100.0),
        "roughness": float(jitter.mean() / max(sd.mean(), 1e-9)),
        "per_tip_amplitude_mm": sd.tolist(),
        "per_tip_jitter_mm": jitter.tolist(),
        "hand_length_mm": float(scale),
    }


def jitter_mm(points):
    """Median norm of the discrete second difference, per point, in mm."""
    import numpy as _np
    d2 = points[2:] - 2.0 * points[1:-1] + points[:-2]
    return _np.median(_np.linalg.norm(d2, axis=2), axis=0) * 1000.0


def both_frames(wrist, index1, middle1, ring1, pinky1, tips, hand_length_m):
    """Articulation-only and articulation-plus-wrist-rotation, together.

    The wrist-local frame cancels wrist rotation exactly -- the knuckles hang
    rigidly off the wrist, so the basis turns with it. That is what makes it the
    right frame for comparing articulation, and it is also why it is blind to a
    thrashing wrist. The world-axes measurement below removes only wrist
    *translation*, so it sees both.
    """
    import numpy as _np
    basis = local_frame(wrist, index1, middle1, ring1, pinky1)
    articulation = to_local(tips, wrist, basis)
    with_rotation = tips - wrist[:, None, :]
    out = metrics(articulation, hand_length_m)
    out["jitter_with_wrist_mm"] = float(jitter_mm(with_rotation).mean())
    sd = _np.sqrt((with_rotation.std(axis=0) ** 2).sum(axis=1)) * 1000.0
    out["amplitude_with_wrist_mm"] = float(sd.mean())
    return out
