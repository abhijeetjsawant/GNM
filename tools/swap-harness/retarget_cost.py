#!/usr/bin/env python3
"""What does stage 6 cost -- the positions-to-rotations retarget?

`positions_to_body_track` consumes only bone DIRECTIONS and rebuilds the body on
DETAILED_HUMANOID's **canonical bone lengths**. The performer's own limb lengths
are discarded. So every delivered joint inherits the difference between this rig's
proportions and the subject's -- permanently, in every frame, however good the
detector becomes. This sizes that, isolating the stage by feeding it the
pipeline's own triangulated positions and reading what comes back out.

Two things had to be got right, and both were found by a control that failed:

1. The stage is NOT just a retarget. It ends in `project_generated_foot_contacts`
   (ground_band 0.08 m, maximum_root_correction_m 0.08), which plants the character
   on the floor. That is a deliberate global vertical shift -- ~90 mm here -- and
   it is not error. Everything below is therefore scored ROOT-RELATIVE.

2. "Hips must return the pelvis" is not a usable control, because root-relative it
   is identically zero. The control that CAN fail is a ROUND TRIP: take the rig's
   own output, forward-kinematic it into landmark positions, and feed those back
   in. That body has canonical proportions BY CONSTRUCTION, so the converter must
   reproduce it. Whatever it costs there is the converter's own approximation; the
   excess on a real performer is the proportion mismatch.

   arm A  round trip on a canonical-proportioned body  -> converter error
   arm B  the real performer                           -> converter + proportions
   B - A                                               -> the cost of not fitting a body
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from autoanim_gnm import commercial_multiview as cm
from autoanim_gnm.body import DETAILED_HUMANOID, forward_kinematics_positions

# Which rig joint carries which captured landmark, read off the chain table in
# positions_to_body_track: each chain rotates the parent so the child's REST
# offset points along the measured direction, so the CHILD's origin is the landmark.
RIG_FOR = {
    "neck": "Neck",
    "left_shoulder": "LeftUpperArm",  "right_shoulder": "RightUpperArm",
    "left_elbow": "LeftLowerArm",     "right_elbow": "RightLowerArm",
    "left_wrist": "LeftHand",         "right_wrist": "RightHand",
    "left_hip": "LeftUpperLeg",       "right_hip": "RightUpperLeg",
    "left_knee": "LeftLowerLeg",      "right_knee": "RightLowerLeg",
    "left_ankle": "LeftFoot",         "right_ankle": "RightFoot",
}
BONES = [("left_shoulder", "left_elbow", "LeftLowerArm"), ("left_elbow", "left_wrist", "LeftHand"),
         ("right_shoulder", "right_elbow", "RightLowerArm"), ("right_elbow", "right_wrist", "RightHand"),
         ("left_hip", "left_knee", "LeftLowerLeg"), ("left_knee", "left_ankle", "LeftFoot"),
         ("right_hip", "right_knee", "RightLowerLeg"), ("right_knee", "right_ankle", "RightFoot")]
# Rig child whose rest offset IS that bone, paired with the limb the pipeline
# already measures per subject per take in `estimate_limb_lengths_m`.
SCALE_FROM = {
    "LeftLowerArm": ("left_shoulder", "left_elbow"), "LeftHand": ("left_elbow", "left_wrist"),
    "RightLowerArm": ("right_shoulder", "right_elbow"), "RightHand": ("right_elbow", "right_wrist"),
    "LeftLowerLeg": ("left_hip", "left_knee"), "LeftFoot": ("left_knee", "left_ankle"),
    "RightLowerLeg": ("right_hip", "right_knee"), "RightFoot": ("right_knee", "right_ankle"),
}
TORSO_CHAIN = ("Spine", "Chest", "UpperChest", "Neck")
SHA = "0" * 64


def scaled_skeleton(src_z_up, torso=True):
    """DETAILED_HUMANOID with this performer's OWN bone lengths substituted in.

    The pipeline already measures them -- `estimate_limb_lengths_m`, one set per
    subject per take -- and then `positions_to_body_track` throws them away for the
    canonical rest offsets. This puts them back, changing lengths only: directions,
    joint count, names and parents are untouched, so every downstream contract on
    the 55-joint rig still holds. The clavicle anchor defect is deliberately LEFT IN,
    so bone length and that defect stay separable.
    """
    import dataclasses
    lengths = cm.estimate_limb_lengths_m(src_z_up)
    joints = list(DETAILED_HUMANOID.joints)
    for name, key in SCALE_FROM.items():
        if key not in lengths:
            continue
        i = DETAILED_HUMANOID.index(name)
        rest = np.asarray(joints[i].rest_translation_m, dtype=np.float64)
        n = np.linalg.norm(rest)
        if n < 1e-9:
            continue
        joints[i] = dataclasses.replace(
            joints[i], rest_translation_m=tuple(rest / n * lengths[key]))
    # Torso: scale the whole pelvis->neck chain to the measured span.
    pelvis = 0.5 * (src_z_up[:, cm.JOINT_INDEX["left_hip"]] + src_z_up[:, cm.JOINT_INDEX["right_hip"]])
    span = float(np.median(np.linalg.norm(src_z_up[:, cm.JOINT_INDEX["neck"]] - pelvis, axis=1)))
    canon = sum(float(np.linalg.norm(DETAILED_HUMANOID.joints[DETAILED_HUMANOID.index(n)].rest_translation_m))
                for n in TORSO_CHAIN)
    # The scoring origin is the HIP MIDPOINT, which hangs below Hips by the upper-leg
    # rest offset, while the chain sum measures Hips->Neck. Scaling by span/canon
    # therefore forces hipmid->Neck = span + drop, a guaranteed overshoot -- it showed
    # up as arm C's neck median sitting at exactly 80.00 mm on both subjects, and it
    # made torso scaling look harmful when it is not. Caught by Fable.
    drop = -0.5 * (np.asarray(DETAILED_HUMANOID.joints[DETAILED_HUMANOID.index("LeftUpperLeg")].rest_translation_m)[1]
                   + np.asarray(DETAILED_HUMANOID.joints[DETAILED_HUMANOID.index("RightUpperLeg")].rest_translation_m)[1])
    if canon > 1e-9 and torso:
        k = (span - drop) / canon
        for n in TORSO_CHAIN:
            i = DETAILED_HUMANOID.index(n)
            joints[i] = dataclasses.replace(
                joints[i], rest_translation_m=tuple(np.asarray(joints[i].rest_translation_m) * k))
    return dataclasses.replace(DETAILED_HUMANOID, joints=tuple(joints))
Z_UP_FROM_Y_UP = lambda p: np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)
Y_UP_FROM_Z_UP = lambda p: np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def retarget_then_fk(src_z_up, skeleton=None):
    skel = skeleton or DETAILED_HUMANOID
    saved = cm.DETAILED_HUMANOID
    cm.DETAILED_HUMANOID = skel                    # the retarget reads it from its own namespace
    try:
        bt = cm.positions_to_body_track(src_z_up, sample_rate_hz=30, provenance_sha256=SHA)
    finally:
        cm.DETAILED_HUMANOID = saved
    return forward_kinematics_positions(
        np.asarray(bt.root_translation_m, dtype=np.float64),
        np.asarray(bt.local_rotations_xyzw, dtype=np.float64),
        skeleton=skel).astype(np.float64)


def score(fk, ref_y_up, skeleton=None):
    """Root-relative per-landmark error, mm. Removes the ground-projection shift."""
    skel = skeleton or DETAILED_HUMANOID
    fk_o = 0.5 * (fk[:, skel.index("LeftUpperLeg")] + fk[:, skel.index("RightUpperLeg")])
    rf_o = 0.5 * (ref_y_up[:, cm.JOINT_INDEX["left_hip"]] + ref_y_up[:, cm.JOINT_INDEX["right_hip"]])
    out = {}
    for name, rig in RIG_FOR.items():
        a = fk[:, skel.index(rig)] - fk_o
        b = ref_y_up[:, cm.JOINT_INDEX[name]] - rf_o
        out[name] = np.linalg.norm(a - b, axis=1) * 1000
    return out


def landmarks_from_fk(fk):
    """Rebuild a 19-joint observation set from a canonical-rig FK pose."""
    out = np.zeros((len(fk), len(cm.JOINT_NAMES), 3))
    for name, rig in RIG_FOR.items():
        out[:, cm.JOINT_INDEX[name]] = fk[:, DETAILED_HUMANOID.index(rig)]
    head = fk[:, DETAILED_HUMANOID.index("Head")]
    for name in cm.JOINT_NAMES:                      # unused by the retarget, must be finite
        i = cm.JOINT_INDEX[name]
        if not out[:, i].any():
            out[:, i] = head
    out[:, cm.JOINT_INDEX["root"]] = 0.5 * (out[:, cm.JOINT_INDEX["left_hip"]] + out[:, cm.JOINT_INDEX["right_hip"]])
    return out


print(f"rig DETAILED_HUMANOID, {len(DETAILED_HUMANOID.joints)} joints, canonical rest lengths")
print("scored ROOT-RELATIVE (hip midpoint), so the ground projection is removed\n")
for subject in (0, 1):
    p = ROOT / f"artifacts/commercial-multiview-soma77/subject-0{subject}.body-track.npz"
    if not p.exists():
        print(f"missing {p}"); continue
    src = np.load(p)["triangulated_world_positions_z_up_m"]
    src = src[np.isfinite(src).all(axis=(1, 2))]

    fk_b = retarget_then_fk(src)                               # arm B, real performer
    err_b = score(fk_b, Y_UP_FROM_Z_UP(src))

    synth_y = landmarks_from_fk(fk_b)                          # a canonical-proportioned body
    fk_a = retarget_then_fk(Z_UP_FROM_Y_UP(synth_y))           # arm A, round trip
    err_a = score(fk_a, synth_y)

    skel_c = scaled_skeleton(src, torso=True)                  # arm C, own bones + torso
    fk_c = retarget_then_fk(src, skeleton=skel_c)
    err_c = score(fk_c, Y_UP_FROM_Z_UP(src), skeleton=skel_c)
    skel_d = scaled_skeleton(src, torso=False)                 # arm D, own LIMB bones only
    fk_d = retarget_then_fk(src, skeleton=skel_d)
    err_d = score(fk_d, Y_UP_FROM_Z_UP(src), skeleton=skel_d)

    allA = np.concatenate(list(err_a.values())); allB = np.concatenate(list(err_b.values()))
    ok = np.median(allA) < 1.0
    print(f"=== subject {subject}: {len(src)} frames ===")
    allC = np.concatenate(list(err_c.values())); allD = np.concatenate(list(err_d.values()))
    print(f"{'landmark':>16} {'A round-trip':>13} {'B canonical':>12} {'C +torso':>10} {'D limbs only':>13} {'D recovers':>11}")
    print("-" * 82)
    for name in RIG_FOR:
        a_, b_ = np.median(err_a[name]), np.median(err_b[name])
        c_, d_ = np.median(err_c[name]), np.median(err_d[name])
        rec = 100.0 * (b_ - d_) / (b_ - a_) if (b_ - a_) > 1e-6 else float("nan")
        print(f"{name:>16} {a_:>10.2f} mm {b_:>9.1f} mm {c_:>7.1f} mm {d_:>10.1f} mm {rec:>10.0f}%")
    print(f"{'ALL median':>16} {np.median(allA):>10.2f} mm {np.median(allB):>9.1f} mm {np.median(allC):>7.1f} mm "
          f"{np.median(allD):>10.1f} mm "
          f"{100.0*(np.median(allB)-np.median(allD))/max(np.median(allB)-np.median(allA),1e-6):>10.0f}%")
    mA, mB = np.mean(allA), np.mean(allB)
    mC, mD = np.mean(allC), np.mean(allD)
    print(f"{'ALL mean':>16} {mA:>10.2f} mm {mB:>9.1f} mm {mC:>7.1f} mm {mD:>10.1f} mm "
          f"{100.0*(mB-mD)/max(mB-mA,1e-6):>10.0f}%")
    print(f"  Report the MEAN, not the pooled median: the landmark population is bimodal")
    print(f"  (legs ~25 mm, arms ~180 mm), so a pooled median jumps the gap on one subject")
    print(f"  and sits pinned on the other. The median row is retained only to show that.")
    print(f"  D = the performer's own LIMB bone lengths. C additionally rescales the spine.")
    print(f"  CONTROL arm A (canonical body must round-trip): {'PASS' if ok else 'FAIL'} "
          f"at {np.median(allA):.2f} mm median")

    span_c = sum(np.linalg.norm(DETAILED_HUMANOID.joints[DETAILED_HUMANOID.index(n)].rest_translation_m)
                 for n in ("Spine", "Chest", "UpperChest", "Neck")) * 1000
    span_s = np.median(np.linalg.norm(src[:, cm.JOINT_INDEX["neck"]] - 0.5 * (
        src[:, cm.JOINT_INDEX["left_hip"]] + src[:, cm.JOINT_INDEX["right_hip"]]), axis=1)) * 1000
    print(f"\n  torso span pelvis->neck: canonical {span_c:.0f} mm, performer {span_s:.0f} mm, "
          f"delta {span_s-span_c:+.0f} mm  (every arm joint hangs off this)")
    print(f"  {'bone':>28} {'canonical':>10} {'performer':>10} {'delta':>9}")
    for a, b, rigchild in BONES:
        canon = np.linalg.norm(DETAILED_HUMANOID.joints[DETAILED_HUMANOID.index(rigchild)].rest_translation_m) * 1000
        subj = np.median(np.linalg.norm(src[:, cm.JOINT_INDEX[b]] - src[:, cm.JOINT_INDEX[a]], axis=1)) * 1000
        print(f"  {a+'->'+b:>28} {canon:>7.0f} mm {subj:>7.0f} mm {subj-canon:>+6.0f} mm")
    print()
