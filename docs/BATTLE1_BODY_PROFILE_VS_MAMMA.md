# Where our body track disagrees with MAMMA's, joint by joint

MAMMA ran on this fixture. Its fit is retained. Both systems solve the same four
cameras and the same 150 frames into the same calibrated world frame, so they can
be compared directly, per joint, per frame — no registration, no scaling.

**This cannot produce an accuracy number.** MAMMA is not ground truth; it is a
system that scores 13.5 mm single-person and 20 mm two-person against Vicon on
its own benchmarks. What the comparison can do is say *where* we disagree, and
separate two very different reasons for it:

- a **constant offset** is a joint-definition difference — the two skeletons
  simply do not put "the hip" in the same place, and we have already found two of
  these (MHR's hand root sits 2.5–3 cm forward of SMPL-X's wrist; Battle 0 found
  Apple Vision and MediaPipe disagreeing by ~10% of limb length);
- the **spread around that offset** is real per-frame disagreement, and it is the
  only part a better estimator can fix.

Subject pairing came from wrist separation and is unambiguous: 19 and 29 mm for
the true pairing against 1.19 and 1.20 m for the swap. Our subject 0 is MAMMA's
`body_id-01`.

## The whole skeleton carries a 16 mm vertical offset

Before reading anything per-joint: the median bias vector across all fourteen body
joints is **[0.6, 2.2, 16.2] mm**. Almost pure +Z. Our skeleton sits 16 mm higher
than MAMMA's, everywhere, all the time.

That is one number for the whole rig, not fourteen, and it is the kind of thing a
floor-contact term or a pelvis-height convention produces. Removing it drops the
mean per-joint bias from 25.0 mm to 16.7 mm — a third of our apparent systematic
disagreement with MAMMA was a single global constant.

## Per joint, after the global offset is removed

| joint | bias | | joint | bias |
|---|---:|---|---|---:|
| **right_elbow** | **1.2 mm** | | left_hip | 17.1 mm |
| **left_knee** | **4.8 mm** | | right_hip | 19.8 mm |
| **right_wrist** | **5.3 mm** | | root | 26.1 mm |
| **right_shoulder** | **6.1 mm** | | right_ankle | 27.4 mm |
| right_knee | 7.5 mm | | left_ankle | 27.6 mm |
| left_elbow | 10.2 mm | | **neck** | **58.1 mm** |
| left_shoulder | 10.9 mm | | | |
| left_wrist | 12.3 mm | | | |

**The arm chain agrees with MAMMA to between 1 and 12 mm of systematic bias.**
Shoulders, elbows and wrists — the joints that carry acting — are the best-agreeing
joints in the skeleton. That is the strongest evidence so far that the clean-room
reconstruction is sound, and it is independent of every reprojection number, which
can only ever measure agreement with our own detector.

The outliers are all where the two skeletons are known to differ:

- **neck, 58 mm** — SOMA-77 predicts `Neck2`, SMPL-X's joint 12 is a different
  vertebra. Definitional, and the head is GNM's anyway.
- **ankles, 27 mm** — SOMA's `LeftFoot` against SMPL-X's `left_ankle`.
- **root and hips, 17–26 mm** — pelvis conventions differ between every skeleton
  ever published.

## Per-frame spread — the part that is actually ours to fix

| joint | spread |
|---|---:|
| ankles | 19–20 mm |
| knees | 18 mm |
| elbows, wrists | 17–19 mm |
| shoulders | 23–26 mm |
| root | 23 mm |
| hips | 27–34 mm |
| **body mean** | **24.7 mm** |

This is the number that matters. Bias is a mapping problem — a retargeting
calibration absorbs it, and it costs nothing at runtime. Spread is noise, and no
amount of skeleton alignment removes it.

**About 18 mm of per-frame spread on the arm chain, against a system that is
13.5–20 mm from Vicon.** By the triangle inequality that puts our true arm error
somewhere between roughly 0 and 38 mm, most plausibly in the low twenties — but
the honest statement is that *we still cannot measure it*, because both terms in
that comparison are estimates and only one of them has ever been checked against
markers.

## What this changes

Three things.

**Our apparent error was substantially bias, and bias is cheap.** 25.0 mm of mean
systematic disagreement was 16.4 mm of one global constant plus a set of joint
definitions we can enumerate. The reconstruction is closer to MAMMA than the raw
per-joint distances suggested.

**The remaining lever on the body is per-frame spread, not systematics.** Roughly
18 mm on the arm chain. That is detector noise and estimator noise, in unknown
proportion — which is precisely the question the next increment has to answer
rather than assume.

**Head landmarks are meaningless and should stop being reported.** Nose and ears
carry 114–125 mm of bias because SOMA-77 has no ear landmarks at all and no nose;
the worker substitutes `Head`, and documents that it is an approximation. Nothing
downstream uses them — the head is GNM's — but they inflate every whole-skeleton
mean we quote, and this document's body-only figures exclude them for that reason.
