# Battle 2 and the standing asks — what has to leave the building this week

Everything here is **lead time, not engineering time.** None of it competes with
the estimator work; all of it is on the critical path for a claim we cannot make
until it lands. The plan already says Battle 2 runs *in parallel with* Battle 1,
and this is the list that makes "in parallel" real.

The specification is in `OWNED_BODY_CAPTURE_PLAN.md` under Battle 2 and in
research §9b. This document is only the actionable residue: what to book, what to
ask for, and what to say.

---

## 1. The reference capture — book it, then design it

**Why now.** Every millimetre figure this project has produced is reprojection,
held-out reprojection, or agreement with MAMMA. None of them is accuracy. And the
MAMMA fixture has a retirement date built into the plan: its licence bars
producing artifacts for commercial purposes, so it is deleted from the evaluation
path the day Battle 2 delivers. Until then **no accuracy claim in the plan can be
evaluated at all.**

**The three things with real lead time**, in order of how long they take:

| item | why it gates | typical lead |
|---|---|---|
| **marker reference** — rented optical suit, or a hired session in a Vicon/OptiTrack volume | this *is* the ground truth; without it the shoot produces more unmeasurable video | weeks |
| **genlock hardware** | MAMMA's genlocked rig still slipped two frames at 60 fps; the budget is ≤5 ms residual, and a soft-synced shoot cannot be rescued afterwards | days–weeks |
| **performers, with releases covering ML training use** | consent that does not cover training makes the whole capture unusable for the thing it exists for | days, plus scheduling |

**Ask when enquiring about the marker session**, because the answers change the
shoot design: can they export raw marker trajectories rather than a solved
skeleton (a solved skeleton is another vendor's body model and inherits its
conventions and its errors); can our cameras record simultaneously in their
volume; what is their own residual error, so we know the reference's own noise
floor before comparing anything to it.

**The rig spec, from the plan** — these are physical terms, not preferences, and
they are what make 20 mm reachable on four cameras: 60 fps, fast-readout sensors,
1/500–1/1000 s shutter with enough light to support it, cameras *surrounding* the
volume rather than in an arc, board-plus-wand bundle-adjusted calibration every
shoot, fitted wardrobe, and a body scan of each performer.

**Shoot list:** dialogue two-hander, hands near face, seated acting, crossing
identities, occlusion, and one extreme-range-of-motion block — the MOYO gap — for
which a yoga or movement performer for a day, owned outright, is the cheapest fix.

**The exit gate is deliberately not reprojection RMS.** Calibration is validated by
**triangulating known lengths**, because wand-only calibration overfits its own
recording and will happily report a beautiful RMS on a rig that is wrong. Sync
verified ≤5 ms in every take. Marker reference time-aligned. Provenance documented
for the whole set.

---

## 2. Motion corpora — three emails, all low-effort, all latency

Research §9a established that motion data is effectively free but that several of
the useful corpora are **request-gated**, and a request sent today costs nothing
and may take weeks to answer. Send all three now; they are independent.

**KIT Whole-Body Human Motion Database**, **HDM05** (Universität Bonn), and
**Truebones Zoo**. A serviceable template, adjust per recipient:

> Subject: Licence enquiry — commercial use of *[corpus]* for synthetic training data
>
> Hello,
>
> I am building an in-house markerless motion-capture system and would like to
> understand whether *[corpus]* can be used commercially.
>
> Specifically: we would retarget the motion onto our own body model, render
> synthetic images from it, and train a pose-estimation model on those renders,
> which we would then use in commercial production. The motion files themselves
> would never be redistributed in any form, converted or otherwise — only the
> resulting renders and model weights.
>
> Is that within the terms you grant? If it requires a separate commercial
> licence, I would be glad to discuss one.
>
> Thank you,
> *[name and affiliation]*

The "never redistributed, even converted" sentence is not filler — CMU's terms
specifically bar reselling motion "even in converted form", and stating the
constraint up front is what makes the answer usable.

**EgoSuite-Open100K** (LightwheelAI, on HuggingFace) needs the same request plus
an access approval. When the licence arrives, read it against exactly four
questions: does it permit commercial use; does it permit *training* on the data;
does it permit shipping a model trained on it; and does it impose attribution or
copyleft on the outputs. Anything less than a clear yes on the first three makes
it unusable for us regardless of how good the data is.

---

## 3. NVIDIA — two separate things, do not conflate them

**The backbone question.** GEM-X's SOMA-77 is the detector we currently depend on,
and how its backbone was initialised matters, because a DINOv3 initialisation
carries licence terms of its own. Ask NVIDIA directly rather than inferring it
from the checkpoint.

**The attribution obligation.** SOMA-77 ships under the NVIDIA Open Model License,
which carries attribution requirements. This needs to be on the ship checklist
*now*, while the dependency is fresh, not discovered during a release.

---

## 4. Freedom-to-operate counsel

Research §8 found two live patents: SMPL, US10395411B2, running to roughly 2036,
and Brown, US9189886B2, to roughly 2032. The clean-room design routes around
SMPL-X by using MHR, but "we think we route around it" is an engineering opinion,
not a legal one, and the difference matters precisely when the product ships.

Engage counsel with the specific question — not "are we clear", which no one will
answer, but: does fitting an articulated body model with fixed bone lengths to
multi-view 2D landmarks, using MHR's parameterisation, read on either claim set.
Give them `OWNED_BODY_CAPTURE_RESEARCH.md` §8 as the starting brief; the
archaeology is already done.

---

## Why this list is short

Each item is a message or a booking. None of it is a day's work. All of it has a
queue in front of it that only starts moving once it is sent — and the estimator
work continues at full speed regardless, which is the entire point of doing this
in parallel rather than after.
