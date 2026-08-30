# Body lane — plan of record

**Snapshot: 2026-08-30.** Written to be picked up cold by a session that does not have
the conversation this came from. Read sections 0–2 before doing anything.

**Section 3 (MEASURED) will rot.** Every entry carries the instrument that produced
it and that instrument's blindness, inline, because cross-references die when the
reader is grepping. If a new measurement contradicts a MEASURED entry, **stop and
re-verify the instrument before proceeding** — that is a tripwire, not advice.

---

## 0. First action, before any engineering

**Ask the user whether the Battle 2 marker session is booked.** If it is not, it
remains the standing blocker and both advisors' rank 0.

Nothing in this lane is ground-truth-verified. Every number below is relative to
MAMMA's retained output, or to synthetic truth that contains none of calibration
error, lens distortion, sync error, soft-tissue artefact or joint-definition error —
all five first-order on real footage. **The marker session is the only path to an
accuracy claim.** Checklist: `docs/BATTLE2_ACTION_CHECKLIST.md`. The three items with
real lead time are the marker reference (weeks), genlock (days–weeks) and performer
releases covering ML training use.

---

## 1. Standing rules — instructions, not history

**No gate a constant can pass.** Every acceptance band ships with a demonstration
that a degenerate solution fails it. Four constructs in this lane could not fail: a
bone-length gate reading 0.00% because bone lengths are constants in that estimator;
a jitter gate reading better-than-MAMMA on a frozen rest-pose hand; a control
comparing against a *constant* σ, which is an algebraic identity; and a `both` arm
byte-identical to the arm it was meant to be compared against.

**Same denominator.** Score both arms of a comparison on the same population. A
composition shift has masqueraded as an effect twice — most recently a visibility
channel "worth 1.3 mm" that was worth 0.16 mm once both arms were scored on identical
landmarks.

**The recurring defect is a correct measurement carrying a claim it does not
support.** Seven instances so far. The measurement is usually right; the sentence
attached to it is where the error lives. Before writing a conclusion, ask what the
instrument is structurally blind to.

**Confirm a suspected defect with a second, independent measurement before acting.**

---

## 2. What this lane is trying to do

A commercially clean, in-house markerless multi-camera body capture matching MAMMA:
~13.5 mm MPJPE single-person, ~20 mm two-person, against Vicon. MAMMA and SMPL-X are
non-commercial; hence the clean-room rebuild. Body model is Meta MHR (Apache-2.0,
weights included). Full context: `docs/OWNED_BODY_CAPTURE_PLAN.md` and
`docs/OWNED_BODY_CAPTURE_RESEARCH.md`.

---

## 3. MEASURED — with instrument and blindness inline

**The gap is the detector, not the geometry.**
Our `triangulate_point`, unchanged, fed MAMMA's 2D lands **8.9 mm** from MAMMA's own
512 landmarks (exact `verts_512` regressor), **10.2 mm** at full coverage with no
visibility gating. Fed *our* detector's 2D, the same geometry is out by 24.7 mm of
spread plus 33.9 mm of per-joint bias.
*Instrument:* substitution, `tools/swap-harness/swap_true.py`.
*Blindness:* **saturates at ~10 mm** — MAMMA's own fit sits 9.3–11.7 mm from its own
triangulation, so nothing below ~10 mm against this reference is readable. Stops at
triangulation; never exercises `solve_sequence_positions`.

**The detector's error tail is the story: ours is 7.5× MAMMA's.**
MAMMA's 2D residual (visibility ≥ 0.5, native 3840): 1.44 / 2.50 / **4.29** / 6.99 /
10.37 / **13.08** px. Ours scaled to 3840: 1.59 / 4.11 / **9.33** / 18.10 / 34.50 /
**98.40**.
*Instrument:* `tools/swap-harness/mamma_residuals.py`.
*Blindness:* different statistics — MAMMA's is a fit residual, deflated because its
fit consumed the landmarks it is scored against. **The median ratio is unfair to us
by an unknown amount; the tail ratio is too large to be entirely artefact.**

**Triangulation has little headroom left.**
Single-frame Cramér-Rao bound on this rig is 33.2 mm at four clean views; we measure
16.4 mm — half the memoryless bound, because temporal and limb priors aggregate
across frames.
*Instrument:* `tools/swap-harness/crlb.py`. *Blindness:* single-frame; says nothing
about the sequence solve.

**Resolution is not a lever.** 1280 → 3840 gives 14.5 → 14.8 px spread. At 1280 the
person crop is already 286 px against the model's 256 px input, so it is downsampled
either way.
*Instrument:* `tools/swap-harness/res_ablation.py`, spread only, native-pixel
normalised. *Blindness:* spread only; bias not comparable. **Would still matter for
hands, at ~20 px.**

**σ and visibility cannot be ranked.** Corrected pricing: baseline 11.62 mm
(mixture) / 14.38 (lognormal); a σ head buys +0.95 / **+2.15**; a visibility head
buys **+2.75** / +1.19; a *wrong* channel costs −3.6 under both. **The ordering
reverses between two noise models that both fit our data.** On real data a visibility
channel used as a weight is worth **+0.16 mm** on identical landmarks.
*Instrument:* synthetic fixture + the paired real-data comparison.
*Consequence:* **Battle 4 keeps MammaNet's full μ/σ/visibility triple.** The
reordering that briefly appeared in `OWNED_BODY_CAPTURE_PLAN.md` is withdrawn.

**SAM 3D Body is integrated, licence-clean, and useful only as an orientation
source.**
Runs at ~45.8 s/person-frame on CPU. Emits `pred_joint_coords` on **exactly our
127-joint MHR rig** (checked), `pred_global_rots`, `hand_pose_params`, and accepts our
calibrated intrinsics.
- *Fusing four per-view fits:* **do not build.** 118.2 mm cross-view disagreement,
  67% along the viewing ray but **42.8 mm transverse**, which fusion cannot fix;
  averaging four leaves ~21 mm against our existing ~10 mm.
  (`tools/swap-harness/sam3d_fusion_conditioning.py`)
- *Bone direction:* raw 6.9° median / 72.1° p90; after cross-view majority at 15°,
  **3.2° median / 6.9° p90 on 74% of bone-frames**. At a 0.25 m forearm, 3.2° ≈ 13 mm.
  **74% is n=160 bone-frames — a ±8-point binomial band. Do not treat it as precise.**
  (`tools/swap-harness/sam3d_orientation_agreement.py`)
- *Twist:* **not usable.** 10.7° median, 139.7° p90, forearms 22–26° — the textbook
  monocular blind spot. **The orientation source is scoped to direction only.**
*Blindness of both orientation results:* reference-free. Four monocular fits can agree
and be wrong together through shared appearance bias. **Agreement is a necessary
condition, never an accuracy claim.**

---

## 4. WITHDRAWN — do not re-derive these

Each was published and then refuted **by our own follow-up measurement**. They are
listed so a fresh session does not rediscover them as findings.

| withdrawn claim | why |
|---|---|
| "SAM 3D's 2D is 1.83× worse, 2.70× in the tail" | compared its 10-frame ladder against SOMA's hard-coded **full-take** row. On identical frames: 1.42× median, and **0.90/0.94 in the tail — SAM 3D is slightly better**. Both tails are association-contaminated |
| "the epipolar ladder can rank detector families" | it sees only the **incoherent** component. SOMA's error is 18.6 px coherent bias + 16.6 px spread — bias is the larger term and passes through triangulation untouched |
| "the temporal smoother helps on real footage" | scored against MAMMA's *fitted* mesh, which is 2.6–2.9× smoother than its own raw triangulation. The test rewards smoothing. **G10's re-reading survives; the number does not** |
| "μ 10.4 mm / visibility 4.5 / σ 0.4" | noise fitted to `_epipolar_distance_px`, which returns the **symmetric** distance, as if one-sided — 2.00× overstatement |
| "visibility is worth 1.3 mm" | cross-denominator. 0.16 mm on identical landmarks |
| "SOMA-77 carries a 61 mm body-fixed offset" | true in the synthetic domain only; **does not transfer** to real footage (cosine +0.039) |
| "SAM 3D is promptable with our 2D" | `process_one_image` has **no keypoint-prompt parameter**; the internal sampler uses the model's *own* predictions |
| "the estimator has little headroom" | scoped to **triangulation**; the sequence solve was never in the swap |

---

## 5. OPEN — not measured, do not assume

- **Absolute accuracy of anything.** Gated on Battle 2.
- **Detector bias against truth on real footage.** The synthetic fixture cannot
  measure it — flat-shaded renders do not transfer.
- **Whether bone directions actually improve the reconstruction.** Consuming them
  means adding an **orientation term to a position-only solver** — a real estimator
  change with its own increment discipline, not a data hookup.
- **Forearm twist**, which nothing available estimates well.
- **Association under contact.** MAMMA's pre-associated 2D is a truth-grade reference
  for it and has not been used.

---

## 6. Build sequence

Each step names what it changes, what measures it, the pre-registered band, and the
degenerate solution that band must reject. **Do not skip the degenerate check** — four
gates in this lane could not fail, and each cost a day of false confidence.

### Step 1 — association under contact *(measure, do not build)*

MAMMA's retained `ma_2d` is **pre-associated by subject**, which makes it a
truth-grade reference for the one stage nothing else can check. Strip the labels,
pool 2×512 detections per camera, run `associate_frame_graph`, score against MAMMA's
split. The clip is two people pushing and lifting each other — the hard case.

*Band:* switch rate ≤ 2% of subject-frames, **and** the inter-subject 3D distance and
epipolar-affinity margin distributions published alongside the count.
*Degenerate solution the band must reject:* "zero switches" is passable by a constant
when bodies are far apart. **The margin distribution is what makes the count mean
anything** — without it, report nothing.
*Why first:* it is needed under every candidate architecture, it is the cheapest rung
with a truth-grade reference, and it doubles as the fix for the orientation harness's
own association confound (a 160 px gate currently admits cross-person pairs and
inflates every tail measured through it).

### Step 2 — bone directions into the solver *(build, and it is a real estimator change)*

`solve_sequence_positions` is position-only. Consuming bone directions means adding an
orientation residual, with its own weight, its own Jacobian sparsity, and the same
increment discipline as the hand fit's temporal prior.

*Band, pre-registered:* held-out reprojection **must not regress**, and the
G10-corrected synthetic fixture must show the directions actually reduce true 3D
error. Coverage is 74% of bone-frames (**±8 points at n=160**) so the term must
degrade gracefully where directions are absent.
*Degenerate solution the band must reject:* a zero-weight orientation term reproduces
the current solver exactly and would pass any "does not regress" test. **The band
requires a measured improvement, not the absence of harm.**

### Step 3 — the hands question, reopened

SAM 3D's `hand_pose_params` come from a learned prior into our own rig, against the
27-DoF chain built by hand that finished at 33.9 mm held-out. It has not been
compared. Note the resolution finding cuts the other way here: at a ~20 px hand crop,
**resolution should matter**, unlike for the body.

*Band:* held-out leave-one-camera-out on the same four folds and four hands as
increment 6, against arm D's 33.9 mm.
*Degenerate solution:* a frozen mean hand scores ~98 mm; the band must reject it.

### Step 4 — Battle 3/4, informed by the above

Unchanged in shape, corrected in content: **train for the full μ/σ/visibility triple**,
because σ and visibility cannot be ranked. The target the detector must hit is
MAMMA's residual distribution and **especially its tail** — 13.08 px at p95 against
our 98.40.

---

## 7. What of MAMMA stays in the loop, and until when

Its licence bars producing artifacts for commercial purposes. **The line is: MAMMA is
a measuring instrument, never in the shipping path.** Nothing MAMMA produces may enter
a delivered artifact, a trained model's weights, or a calibration constant that ships.

| retained | what it is for | what replaces it | retirement trigger |
|---|---|---|---|
| `ma_2d` landmarks + visibility | the substitution slot: the only way to hold our geometry fixed and vary the detector | our own detector, once one exists | Battle 4 delivers a detector, **or** Battle 2 delivers markers — whichever first |
| fitted meshes / `pred_joints` | the per-joint bias and spread reference | marker reference | Battle 2 delivers |
| `triangulated_3d_pts` | shows what *their* fitter does to *their* input, so our geometry can be scored against a like stage | nothing needed after Battle 2 | Battle 2 delivers |
| `verts_512.pkl` regressor | exact landmark correspondence, so comparisons are not nearest-vertex approximations | — | with the above |
| `ma_masks` | **not used.** We consume no masks | — | already retired |

**A calibration constant fitted on MAMMA outputs cannot ship** — that is a licence
trap, not a technical one, and it is why the arm (i) offset work would have been
unusable even had it transferred.

---

## 8. Instrument table — what can produce a readable number

| instrument | readable to | blind to |
|---|---|---|
| 3D vs MAMMA's fit | **~10 mm floor** — its own fit sits 9.3–11.7 mm from its own triangulation | anything below that is reference wobble |
| 2D vs fit projections, **spread** | ~4–5 px (~8–10 mm) per camera | bias, without convention caveats |
| cross-view epipolar ladder | the **incoherent** component only | coherent bias — which is SOMA's *larger* term. **Cannot rank detector families** |
| cross-view consistency of monocular fits | fully readable, reference-free | shared appearance bias — necessary condition, never accuracy |
| synthetic fixture, corrected noise | sub-mm, truth-referenced | calibration, distortion, sync, soft tissue, joint definition — all first-order on real footage |
| association vs MAMMA's subject split | truth-grade labels | — |
| marker session | **the only absolute accuracy instrument** | not yet booked |

---

## 9. Traps that cost real time

- **`_epipolar_distance_px` returns the SYMMETRIC distance** — the sum of two
  one-sided distances. **Halve it before fitting a per-view σ.** Measured ratio 1.962.
- **MPS is impossible for SAM 3D.** float64 inside the TorchScript MHR module, out of
  reach of any external shim. CPU only, ~45.8 s/person-frame.
- **SAM 3D hardcodes `.cuda()`** in at least three places. Shims live in
  `workers/commercial_multiview/sam3d_body_pose.py`, never in the vendored checkout —
  its licence forbids reverse engineering.
- **`cam_int` must be a torch tensor of shape (B,3,3)**, not numpy, not (3,3).
- **The system Python 3.13 has a broken cert store.** Set `SSL_CERT_FILE` to certifi's
  bundle or `torch.hub` reports "no internet connection".
- **The upstream prints to stdout**, mixing with JSONL. Use `AUTOANIM_SAM3D_OUT`.
- **`gt_joints` / `gt_vertices` in MAMMA's output are byte-copies of `pred_*`.**
  There is **no ground truth on disk.** Never score against a variable named `gt_`.
- **The Bash working directory persists between calls.** A relative path after a `cd`
  put a venv inside a vendored source tree.

---

## 10. Where we stand against MAMMA — the honest paragraph

Our geometry is not the problem: given MAMMA's own 2D, our unchanged triangulator
lands within 9–10 mm of MAMMA's own landmarks, which is inside the range MAMMA's
fitter itself moves from its own triangulation — so on that axis we are at the
instrument's floor and cannot be distinguished from it. What separates us is the
detector, and specifically its **tail**: at p95 our 2D error is roughly 7.5× MAMMA's,
98 px against 13 at native resolution. That single ratio is the most durable finding
in this lane and it survived every correction. **But no number here is
ground-truth-verified.** Everything is relative to MAMMA's own output — an estimate,
not truth — or to synthetic data missing five first-order error terms. We do not know
whether our body track is at 20 mm or 40 mm against markers, and we will not know
until Battle 2 delivers. The honest position is that we have strong evidence about
*where* our error is, no measurement of *how much* of it there is, and a detector
programme that is now correctly aimed but not begun.
