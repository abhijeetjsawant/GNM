# Tongue–lip soft-contact milestone

Status: implemented and verified for the controlled GNM range-of-motion
diagnostic. This milestone replaces the showcase-only coefficient workaround;
it does not claim anatomically calibrated speech biomechanics.

## Why the current result floats

GNM supplies independent parametric surfaces for the tongue, lips, teeth, and
mouth sock. It produces coherent authored shapes, but it does not solve contact
between those surfaces. The current showcase therefore prevents a visible
intersection by changing expression coefficients. That leaves about 2.5 mm
between the tongue and lower lip at the closest keyframe: safe, but visibly
unsupported.

AutoAnim's existing Rust physics engine cannot fix that. Its P0 contract is a
target-relative *surface* secondary-motion residual with stretch and tether
constraints. It has no volume, collision, jaw, muscle, or tongue model.

## Research basis

- Extended Position Based Dynamics (XPBD) adds compliance that is independent
  of time step and solver iteration count, and applies to volume, strain, and
  contact constraints. It is a practical real-time foundation for an editable
  character tool: <https://matthias-research.github.io/pages/publications/XPBD.pdf>
- ArtiSynth's published biomechanical tongue is a volumetric finite-element
  model (946 nodes and 3,700 tetrahedra), driven by 11 muscle groups and coupled
  to jaw/hyoid mechanics. This establishes that a production anatomical tongue
  is a volume, not a displaced render surface:
  <https://www.artisynth.org/Demo/BiomechanicalTongueModel>
- Incremental Potential Contact (IPC) provides intersection- and
  inversion-free deformable contact with friction:
  <https://ipc-sim.github.io/>
- The official IPC Toolkit provides collision detection, continuous collision
  detection, barriers, and friction, but explicitly is not a complete
  simulation system. Its C++ dependency stack is also substantially heavier
  than AutoAnim's Rust core:
  <https://ipctk.xyz/v1.1.1/>
- Efficient IPC has been demonstrated on an actuated soft-body face using
  projective dynamics plus contact:
  <https://cgl.ethz.ch/Downloads/Publications/Papers/2023/Yan23b/Yan23b.pdf>

## Recovered GNM topology

The GNM v3 tongue group contains 933 vertices and 1,824 triangles. It is an open
surface with one 40-edge boundary loop. For offline preprocessing, that loop can
be capped solely for inside/outside classification. A Delaunay tetrahedralization
of the original 933 vertices is then filtered against the actual animation to
retain consistently oriented interior cells, with no new render vertices. The
lower- and upper-lip groups are included as deformable contact surfaces.

The cap is never rendered or simulated. It is a deterministic preprocessing
device used to recover volumetric connectivity from the released GNM asset.

## P1 architecture

GNM remains the kinematic target generator. A new independent Rust solver owns a
combined tongue/lower- and upper-lip state:

1. Target motion is interpolated over substeps so contact is approached
   continuously for the controlled sequence.
2. Tongue tetrahedra receive XPBD signed-volume constraints.
3. Tongue and lip surface edges receive XPBD stretch constraints.
4. Both surfaces receive compliant target attachments. Posterior tongue and lip
   corners are more strongly attached than the contacting tissue.
5. Reciprocal tongue-vertex/lip-triangle and lip-vertex/tongue-triangle pairs
   receive unilateral XPBD contact constraints. Corrections are distributed to
   both tissues, so neither lip is a rigid invisible wall.
6. A verified native anchor residual is baked into one smooth corrective basis
   over the settle/hold/release interval. The raw cache remains the audit
   source; the low-rank bake is what the GLB and preview export.

The contact set is conservatively prepared from the complete target sequence.
This avoids an expensive all-pairs search in P1 while retaining real two-way
soft contact for the known animation.

## Acceptance criteria

The implementation may replace the showcase workaround only when all of these
pass:

- The uncorrected GNM target actually attempts contact or intersection; the
  test must not be satisfied by retaining the old 2.5 mm coefficient gap.
- Every active tongue keyframe has zero exact tongue/lip triangle crossings
  after simulation.
- At least one active interval has supported near-contact, with a tongue/lip
  surface separation no greater than 1.25 mm.
- The tongue and lower lip both move by at least 0.10 mm from their kinematic
  targets during contact.
- All tetrahedra retain positive signed volume. The reported minimum and
  maximum volume ratios remain within the test's declared safety bounds.
- Output is finite and deterministic across a whole sequence and equivalent
  chunked processing.
- Invalid shapes, indices, non-finite values, and closed native handles fail
  explicitly at the Python/C boundary.
- Rust unit/C-ABI tests, Python binding tests, real-GNM integration tests, GLB
  reconstruction checks, preview generation, and browser visual inspection all
  pass.

## Explicit P1 limitations and production follow-up

P1 is a real volumetric soft-contact simulation, but it is not yet a
production anatomical speech model:

- Tetrahedra are recovered from the GNM surface; material parameters are not
  fitted to measured human tissue.
- Target attachments approximate muscle actuation. Named intrinsic/extrinsic
  tongue muscles, jaw, hyoid, palate, teeth, friction, saliva, and wet adhesion
  are not modeled.
- The conservative contact set and substepping are suitable for this controlled
  animation, not a universal no-tunneling guarantee.
- General audio/video animation needs continuous collision detection and a
  barrier/friction formulation. IPC (or an equivalent Rust implementation) is
  the production hardening path.
- Visual realism also depends on a wet oral shader, correct inner-lip thickness,
  tongue normals, and high-resolution albedo/roughness/normal/displacement maps;
  mechanics alone cannot supply those cues.

P2 should add jaw/hyoid rigid bodies, palate and teeth colliders, calibrated
muscle activation, continuous collision detection, friction, and material
fitting against captured tongue/lip motion. P3 can evaluate Metal compute after
the CPU solver and correctness tests establish a stable reference.

## Verified result — 2026-07-28

Job `01kyn5mrbz5zwa5vpy6sgz4w2z` passed the scoped build/review/test loop:

- 933 tongue vertices, 145 lower-lip vertices, and 145 upper-lip vertices;
- 2,848 retained tetrahedra with 95.75% neutral capped-volume coverage;
- 6,473 conservative reciprocal contact candidates;
- 8 substeps, 16 iterations, 216,580 applied contact projections;
- 0 inverted tetrahedron samples; volume ratio range 0.389–2.318;
- native minimum contact separation 0.675 mm;
- final baked keyframe candidate separation 0.905 mm versus a 0.092 mm
  kinematic target approach;
- tongue motion from target 1.007 mm and lower-lip motion 0.595 mm;
- zero exact tongue/lower-lip and tongue/upper-lip triangle crossings across
  the 97-frame real-GNM contact integration test;
- chunked and whole processing are bit-identical;
- 9 GLB morph targets, 0.056 mm mesh p95 error, 0.356 mm mesh maximum error,
  and no oral-semantic reconstruction change;
- browser viewer loaded 18,437 render vertices and 35,324 triangles with no
  console warnings/errors; the contact pose at 13.20 s was visually inspected.

The release native solve measured about 16.2 seconds for 451 frames on the
development machine (roughly 36 ms/frame) before topology preparation, preview
rendering, and GLB compression. P1 is therefore an offline bake, not a live
30/60 FPS production solver. Its one-basis viewer bake is real-time after
export. Profiling and parallel graph coloring are prerequisites before claiming
interactive native simulation; GPU work should follow a qualified CPU
reference rather than replace it prematurely.

Rust core/C-ABI tests: 27 passed. Focused Python physics/showcase tests: 13
passed before the final bake test was added; the final soft-contact plus
showcase set is 5/5.

The complete repository suite passed on 2026-07-29: 835 passed and 3 opt-in
tests skipped in 743.88 seconds, with no failures. The skips require external
Claire, NVIDIA Audio2Face v3, or OpenCV calibration sample assets and remain
explicitly documented by their tests.
