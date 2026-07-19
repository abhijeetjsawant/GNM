# AutoAnim Physics P0

This workspace contains the isolated Phase P0 CPU physics core. It is not yet
wired into the Python pipeline or macOS UI.

The core consumes an animated target surface and simulates only a bounded
secondary-motion residual. With zero residual and no external acceleration, an
arbitrarily moving target is reproduced exactly. This prevents the solver from
softening authored lipsync or expression timing.

## Properties

- canonical, unique undirected edges derived from triangle topology;
- fixed-order CSR gathers and edge-parallel Jacobi XPBD phases;
- a dedicated Rayon thread pool rather than the global pool;
- per-vertex motion weights, where zero is a hard protected/pinned vertex;
- a conservative 0.75 mm default Euclidean hard displacement cap after
  prediction and every iteration;
- strict finite, shape, configuration, and topology validation;
- deterministic hashes and numerical health metrics;
- scalar reference, stable-Rust auto-vectorized, and real AArch64 NEON
  contiguous inertial predictors;
- panic-contained C ABI with caller-owned output buffers and JSON reports.

The AArch64 NEON predictor remains callable for research, but it has not met the
1.10x predictor qualification gate. `Auto` therefore selects the stable
auto-vectorized path and all production reports set `simd_claim` to false.
There is no GPU implementation or GPU performance claim in P0.

This is a surface secondary-motion solver, not a muscle, tissue-volume, jaw,
tongue-contact, or collision model. It does not improve phoneme timing or acting
choices. GPU compute is intentionally deferred until profiling proves that a
single face benefits after host/device synchronization costs.

The checked-in GNM Head v3 topology was passed through the release C ABI during
P0 verification: 17,821 vertices and 35,324 triangles canonicalized to 53,135
unique edges with all indices inside `0..17,821`.

## Measured gates

The release benchmark distinguishes the physics-only path from production
hashing and health metrics. The physics-only entry point marks its report
`diagnostics_complete = false`; normal application code must use
`simulate_chunk`.

On an M2 Max, 20 independent 120-frame runs over 11,556 vertices measured:

- scalar, one thread: 3.557 ms/frame p50 and 3.598 ms/frame p95;
- scalar, eight threads: 1.586 ms/frame p50 and 1.625 ms/frame p95;
- Auto, eight threads: 1.601 ms/frame p50 and 1.679 ms/frame p95;
- full reporting, Auto/eight threads: 2.510 ms/frame p50 and 2.542 ms/frame p95.

The physics p95 gate passed (1.679 ms versus 4 ms) and scalar multithreading
passed (2.242x p50 versus 1.5x). The isolated predictor SIMD gate failed:
stable auto-vectorization was 1.037x and explicit NEON was 1.039x versus scalar,
below the required 1.10x. This is why Auto and reports make no SIMD claim.

## Commands

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo test --workspace --release
cargo run --release -p autoanim-physics-core --example benchmark
cargo bench -p autoanim-physics-core --bench secondary_motion
```
