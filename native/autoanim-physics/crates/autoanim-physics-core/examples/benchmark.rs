use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use autoanim_physics_core::{
    MathKernelKind, MathKernelRequest, PhysicsConfig, PhysicsTopology, Simulator,
    apply_inertial_predictor,
};

struct Inputs {
    topology: Arc<PhysicsTopology>,
    targets: Vec<f32>,
    accelerations: Vec<f32>,
    vertices: usize,
    frames: usize,
}

#[derive(Clone, Copy)]
struct Summary {
    p50: f64,
    p95: f64,
}

fn percentiles(mut samples: Vec<f64>) -> Summary {
    samples.sort_by(f64::total_cmp);
    let p50_index = (samples.len() * 50).div_ceil(100).saturating_sub(1);
    let p95_index = (samples.len() * 95).div_ceil(100).saturating_sub(1);
    Summary {
        p50: samples[p50_index],
        p95: samples[p95_index],
    }
}

fn run_solver_case(
    label: &str,
    input: &Inputs,
    threads: usize,
    request: MathKernelRequest,
    physics_only: bool,
    runs: usize,
) -> Summary {
    let mut samples = Vec::with_capacity(runs);
    let mut selected = MathKernelKind::ScalarReference;
    for _ in 0..runs {
        let mut simulator = Simulator::new(
            input.topology.clone(),
            vec![1.0; input.vertices],
            PhysicsConfig::default(),
            threads,
            request,
        )
        .unwrap();
        selected = simulator.selected_kernel();
        let started = Instant::now();
        let output = if physics_only {
            simulator
                .simulate_chunk_physics_only_for_benchmark(
                    black_box(&input.targets),
                    Some(black_box(&input.accelerations)),
                )
                .unwrap()
        } else {
            simulator
                .simulate_chunk(
                    black_box(&input.targets),
                    Some(black_box(&input.accelerations)),
                )
                .unwrap()
        };
        black_box(output);
        samples.push(started.elapsed().as_secs_f64() * 1_000.0 / input.frames as f64);
    }
    let summary = percentiles(samples);
    println!(
        "scope={} case={label} vertices={} frames={} runs={runs} threads={threads} kernel={} p50_ms_per_frame={:.3} p95_ms_per_frame={:.3}",
        if physics_only {
            "physics-only"
        } else {
            "full-reporting"
        },
        input.vertices,
        input.frames,
        selected.name(),
        summary.p50,
        summary.p95,
    );
    summary
}

fn run_predictor_case(kind: MathKernelKind, values: usize, runs: usize) -> Summary {
    const PASSES_PER_RUN: usize = 2_000;
    let current: Vec<f32> = (0..values).map(|index| index as f32 * 0.000_031).collect();
    let previous: Vec<f32> = (0..values).map(|index| index as f32 * -0.000_017).collect();
    let mut output = vec![0.0; values];
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let started = Instant::now();
        for _ in 0..PASSES_PER_RUN {
            apply_inertial_predictor(
                kind,
                black_box(&current),
                black_box(&previous),
                black_box(&mut output),
                black_box(0.88),
            )
            .unwrap();
        }
        black_box(&output);
        samples.push(started.elapsed().as_secs_f64() * 1_000_000.0 / PASSES_PER_RUN as f64);
    }
    let summary = percentiles(samples);
    println!(
        "scope=predictor kernel={} values={values} runs={runs} passes_per_run={PASSES_PER_RUN} p50_us_per_pass={:.3} p95_us_per_pass={:.3}",
        kind.name(),
        summary.p50,
        summary.p95,
    );
    summary
}

fn make_inputs() -> Inputs {
    let rows = 107;
    let cols = 108;
    let frames = 120;
    let vertices = rows * cols;
    let mut triangles = Vec::with_capacity((rows - 1) * (cols - 1) * 2);
    let mut frame = Vec::with_capacity(vertices * 3);
    for row in 0..rows {
        for col in 0..cols {
            frame.extend_from_slice(&[col as f32 * 0.001, row as f32 * 0.001, 0.0]);
        }
    }
    for row in 0..rows - 1 {
        for col in 0..cols - 1 {
            let a = (row * cols + col) as u32;
            let b = a + 1;
            let c = a + cols as u32;
            let d = c + 1;
            triangles.push([a, b, d]);
            triangles.push([a, d, c]);
        }
    }
    let topology = Arc::new(PhysicsTopology::from_triangles(vertices, &triangles).unwrap());
    let targets = frame.repeat(frames);
    let mut accelerations = vec![0.0; frames * 3];
    accelerations[2] = 40.0;
    Inputs {
        topology,
        targets,
        accelerations,
        vertices,
        frames,
    }
}

fn main() {
    let input = make_inputs();
    let runs = std::env::var("AUTOANIM_BENCH_RUNS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20);
    assert!(runs >= 2, "AUTOANIM_BENCH_RUNS must be at least 2");
    let predictor_runs = runs.max(20);
    let parallel_threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(8);

    let scalar_one = run_solver_case(
        "scalar-1t",
        &input,
        1,
        MathKernelRequest::Scalar,
        true,
        runs,
    );
    let scalar_many = run_solver_case(
        "scalar-nt",
        &input,
        parallel_threads,
        MathKernelRequest::Scalar,
        true,
        runs,
    );
    let auto_many = run_solver_case(
        "auto-nt",
        &input,
        parallel_threads,
        MathKernelRequest::Auto,
        true,
        runs,
    );
    println!(
        "gate=physics-only-p95 threshold_ms=4.000 measured_ms={:.3} pass={}",
        auto_many.p95,
        auto_many.p95 <= 4.0,
    );
    println!(
        "gate=multithread-speedup threshold=1.500 measured_p50={:.3} measured_p95={:.3} pass={}",
        scalar_one.p50 / scalar_many.p50,
        scalar_one.p95 / scalar_many.p95,
        scalar_one.p50 / scalar_many.p50 >= 1.5 && scalar_one.p95 / scalar_many.p95 >= 1.5,
    );

    // Full-reporting measurements make the hashing and metric cost explicit.
    run_solver_case(
        "auto-nt",
        &input,
        parallel_threads,
        MathKernelRequest::Auto,
        false,
        runs,
    );

    let scalar_predictor = run_predictor_case(
        MathKernelKind::ScalarReference,
        input.vertices * 3,
        predictor_runs,
    );
    let stable_predictor = run_predictor_case(
        MathKernelKind::StableAutoVectorized,
        input.vertices * 3,
        predictor_runs,
    );
    println!(
        "gate=stable-predictor-speedup threshold=1.100 measured_p50={:.3} pass={}",
        scalar_predictor.p50 / stable_predictor.p50,
        scalar_predictor.p50 / stable_predictor.p50 >= 1.1,
    );
    #[cfg(target_arch = "aarch64")]
    {
        let neon_predictor = run_predictor_case(
            MathKernelKind::NeonIntrinsics,
            input.vertices * 3,
            predictor_runs,
        );
        let speedup = scalar_predictor.p50 / neon_predictor.p50;
        println!(
            "gate=neon-predictor-speedup threshold=1.100 measured_p50={speedup:.3} pass={} auto_selects_neon=false simd_claim=false",
            speedup >= 1.1,
        );
    }
}
