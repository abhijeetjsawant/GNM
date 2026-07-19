use std::hint::black_box;
use std::sync::Arc;

use autoanim_physics_core::{MathKernelRequest, PhysicsConfig, PhysicsTopology, Simulator};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

fn grid(rows: usize, cols: usize) -> (Arc<PhysicsTopology>, Vec<f32>) {
    let mut triangles = Vec::with_capacity((rows - 1) * (cols - 1) * 2);
    let mut target = Vec::with_capacity(rows * cols * 3);
    for row in 0..rows {
        for col in 0..cols {
            target.extend_from_slice(&[col as f32 * 0.001, row as f32 * 0.001, 0.0]);
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
    (
        Arc::new(PhysicsTopology::from_triangles(rows * cols, &triangles).unwrap()),
        target,
    )
}

fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("target_relative_xpbd");
    for (rows, cols) in [(32, 32), (107, 108)] {
        let vertices = rows * cols;
        let (topology, target) = grid(rows, cols);
        let threads = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(8);
        group.throughput(Throughput::Elements(vertices as u64));
        for (label, worker_count, request) in [
            ("scalar-1t", 1, MathKernelRequest::Scalar),
            ("scalar-nt", threads, MathKernelRequest::Scalar),
            ("auto-1t", 1, MathKernelRequest::Auto),
            ("auto-nt", threads, MathKernelRequest::Auto),
        ] {
            group.bench_with_input(
                BenchmarkId::new(label, vertices),
                &vertices,
                |bencher, _| {
                    bencher.iter_batched(
                        || {
                            Simulator::new(
                                topology.clone(),
                                vec![1.0; vertices],
                                PhysicsConfig::default(),
                                worker_count,
                                request,
                            )
                            .unwrap()
                        },
                        |mut simulator| {
                            black_box(
                                simulator
                                    .simulate_chunk_physics_only_for_benchmark(
                                        black_box(&target),
                                        Some(&[0.0, 0.0, 40.0]),
                                    )
                                    .unwrap(),
                            )
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
