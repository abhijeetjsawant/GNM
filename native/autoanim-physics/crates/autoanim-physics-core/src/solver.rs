use std::sync::Arc;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::kernel::{self, Capabilities, MathKernelKind, MathKernelRequest};
use crate::{PhysicsError, PhysicsTopology, Vec3, add, length, lerp, scale, sub};

const PARALLEL_VERTEX_THRESHOLD: usize = 4_096;
const PARALLEL_EDGE_THRESHOLD: usize = 8_192;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhysicsConfig {
    pub frames_per_second: f32,
    pub substeps: u32,
    pub iterations: u32,
    pub velocity_retention: f32,
    pub stretch_compliance: f32,
    pub tether_compliance: f32,
    pub max_displacement_m: f32,
    pub jacobi_relaxation: f32,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            frames_per_second: 60.0,
            substeps: 2,
            iterations: 5,
            velocity_retention: 0.88,
            stretch_compliance: 2.0e-7,
            tether_compliance: 8.0e-6,
            max_displacement_m: 0.00075,
            jacobi_relaxation: 0.8,
        }
    }
}

impl PhysicsConfig {
    pub fn validate(&self) -> Result<(), PhysicsError> {
        finite_range(
            "frames_per_second",
            self.frames_per_second,
            0.0,
            480.0,
            false,
        )?;
        if !(1..=16).contains(&self.substeps) {
            return Err(PhysicsError::InvalidConfig(
                "substeps must be in 1..=16".into(),
            ));
        }
        if !(1..=64).contains(&self.iterations) {
            return Err(PhysicsError::InvalidConfig(
                "iterations must be in 1..=64".into(),
            ));
        }
        finite_range(
            "velocity_retention",
            self.velocity_retention,
            0.0,
            1.0,
            true,
        )?;
        finite_range(
            "stretch_compliance",
            self.stretch_compliance,
            0.0,
            f32::MAX,
            true,
        )?;
        finite_range(
            "tether_compliance",
            self.tether_compliance,
            0.0,
            f32::MAX,
            true,
        )?;
        finite_range(
            "max_displacement_m",
            self.max_displacement_m,
            0.0,
            1.0,
            false,
        )?;
        finite_range("jacobi_relaxation", self.jacobi_relaxation, 0.0, 1.0, false)?;
        Ok(())
    }
}

fn finite_range(
    name: &str,
    value: f32,
    minimum: f32,
    maximum: f32,
    inclusive_minimum: bool,
) -> Result<(), PhysicsError> {
    let minimum_ok = if inclusive_minimum {
        value >= minimum
    } else {
        value > minimum
    };
    if !value.is_finite() || !minimum_ok || value > maximum {
        let comparison = if inclusive_minimum {
            "at least"
        } else {
            "greater than"
        };
        return Err(PhysicsError::InvalidConfig(format!(
            "{name} must be finite, {comparison} {minimum}, and at most {maximum}"
        )));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimulationReport {
    pub schema_version: u32,
    pub backend: String,
    pub kernel: MathKernelKind,
    pub threads: usize,
    pub frame_count: u64,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub substeps: u32,
    pub iterations: u32,
    pub topology_sha256: String,
    pub config_sha256: String,
    pub input_sha256: String,
    pub output_sha256: String,
    /// True only for the implementation backed by explicit AArch64 NEON
    /// intrinsics. The auto-vectorized implementation never claims SIMD.
    pub simd_claim: bool,
    pub fallback_reason: Option<String>,
    pub max_displacement_m: f32,
    pub rms_displacement_m: f32,
    pub max_edge_strain: f32,
    pub pinned_drift_m: f32,
    pub finite: bool,
    pub target_relative: bool,
    pub externally_accelerated_frames: u64,
    pub diagnostics_complete: bool,
}

/// A stateful simulation using its own Rayon pool, never the process-global pool.
pub struct Simulator {
    topology: Arc<PhysicsTopology>,
    config: PhysicsConfig,
    motion_weights: Vec<f32>,
    pool: Arc<rayon::ThreadPool>,
    threads: usize,
    kernel: MathKernelKind,
    fallback_reason: Option<String>,
    residual: Vec<f32>,
    previous_residual: Vec<f32>,
    scratch: Vec<f32>,
    current_target: Vec<Vec3>,
    last_target: Vec<Vec3>,
    interpolated_target: Vec<Vec3>,
    initialized: bool,
    edge_lambdas: Vec<f32>,
    edge_corrections: Vec<Vec3>,
    edge_rest_lengths: Vec<f32>,
    edge_inverse_denominators: Vec<f32>,
    tether_lambdas: Vec<Vec3>,
    tether_inverse_denominators: Vec<f32>,
    next_residual: Vec<Vec3>,
    input_hasher: Sha256,
    output_hasher: Sha256,
    frame_count: u64,
    accelerated_frames: u64,
    max_displacement: f32,
    displacement_squared_sum: f64,
    displacement_samples: u64,
    max_edge_strain: f32,
    pinned_drift: f32,
    diagnostics_complete: bool,
}

impl Simulator {
    pub fn new(
        topology: Arc<PhysicsTopology>,
        motion_weights: Vec<f32>,
        config: PhysicsConfig,
        threads: usize,
        kernel_request: MathKernelRequest,
    ) -> Result<Self, PhysicsError> {
        config.validate()?;
        if threads == 0 || threads > 256 {
            return Err(PhysicsError::InvalidConfig(
                "threads must be in 1..=256".into(),
            ));
        }
        if motion_weights.len() != topology.vertex_count() {
            return Err(PhysicsError::InvalidInput(format!(
                "motion_weights has length {}, expected {}",
                motion_weights.len(),
                topology.vertex_count()
            )));
        }
        for (index, weight) in motion_weights.iter().copied().enumerate() {
            if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
                return Err(PhysicsError::InvalidInput(format!(
                    "motion weight {index} must be finite and in 0..=1"
                )));
            }
        }
        let kernel = kernel::select_kernel(kernel_request)?;
        let fallback_reason = match kernel_request {
            MathKernelRequest::Auto => Some(
                "explicit SIMD research kernels have not met the 1.10x predictor qualification gate; selected stable auto-vectorization without a SIMD claim"
                    .into(),
            ),
            MathKernelRequest::Neon => Some(
                "explicit NEON research selection is callable but is not production-qualified"
                    .into(),
            ),
            _ => None,
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("autoanim-physics-{index}"))
            .build()
            .map_err(|error| PhysicsError::ThreadPool(error.to_string()))?;
        let vertex_count = topology.vertex_count();
        let edge_count = topology.edges().len();
        let dt = 1.0 / config.frames_per_second / config.substeps as f32;
        let dt_squared = dt * dt;
        let edge_alpha = config.stretch_compliance / dt_squared;
        let tether_alpha = config.tether_compliance / dt_squared;
        let edge_inverse_denominators = topology
            .edges()
            .iter()
            .map(|edge| {
                let denominator =
                    motion_weights[edge.a as usize] + motion_weights[edge.b as usize] + edge_alpha;
                if denominator > 0.0 {
                    1.0 / denominator
                } else {
                    0.0
                }
            })
            .collect();
        let tether_inverse_denominators = motion_weights
            .iter()
            .map(|weight| {
                let denominator = *weight + tether_alpha;
                if denominator > 0.0 {
                    1.0 / denominator
                } else {
                    0.0
                }
            })
            .collect();
        Ok(Self {
            topology,
            config,
            motion_weights,
            pool: Arc::new(pool),
            threads,
            kernel,
            fallback_reason,
            residual: vec![0.0; vertex_count * 3],
            previous_residual: vec![0.0; vertex_count * 3],
            scratch: vec![0.0; vertex_count * 3],
            current_target: vec![[0.0; 3]; vertex_count],
            last_target: vec![[0.0; 3]; vertex_count],
            interpolated_target: vec![[0.0; 3]; vertex_count],
            initialized: false,
            edge_lambdas: vec![0.0; edge_count],
            edge_corrections: vec![[0.0; 3]; edge_count],
            edge_rest_lengths: vec![0.0; edge_count],
            edge_inverse_denominators,
            tether_lambdas: vec![[0.0; 3]; vertex_count],
            tether_inverse_denominators,
            next_residual: vec![[0.0; 3]; vertex_count],
            input_hasher: Sha256::new(),
            output_hasher: Sha256::new(),
            frame_count: 0,
            accelerated_frames: 0,
            max_displacement: 0.0,
            displacement_squared_sum: 0.0,
            displacement_samples: 0,
            max_edge_strain: 0.0,
            pinned_drift: 0.0,
            diagnostics_complete: true,
        })
    }

    pub fn capabilities(&self) -> Capabilities {
        Capabilities::detect()
    }

    pub fn selected_kernel(&self) -> MathKernelKind {
        self.kernel
    }

    /// Simulates a contiguous chunk of frames. Targets are xyz-interleaved and
    /// accelerations, when present, contain one face-local xyz vector per frame.
    pub fn simulate_chunk(
        &mut self,
        targets: &[f32],
        accelerations: Option<&[f32]>,
    ) -> Result<Vec<f32>, PhysicsError> {
        let (frames, frame_width) = self.validate_chunk_inputs(targets, accelerations)?;

        // Hash only after the whole chunk passes validation, preserving atomic
        // behavior for rejected input.
        // Canonical per-frame hashing keeps reports invariant when the same
        // stream is submitted in differently sized chunks.
        for frame_index in 0..frames {
            hash_f32_slice(
                &mut self.input_hasher,
                &targets[frame_index * frame_width..(frame_index + 1) * frame_width],
            );
            self.input_hasher
                .update([u8::from(accelerations.is_some())]);
            if let Some(values) = accelerations {
                hash_f32_slice(
                    &mut self.input_hasher,
                    &values[frame_index * 3..frame_index * 3 + 3],
                );
            }
        }

        let pool = Arc::clone(&self.pool);
        let output = pool.install(|| self.simulate_validated_chunk(targets, accelerations, true));
        hash_f32_slice(&mut self.output_hasher, &output);
        Ok(output)
    }

    /// Runs the validated physics and output-copy path without hashes or health
    /// metrics. This exists only to measure the solver's physics-only gate.
    /// Calling `report()` afterwards returns `diagnostics_complete = false` and
    /// must not be treated as a production report.
    pub fn simulate_chunk_physics_only_for_benchmark(
        &mut self,
        targets: &[f32],
        accelerations: Option<&[f32]>,
    ) -> Result<Vec<f32>, PhysicsError> {
        self.validate_chunk_inputs(targets, accelerations)?;
        self.diagnostics_complete = false;
        let pool = Arc::clone(&self.pool);
        Ok(pool.install(|| self.simulate_validated_chunk(targets, accelerations, false)))
    }

    fn validate_chunk_inputs(
        &self,
        targets: &[f32],
        accelerations: Option<&[f32]>,
    ) -> Result<(usize, usize), PhysicsError> {
        let frame_width = self.topology.vertex_count() * 3;
        if targets.is_empty() || !targets.len().is_multiple_of(frame_width) {
            return Err(PhysicsError::InvalidInput(format!(
                "targets must contain a nonzero whole number of {frame_width}-float frames"
            )));
        }
        let frames = targets.len() / frame_width;
        if let Some(values) = accelerations
            && values.len() != frames * 3
        {
            return Err(PhysicsError::InvalidInput(format!(
                "accelerations has length {}, expected {}",
                values.len(),
                frames * 3
            )));
        }
        validate_finite("targets", targets)?;
        if let Some(values) = accelerations {
            validate_finite("accelerations", values)?;
        }
        Ok((frames, frame_width))
    }

    fn simulate_validated_chunk(
        &mut self,
        targets: &[f32],
        accelerations: Option<&[f32]>,
        record_diagnostics: bool,
    ) -> Vec<f32> {
        let frame_width = self.topology.vertex_count() * 3;
        let frames = targets.len() / frame_width;
        let mut output = Vec::with_capacity(targets.len());
        for frame_index in 0..frames {
            let flat_target = &targets[frame_index * frame_width..(frame_index + 1) * frame_width];
            for (target, value) in self
                .current_target
                .iter_mut()
                .zip(flat_target.chunks_exact(3))
            {
                *target = [value[0], value[1], value[2]];
            }
            let acceleration = accelerations
                .map(|values| {
                    [
                        values[frame_index * 3],
                        values[frame_index * 3 + 1],
                        values[frame_index * 3 + 2],
                    ]
                })
                .unwrap_or([0.0; 3]);
            if record_diagnostics && acceleration != [0.0; 3] {
                self.accelerated_frames += 1;
            }
            self.simulate_frame(acceleration);
            for (target_position, residual) in self
                .current_target
                .iter()
                .zip(self.residual.chunks_exact(3))
            {
                let value = [
                    target_position[0] + residual[0],
                    target_position[1] + residual[1],
                    target_position[2] + residual[2],
                ];
                output.extend_from_slice(&value);
            }
            let frame_start = output.len() - frame_width;
            if record_diagnostics {
                self.update_metrics(&output[frame_start..]);
            }
            self.last_target.copy_from_slice(&self.current_target);
            self.frame_count += 1;
        }
        output
    }

    fn simulate_frame(&mut self, acceleration: Vec3) {
        if !self.initialized {
            self.last_target.copy_from_slice(&self.current_target);
            self.initialized = true;
        }
        let dt = 1.0 / self.config.frames_per_second / self.config.substeps as f32;
        let dt_squared = dt * dt;
        let alpha_edge = self.config.stretch_compliance / dt_squared;
        let alpha_tether = self.config.tether_compliance / dt_squared;

        for substep in 0..self.config.substeps {
            let alpha = (substep + 1) as f32 / self.config.substeps as f32;
            for ((interpolated, &before), &now) in self
                .interpolated_target
                .iter_mut()
                .zip(&self.last_target)
                .zip(&self.current_target)
            {
                *interpolated = lerp(before, now, alpha);
            }
            self.compute_rest_lengths();

            kernel::predict(
                self.kernel,
                &self.residual,
                &self.previous_residual,
                &mut self.scratch,
                self.config.velocity_retention,
            );
            std::mem::swap(&mut self.previous_residual, &mut self.residual);
            std::mem::swap(&mut self.residual, &mut self.scratch);

            let weights = &self.motion_weights;
            let cap = self.config.max_displacement_m;
            let apply_force = |(index, residual): (usize, &mut [f32])| {
                let weight = weights[index];
                if weight == 0.0 {
                    residual.copy_from_slice(&[0.0; 3]);
                } else {
                    residual[0] += acceleration[0] * dt_squared * weight;
                    residual[1] += acceleration[1] * dt_squared * weight;
                    residual[2] += acceleration[2] * dt_squared * weight;
                    cap_slice(residual, cap * weight);
                }
            };
            if self.threads == 1 || self.motion_weights.len() < PARALLEL_VERTEX_THRESHOLD {
                self.residual
                    .chunks_mut(3)
                    .enumerate()
                    .for_each(apply_force);
            } else {
                self.residual
                    .par_chunks_mut(3)
                    .with_min_len(512)
                    .enumerate()
                    .for_each(apply_force);
            }
            self.edge_lambdas.fill(0.0);
            self.tether_lambdas.fill([0.0; 3]);

            for _ in 0..self.config.iterations {
                self.solve_edges(alpha_edge);
                self.gather_corrections(alpha_tether);
            }
        }
    }

    fn compute_rest_lengths(&mut self) {
        let edges = self.topology.edges();
        let target = &self.interpolated_target;
        let compute = |(edge_index, rest_length): (usize, &mut f32)| {
            let edge = edges[edge_index];
            *rest_length = length(sub(target[edge.a as usize], target[edge.b as usize]));
        };
        if self.threads == 1 || edges.len() < PARALLEL_EDGE_THRESHOLD {
            self.edge_rest_lengths
                .iter_mut()
                .enumerate()
                .for_each(compute);
        } else {
            self.edge_rest_lengths
                .par_iter_mut()
                .with_min_len(1_024)
                .enumerate()
                .for_each(compute);
        }
    }

    fn solve_edges(&mut self, compliance_alpha: f32) {
        let edges = self.topology.edges();
        let residual = &self.residual;
        let target = &self.interpolated_target;
        let rest_lengths = &self.edge_rest_lengths;
        let inverse_denominators = &self.edge_inverse_denominators;
        let solve_edge = |(edge_index, (correction, lambda)): (usize, (&mut Vec3, &mut f32))| {
            let edge = edges[edge_index];
            let a = edge.a as usize;
            let b = edge.b as usize;
            let rest_length = rest_lengths[edge_index];
            let position_a = add(target[a], read_vec3(residual, a));
            let position_b = add(target[b], read_vec3(residual, b));
            let delta = sub(position_a, position_b);
            let distance = length(delta);
            let inverse_denominator = inverse_denominators[edge_index];
            if distance <= 1.0e-12 || inverse_denominator == 0.0 {
                *correction = [0.0; 3];
                return;
            }
            let constraint = distance - rest_length;
            let delta_lambda = (-constraint - compliance_alpha * *lambda) * inverse_denominator;
            *lambda += delta_lambda;
            *correction = scale(delta, delta_lambda / distance);
        };
        if self.threads == 1 || edges.len() < PARALLEL_EDGE_THRESHOLD {
            self.edge_corrections
                .iter_mut()
                .zip(self.edge_lambdas.iter_mut())
                .enumerate()
                .for_each(solve_edge);
        } else {
            self.edge_corrections
                .par_iter_mut()
                .with_min_len(1_024)
                .zip(self.edge_lambdas.par_iter_mut())
                .enumerate()
                .for_each(solve_edge);
        }
    }

    fn gather_corrections(&mut self, tether_compliance_alpha: f32) {
        let topology = &self.topology;
        let weights = &self.motion_weights;
        let residual = &self.residual;
        let edge_corrections = &self.edge_corrections;
        let tether_inverse_denominators = &self.tether_inverse_denominators;
        let relaxation = self.config.jacobi_relaxation;
        let cap = self.config.max_displacement_m;
        let gather = |(vertex, (next, lambda)): (usize, (&mut Vec3, &mut Vec3))| {
            let weight = weights[vertex];
            if weight == 0.0 {
                *next = [0.0; 3];
                return;
            }
            let neighbors = topology.neighbors_of(vertex);
            let mut edge_sum = [0.0; 3];
            // This fixed edge-index order is the determinism boundary.
            for neighbor in neighbors {
                let signed = neighbor.sign as f32;
                let correction = edge_corrections[neighbor.edge_index as usize];
                edge_sum[0] += correction[0] * signed * weight;
                edge_sum[1] += correction[1] * signed * weight;
                edge_sum[2] += correction[2] * signed * weight;
            }
            let edge_scale = if neighbors.is_empty() {
                0.0
            } else {
                1.0 / neighbors.len() as f32
            };
            let old = read_vec3(residual, vertex);
            let mut tether_correction = [0.0; 3];
            for axis in 0..3 {
                let delta_lambda = (-old[axis] - tether_compliance_alpha * lambda[axis])
                    * tether_inverse_denominators[vertex];
                lambda[axis] += delta_lambda;
                tether_correction[axis] = weight * delta_lambda;
            }
            let delta = add(scale(edge_sum, edge_scale), tether_correction);
            *next = add(old, scale(delta, relaxation));
            cap_vec3(next, cap * weight);
        };
        if self.threads == 1 || weights.len() < PARALLEL_VERTEX_THRESHOLD {
            self.next_residual
                .iter_mut()
                .zip(self.tether_lambdas.iter_mut())
                .enumerate()
                .for_each(gather);
        } else {
            self.next_residual
                .par_iter_mut()
                .with_min_len(512)
                .zip(self.tether_lambdas.par_iter_mut())
                .enumerate()
                .for_each(gather);
        }
        for (destination, value) in self.residual.chunks_exact_mut(3).zip(&self.next_residual) {
            destination.copy_from_slice(value);
        }
    }

    fn update_metrics(&mut self, output: &[f32]) {
        for (index, residual) in self.residual.chunks_exact(3).enumerate() {
            let displacement = length([residual[0], residual[1], residual[2]]);
            self.max_displacement = self.max_displacement.max(displacement);
            self.displacement_squared_sum += f64::from(displacement) * f64::from(displacement);
            self.displacement_samples += 1;
            if self.motion_weights[index] == 0.0 {
                self.pinned_drift = self.pinned_drift.max(displacement);
            }
        }
        for edge in self.topology.edges() {
            let a = edge.a as usize;
            let b = edge.b as usize;
            let rest = length(sub(self.current_target[a], self.current_target[b]));
            let actual = length(sub(read_vec3(output, a), read_vec3(output, b)));
            let strain = (actual - rest).abs() / rest.max(1.0e-8);
            self.max_edge_strain = self.max_edge_strain.max(strain);
        }
    }

    pub fn report(&self) -> SimulationReport {
        let rms = if self.displacement_samples == 0 {
            0.0
        } else {
            (self.displacement_squared_sum / self.displacement_samples as f64).sqrt() as f32
        };
        SimulationReport {
            schema_version: 1,
            backend: "cpu-rayon-target-relative-jacobi-xpbd".into(),
            kernel: self.kernel,
            threads: self.threads,
            frame_count: self.frame_count,
            vertex_count: self.topology.vertex_count(),
            edge_count: self.topology.edges().len(),
            substeps: self.config.substeps,
            iterations: self.config.iterations,
            topology_sha256: self.topology.digest().into(),
            config_sha256: config_digest(&self.config, &self.motion_weights),
            input_sha256: digest_clone(&self.input_hasher),
            output_sha256: digest_clone(&self.output_hasher),
            simd_claim: false,
            fallback_reason: self.fallback_reason.clone(),
            max_displacement_m: self.max_displacement,
            rms_displacement_m: rms,
            max_edge_strain: self.max_edge_strain,
            pinned_drift_m: self.pinned_drift,
            finite: self.residual.iter().all(|value| value.is_finite()),
            target_relative: true,
            externally_accelerated_frames: self.accelerated_frames,
            diagnostics_complete: self.diagnostics_complete,
        }
    }
}

fn digest_clone(hasher: &Sha256) -> String {
    format!("{:x}", hasher.clone().finalize())
}

fn config_digest(config: &PhysicsConfig, motion_weights: &[f32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(config.frames_per_second.to_le_bytes());
    hasher.update(config.substeps.to_le_bytes());
    hasher.update(config.iterations.to_le_bytes());
    hasher.update(config.velocity_retention.to_le_bytes());
    hasher.update(config.stretch_compliance.to_le_bytes());
    hasher.update(config.tether_compliance.to_le_bytes());
    hasher.update(config.max_displacement_m.to_le_bytes());
    hasher.update(config.jacobi_relaxation.to_le_bytes());
    hasher.update((motion_weights.len() as u64).to_le_bytes());
    hash_f32_slice(&mut hasher, motion_weights);
    format!("{:x}", hasher.finalize())
}

fn validate_finite(name: &str, values: &[f32]) -> Result<(), PhysicsError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(PhysicsError::InvalidInput(format!(
            "{name}[{index}] is not finite"
        )));
    }
    Ok(())
}

fn hash_f32_slice(hasher: &mut Sha256, values: &[f32]) {
    #[cfg(target_endian = "little")]
    hasher.update(bytemuck::cast_slice(values));

    #[cfg(target_endian = "big")]
    for value in values {
        hasher.update(value.to_le_bytes());
    }
}

#[inline]
fn read_vec3(values: &[f32], index: usize) -> Vec3 {
    let base = index * 3;
    [values[base], values[base + 1], values[base + 2]]
}

#[inline]
fn cap_slice(value: &mut [f32], maximum: f32) {
    let magnitude = length([value[0], value[1], value[2]]);
    if magnitude > maximum && magnitude > 0.0 {
        let factor = maximum / magnitude;
        value[0] *= factor;
        value[1] *= factor;
        value[2] *= factor;
    }
}

#[inline]
fn cap_vec3(value: &mut Vec3, maximum: f32) {
    let magnitude = length(*value);
    if magnitude > maximum && magnitude > 0.0 {
        *value = scale(*value, maximum / magnitude);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(rows: usize, cols: usize) -> (Arc<PhysicsTopology>, Vec<f32>) {
        let mut triangles = Vec::new();
        let mut target = Vec::new();
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

    fn simulator(
        topology: Arc<PhysicsTopology>,
        weights: Vec<f32>,
        threads: usize,
        kernel: MathKernelRequest,
    ) -> Simulator {
        Simulator::new(topology, weights, PhysicsConfig::default(), threads, kernel).unwrap()
    }

    fn repeated_frames(frame: &[f32], count: usize) -> Vec<f32> {
        frame.repeat(count)
    }

    #[test]
    fn unforced_constant_target_is_an_exact_no_op() {
        let (topology, target) = grid(4, 5);
        let input = repeated_frames(&target, 8);
        let mut solver = simulator(topology, vec![1.0; 20], 3, MathKernelRequest::Scalar);
        assert_eq!(solver.simulate_chunk(&input, None).unwrap(), input);
        assert_eq!(solver.report().max_displacement_m, 0.0);
    }

    #[test]
    fn moving_target_follows_exactly_with_zero_residual_and_force() {
        let (topology, base) = grid(5, 6);
        let mut targets = Vec::new();
        for frame in 0..17 {
            for (vertex, point) in base.chunks_exact(3).enumerate() {
                let phase = frame as f32 * 0.31 + vertex as f32 * 0.07;
                targets.extend_from_slice(&[
                    point[0] + phase.sin() * 0.003,
                    point[1] + (phase * 0.7).cos() * 0.002,
                    point[2] + (phase * 1.3).sin() * 0.004,
                ]);
            }
        }
        let mut solver = simulator(topology, vec![1.0; 30], 4, MathKernelRequest::Scalar);
        assert_eq!(solver.simulate_chunk(&targets, None).unwrap(), targets);
        assert_eq!(solver.report().max_displacement_m, 0.0);
    }

    #[test]
    fn report_serialization_has_deterministic_config_and_truthful_simd_fields() {
        let (topology, target) = grid(3, 3);
        let mut solver = simulator(topology.clone(), vec![1.0; 9], 2, MathKernelRequest::Auto);
        solver.simulate_chunk(&target, None).unwrap();
        let report = solver.report();
        assert_eq!(report.kernel, MathKernelKind::StableAutoVectorized);
        let serialized = serde_json::to_value(&report).unwrap();
        assert_eq!(report.config_sha256.len(), 64);
        assert_eq!(serialized["config_sha256"], report.config_sha256);
        assert_eq!(serialized["simd_claim"], false);
        assert!(!report.simd_claim);
        assert!(report.fallback_reason.is_some());

        let changed_config = PhysicsConfig {
            iterations: PhysicsConfig::default().iterations + 1,
            ..PhysicsConfig::default()
        };
        let changed = Simulator::new(
            topology.clone(),
            vec![1.0; 9],
            changed_config,
            2,
            MathKernelRequest::Scalar,
        )
        .unwrap();
        assert_ne!(report.config_sha256, changed.report().config_sha256);
        assert!(!changed.report().simd_claim);
        assert!(changed.report().fallback_reason.is_none());

        let mut changed_weights = vec![1.0; 9];
        changed_weights[4] = 0.5;
        let weighted = Simulator::new(
            topology,
            changed_weights,
            PhysicsConfig::default(),
            2,
            MathKernelRequest::Scalar,
        )
        .unwrap();
        assert_ne!(report.config_sha256, weighted.report().config_sha256);
    }

    #[test]
    fn pinned_vertices_remain_exact_under_force() {
        let (topology, target) = grid(3, 4);
        let mut weights = vec![1.0; 12];
        weights[0] = 0.0;
        weights[5] = 0.0;
        let mut solver = simulator(topology, weights, 2, MathKernelRequest::Scalar);
        let frames = repeated_frames(&target, 5);
        let acceleration = [0.0, 0.0, 140.0].repeat(5);
        let output = solver.simulate_chunk(&frames, Some(&acceleration)).unwrap();
        let width = target.len();
        for frame in output.chunks_exact(width) {
            for vertex in [0, 5] {
                assert_eq!(
                    &frame[vertex * 3..vertex * 3 + 3],
                    &target[vertex * 3..vertex * 3 + 3]
                );
            }
        }
        assert_eq!(solver.report().pinned_drift_m, 0.0);
    }

    #[test]
    fn fully_pinned_zero_compliance_mesh_remains_finite() {
        let (topology, target) = grid(3, 3);
        let config = PhysicsConfig {
            stretch_compliance: 0.0,
            tether_compliance: 0.0,
            ..PhysicsConfig::default()
        };
        let mut solver =
            Simulator::new(topology, vec![0.0; 9], config, 2, MathKernelRequest::Scalar).unwrap();
        let targets = repeated_frames(&target, 3);
        let output = solver
            .simulate_chunk(&targets, Some(&[0.0, 0.0, 500.0].repeat(3)))
            .unwrap();
        assert_eq!(output, targets);
        assert!(solver.report().finite);
    }

    #[test]
    fn response_is_nonzero_finite_and_hard_bounded() {
        let (topology, target) = grid(4, 4);
        let mut solver = simulator(topology, vec![1.0; 16], 3, MathKernelRequest::Scalar);
        let input = repeated_frames(&target, 12);
        let mut acceleration = vec![0.0; 36];
        acceleration[3 + 2] = 8_000.0;
        let output = solver.simulate_chunk(&input, Some(&acceleration)).unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
        let report = solver.report();
        assert!(report.max_displacement_m > 0.0);
        assert!(report.max_displacement_m <= PhysicsConfig::default().max_displacement_m + 1.0e-7);
        assert!(report.finite);
    }

    #[test]
    fn rejected_chunk_does_not_mutate_state_or_hashes() {
        let (topology, target) = grid(3, 3);
        let mut checked = simulator(topology.clone(), vec![1.0; 9], 1, MathKernelRequest::Scalar);
        let before = checked.report();
        let mut invalid = target.clone();
        invalid[4] = f32::NAN;
        assert!(checked.simulate_chunk(&invalid, None).is_err());
        assert_eq!(checked.report(), before);
        let valid_output = checked.simulate_chunk(&target, None).unwrap();

        let mut fresh = simulator(topology, vec![1.0; 9], 1, MathKernelRequest::Scalar);
        assert_eq!(valid_output, fresh.simulate_chunk(&target, None).unwrap());
        assert_eq!(checked.report(), fresh.report());
    }

    #[test]
    fn benchmark_path_matches_physics_and_marks_report_incomplete() {
        let (topology, target) = grid(5, 5);
        let targets = repeated_frames(&target, 6);
        let acceleration = [0.0, 0.0, 30.0].repeat(6);
        let mut production = simulator(
            topology.clone(),
            vec![1.0; 25],
            3,
            MathKernelRequest::Scalar,
        );
        let mut benchmark = simulator(topology, vec![1.0; 25], 3, MathKernelRequest::Scalar);
        let expected = production
            .simulate_chunk(&targets, Some(&acceleration))
            .unwrap();
        let actual = benchmark
            .simulate_chunk_physics_only_for_benchmark(&targets, Some(&acceleration))
            .unwrap();
        assert_eq!(expected, actual);
        assert!(production.report().diagnostics_complete);
        assert!(!benchmark.report().diagnostics_complete);
    }

    #[test]
    fn whole_and_split_chunks_are_bit_identical() {
        let (topology, target) = grid(7, 8);
        let targets = repeated_frames(&target, 14);
        let mut acceleration = vec![0.0; 14 * 3];
        acceleration[3] = 71.0;
        acceleration[15 + 1] = -22.0;
        let mut whole = simulator(
            topology.clone(),
            vec![1.0; 56],
            3,
            MathKernelRequest::Scalar,
        );
        let whole_output = whole.simulate_chunk(&targets, Some(&acceleration)).unwrap();

        let mut split = simulator(topology, vec![1.0; 56], 3, MathKernelRequest::Scalar);
        let width = target.len();
        let mut split_output = split
            .simulate_chunk(&targets[..width * 5], Some(&acceleration[..15]))
            .unwrap();
        split_output.extend(
            split
                .simulate_chunk(&targets[width * 5..], Some(&acceleration[15..]))
                .unwrap(),
        );
        assert_eq!(whole_output, split_output);
        assert_eq!(whole.report(), split.report());
    }

    #[test]
    fn thread_count_does_not_change_results() {
        let (topology, target) = grid(19, 21);
        let targets = repeated_frames(&target, 9);
        let mut acceleration = vec![0.0; 27];
        acceleration[2] = 95.0;
        acceleration[10] = -31.0;
        let mut one = simulator(
            topology.clone(),
            vec![1.0; 399],
            1,
            MathKernelRequest::Scalar,
        );
        let mut four = simulator(topology, vec![1.0; 399], 4, MathKernelRequest::Scalar);
        assert_eq!(
            one.simulate_chunk(&targets, Some(&acceleration)).unwrap(),
            four.simulate_chunk(&targets, Some(&acceleration)).unwrap()
        );
        let mut one_report = one.report();
        let mut four_report = four.report();
        one_report.threads = 0;
        four_report.threads = 0;
        assert_eq!(one_report, four_report);
    }

    #[test]
    fn lower_velocity_retention_damps_more_after_an_impulse() {
        let (topology, target) = grid(5, 5);
        let low_config = PhysicsConfig {
            velocity_retention: 0.15,
            tether_compliance: 1.0e-3,
            ..PhysicsConfig::default()
        };
        let mut high_config = low_config.clone();
        high_config.velocity_retention = 0.98;
        let make = |config| {
            Simulator::new(
                topology.clone(),
                vec![1.0; 25],
                config,
                2,
                MathKernelRequest::Scalar,
            )
            .unwrap()
        };
        let targets = repeated_frames(&target, 20);
        let mut acceleration = vec![0.0; 60];
        acceleration[2] = 60.0;
        let mut low = make(low_config);
        let mut high = make(high_config);
        low.simulate_chunk(&targets, Some(&acceleration)).unwrap();
        high.simulate_chunk(&targets, Some(&acceleration)).unwrap();
        let low_energy: f32 = low.residual.iter().map(|value| value * value).sum();
        let high_energy: f32 = high.residual.iter().map(|value| value * value).sum();
        assert!(
            low_energy < high_energy,
            "{low_energy} should be below {high_energy}"
        );
    }

    #[test]
    fn rotation_and_translation_are_equivalent() {
        let (topology, target) = grid(4, 6);
        let rotate = |value: [f32; 3]| [-value[1], value[0], value[2]];
        let transformed: Vec<f32> = target
            .chunks_exact(3)
            .flat_map(|point| {
                let rotated = rotate([point[0], point[1], point[2]]);
                [rotated[0] + 0.2, rotated[1] - 0.1, rotated[2] + 0.7]
            })
            .collect();
        let frames_a = repeated_frames(&target, 7);
        let frames_b = repeated_frames(&transformed, 7);
        let acceleration_a = [11.0, -7.0, 3.0].repeat(7);
        let rotated_acceleration = rotate([11.0, -7.0, 3.0]);
        let acceleration_b = rotated_acceleration.repeat(7);
        let mut a = simulator(
            topology.clone(),
            vec![1.0; 24],
            2,
            MathKernelRequest::Scalar,
        );
        let mut b = simulator(topology, vec![1.0; 24], 2, MathKernelRequest::Scalar);
        let output_a = a.simulate_chunk(&frames_a, Some(&acceleration_a)).unwrap();
        let output_b = b.simulate_chunk(&frames_b, Some(&acceleration_b)).unwrap();
        for (point_a, point_b) in output_a.chunks_exact(3).zip(output_b.chunks_exact(3)) {
            let expected = rotate([point_a[0], point_a[1], point_a[2]]);
            let expected = [expected[0] + 0.2, expected[1] - 0.1, expected[2] + 0.7];
            for axis in 0..3 {
                assert!((expected[axis] - point_b[axis]).abs() < 3.0e-6);
            }
        }
    }

    #[test]
    fn configuration_and_input_shapes_are_strictly_validated() {
        let (topology, target) = grid(2, 2);
        let invalid_config = PhysicsConfig {
            frames_per_second: f32::INFINITY,
            ..PhysicsConfig::default()
        };
        assert!(
            Simulator::new(
                topology.clone(),
                vec![1.0; 4],
                invalid_config,
                1,
                MathKernelRequest::Scalar
            )
            .is_err()
        );
        assert!(
            Simulator::new(
                topology.clone(),
                vec![1.0, 1.0],
                PhysicsConfig::default(),
                1,
                MathKernelRequest::Scalar
            )
            .is_err()
        );
        assert!(
            Simulator::new(
                topology.clone(),
                vec![1.0; 4],
                PhysicsConfig::default(),
                0,
                MathKernelRequest::Scalar
            )
            .is_err()
        );
        let mut solver = simulator(topology, vec![1.0; 4], 1, MathKernelRequest::Scalar);
        assert!(
            solver
                .simulate_chunk(&target[..target.len() - 1], None)
                .is_err()
        );
        assert!(solver.simulate_chunk(&target, Some(&[0.0, 0.0])).is_err());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn full_neon_solver_matches_scalar_within_tolerance() {
        let (topology, target) = grid(9, 11);
        let targets = repeated_frames(&target, 11);
        let mut acceleration = vec![0.0; 33];
        acceleration[2] = 41.0;
        acceleration[17] = -28.0;
        let mut scalar = simulator(
            topology.clone(),
            vec![1.0; 99],
            3,
            MathKernelRequest::Scalar,
        );
        let mut neon = simulator(topology, vec![1.0; 99], 3, MathKernelRequest::Neon);
        let expected = scalar
            .simulate_chunk(&targets, Some(&acceleration))
            .unwrap();
        let actual = neon.simulate_chunk(&targets, Some(&acceleration)).unwrap();
        assert_eq!(neon.selected_kernel(), MathKernelKind::NeonIntrinsics);
        for (left, right) in expected.iter().zip(actual) {
            assert!((left - right).abs() <= 2.0e-6);
        }
    }
}
