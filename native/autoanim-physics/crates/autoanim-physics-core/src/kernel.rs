use serde::{Deserialize, Serialize};

use crate::PhysicsError;

/// Caller preference for the contiguous inertial predictor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MathKernelRequest {
    #[default]
    Auto,
    Scalar,
    StableAutoVectorized,
    Neon,
}

/// The concrete implementation selected at simulator construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MathKernelKind {
    ScalarReference,
    StableAutoVectorized,
    NeonIntrinsics,
}

impl MathKernelKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::ScalarReference => "scalar_reference",
            Self::StableAutoVectorized => "stable_auto_vectorized",
            Self::NeonIntrinsics => "neon_intrinsics",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    pub architecture: String,
    pub logical_parallelism: usize,
    pub implemented_kernels: Vec<MathKernelKind>,
    pub neon_intrinsics_compiled: bool,
    pub production_simd_qualified: bool,
    pub deterministic_execution: String,
}

impl Capabilities {
    pub fn detect() -> Self {
        let mut implemented_kernels = vec![
            MathKernelKind::ScalarReference,
            MathKernelKind::StableAutoVectorized,
        ];
        if cfg!(target_arch = "aarch64") {
            implemented_kernels.push(MathKernelKind::NeonIntrinsics);
        }
        Self {
            architecture: std::env::consts::ARCH.into(),
            logical_parallelism: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            implemented_kernels,
            neon_intrinsics_compiled: cfg!(target_arch = "aarch64"),
            production_simd_qualified: false,
            deterministic_execution:
                "edge-parallel Jacobi phases with fixed edge-index CSR gathers; no parallel floating-point reductions"
                    .into(),
        }
    }
}

pub(crate) fn select_kernel(request: MathKernelRequest) -> Result<MathKernelKind, PhysicsError> {
    match request {
        MathKernelRequest::Scalar => Ok(MathKernelKind::ScalarReference),
        MathKernelRequest::StableAutoVectorized => Ok(MathKernelKind::StableAutoVectorized),
        MathKernelRequest::Auto => Ok(MathKernelKind::StableAutoVectorized),
        MathKernelRequest::Neon => {
            if cfg!(target_arch = "aarch64") {
                Ok(MathKernelKind::NeonIntrinsics)
            } else {
                Err(PhysicsError::KernelUnavailable(
                    "NEON intrinsics require an aarch64 build".into(),
                ))
            }
        }
    }
}

pub(crate) fn predict(
    kind: MathKernelKind,
    current: &[f32],
    previous: &[f32],
    output: &mut [f32],
    retention: f32,
) {
    debug_assert_eq!(current.len(), previous.len());
    debug_assert_eq!(current.len(), output.len());
    match kind {
        MathKernelKind::ScalarReference => predict_scalar(current, previous, output, retention),
        MathKernelKind::StableAutoVectorized => {
            predict_auto_vectorized(current, previous, output, retention)
        }
        MathKernelKind::NeonIntrinsics => predict_neon(current, previous, output, retention),
    }
}

/// Applies one contiguous inertial prediction pass. This low-level entry point
/// is public so release benchmarks can qualify individual math kernels without
/// conflating them with constraints, hashing, or report generation.
pub fn apply_inertial_predictor(
    kind: MathKernelKind,
    current: &[f32],
    previous: &[f32],
    output: &mut [f32],
    retention: f32,
) -> Result<(), PhysicsError> {
    if current.len() != previous.len() || current.len() != output.len() {
        return Err(PhysicsError::InvalidInput(
            "predictor slices must have equal lengths".into(),
        ));
    }
    if !retention.is_finite() || !(0.0..=1.0).contains(&retention) {
        return Err(PhysicsError::InvalidInput(
            "predictor retention must be finite and in 0..=1".into(),
        ));
    }
    if kind == MathKernelKind::NeonIntrinsics && !cfg!(target_arch = "aarch64") {
        return Err(PhysicsError::KernelUnavailable(
            "NEON intrinsics require an aarch64 build".into(),
        ));
    }
    predict(kind, current, previous, output, retention);
    Ok(())
}

fn predict_scalar(current: &[f32], previous: &[f32], output: &mut [f32], retention: f32) {
    for index in 0..current.len() {
        output[index] = (current[index] - previous[index]).mul_add(retention, current[index]);
    }
}

/// Stable-Rust contiguous loop deliberately shaped for LLVM auto-vectorization.
fn predict_auto_vectorized(current: &[f32], previous: &[f32], output: &mut [f32], retention: f32) {
    for ((out, &now), &before) in output.iter_mut().zip(current).zip(previous) {
        *out = (now - before).mul_add(retention, now);
    }
}

#[cfg(target_arch = "aarch64")]
fn predict_neon(current: &[f32], previous: &[f32], output: &mut [f32], retention: f32) {
    // SAFETY: aarch64 guarantees NEON. The callee bounds every 4-lane access
    // and handles the scalar tail before returning.
    unsafe { predict_neon_inner(current, previous, output, retention) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn predict_neon_inner(
    current: &[f32],
    previous: &[f32],
    output: &mut [f32],
    retention: f32,
) {
    use core::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32, vsubq_f32};

    let lanes = current.len() / 4 * 4;
    // SAFETY: pointers are derived from slices of equal length, and every
    // iteration accesses exactly four initialized elements below `lanes`.
    unsafe {
        let factor = vdupq_n_f32(retention);
        for index in (0..lanes).step_by(4) {
            let now = vld1q_f32(current.as_ptr().add(index));
            let before = vld1q_f32(previous.as_ptr().add(index));
            let result = vaddq_f32(now, vmulq_f32(vsubq_f32(now, before), factor));
            vst1q_f32(output.as_mut_ptr().add(index), result);
        }
    }
    for index in lanes..current.len() {
        output[index] = (current[index] - previous[index]).mul_add(retention, current[index]);
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn predict_neon(_: &[f32], _: &[f32], _: &mut [f32], _: f32) {
    unreachable!("NEON is rejected during kernel selection on non-aarch64 targets")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> (Vec<f32>, Vec<f32>) {
        let current = (0..103).map(|value| value as f32 * 0.013 - 0.5).collect();
        let previous = (0..103).map(|value| value as f32 * -0.007 + 0.2).collect();
        (current, previous)
    }

    #[test]
    fn scalar_and_auto_vectorized_are_identical() {
        let (current, previous) = input();
        let mut scalar = vec![0.0; current.len()];
        let mut vectorized = scalar.clone();
        predict_scalar(&current, &previous, &mut scalar, 0.91);
        predict_auto_vectorized(&current, &previous, &mut vectorized, 0.91);
        assert_eq!(scalar, vectorized);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_within_rounding_tolerance() {
        let (current, previous) = input();
        let mut scalar = vec![0.0; current.len()];
        let mut neon = scalar.clone();
        predict_scalar(&current, &previous, &mut scalar, 0.91);
        predict_neon(&current, &previous, &mut neon, 0.91);
        for (expected, actual) in scalar.iter().zip(neon) {
            assert!((expected - actual).abs() <= 2.0e-7);
        }
    }

    #[test]
    fn capabilities_do_not_overclaim_neon() {
        let capabilities = Capabilities::detect();
        assert_eq!(
            capabilities.neon_intrinsics_compiled,
            cfg!(target_arch = "aarch64")
        );
        assert_eq!(
            capabilities
                .implemented_kernels
                .contains(&MathKernelKind::NeonIntrinsics),
            cfg!(target_arch = "aarch64")
        );
        assert!(!capabilities.production_simd_qualified);
    }
}
