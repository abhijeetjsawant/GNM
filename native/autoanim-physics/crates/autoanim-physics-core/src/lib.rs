//! Deterministic, target-relative secondary-motion physics for facial surfaces.
//!
//! The solver stores dynamics as a residual from each animation target. An
//! unforced zero residual therefore follows an arbitrarily moving target
//! exactly instead of lagging behind the authored animation.

mod kernel;
mod solver;
mod topology;

pub use kernel::{Capabilities, MathKernelKind, MathKernelRequest, apply_inertial_predictor};
pub use solver::{PhysicsConfig, SimulationReport, Simulator};
pub use topology::{Edge, Neighbor, PhysicsTopology};

use thiserror::Error;

/// Errors are deliberately specific so every boundary can reject malformed
/// data before mutating simulation state.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum PhysicsError {
    #[error("{0}")]
    InvalidConfig(String),
    #[error("{0}")]
    InvalidTopology(String),
    #[error("{0}")]
    InvalidInput(String),
    #[error("requested math kernel is unavailable: {0}")]
    KernelUnavailable(String),
    #[error("could not create the dedicated worker pool: {0}")]
    ThreadPool(String),
}

pub(crate) type Vec3 = [f32; 3];

#[inline]
pub(crate) fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub(crate) fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(crate) fn scale(a: Vec3, s: f32) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub(crate) fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

#[inline]
pub(crate) fn length(a: Vec3) -> f32 {
    dot(a, a).sqrt()
}

#[inline]
pub(crate) fn lerp(a: Vec3, b: Vec3, alpha: f32) -> Vec3 {
    add(a, scale(sub(b, a), alpha))
}
