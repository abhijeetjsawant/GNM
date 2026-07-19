//! Minimal panic-safe C ABI for Swift, Objective-C, and other native hosts.

use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::slice;
use std::sync::Arc;

use autoanim_physics_core::{
    MathKernelRequest, PhysicsConfig, PhysicsError, PhysicsTopology, Simulator,
};

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("no error").expect("literal has no NUL"));
}

pub struct AaPhysicsTopology(Arc<PhysicsTopology>);
pub struct AaPhysicsSimulator(Simulator);

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AaPhysicsConfig {
    pub frames_per_second: f32,
    pub substeps: u32,
    pub iterations: u32,
    pub velocity_retention: f32,
    pub stretch_compliance: f32,
    pub tether_compliance: f32,
    pub max_displacement_m: f32,
    pub jacobi_relaxation: f32,
}

impl Default for AaPhysicsConfig {
    fn default() -> Self {
        PhysicsConfig::default().into()
    }
}

impl From<AaPhysicsConfig> for PhysicsConfig {
    fn from(value: AaPhysicsConfig) -> Self {
        Self {
            frames_per_second: value.frames_per_second,
            substeps: value.substeps,
            iterations: value.iterations,
            velocity_retention: value.velocity_retention,
            stretch_compliance: value.stretch_compliance,
            tether_compliance: value.tether_compliance,
            max_displacement_m: value.max_displacement_m,
            jacobi_relaxation: value.jacobi_relaxation,
        }
    }
}

impl From<PhysicsConfig> for AaPhysicsConfig {
    fn from(value: PhysicsConfig) -> Self {
        Self {
            frames_per_second: value.frames_per_second,
            substeps: value.substeps,
            iterations: value.iterations,
            velocity_retention: value.velocity_retention,
            stretch_compliance: value.stretch_compliance,
            tether_compliance: value.tether_compliance,
            max_displacement_m: value.max_displacement_m,
            jacobi_relaxation: value.jacobi_relaxation,
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn aa_physics_default_config() -> AaPhysicsConfig {
    AaPhysicsConfig::default()
}

/// Returns a thread-local message valid until the next failing ABI call on the
/// same thread.
#[unsafe(no_mangle)]
pub extern "C" fn aa_physics_last_error_message() -> *const c_char {
    LAST_ERROR.with(|message| message.borrow().as_ptr())
}

/// Creates canonical topology from flattened triangle indices.
///
/// # Safety
/// `triangles_xyz` must reference `triangle_count * 3` readable `u32` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aa_physics_topology_create(
    vertex_count: usize,
    triangles_xyz: *const u32,
    triangle_count: usize,
) -> *mut AaPhysicsTopology {
    ffi_pointer(|| {
        if triangles_xyz.is_null() {
            return Err(PhysicsError::InvalidInput(
                "triangles pointer is null".into(),
            ));
        }
        let scalar_count = triangle_count
            .checked_mul(3)
            .ok_or_else(|| PhysicsError::InvalidInput("triangle count overflow".into()))?;
        // SAFETY: the caller contract guarantees this readable range.
        let flattened = unsafe { slice::from_raw_parts(triangles_xyz, scalar_count) };
        let triangles: Vec<[u32; 3]> = flattened
            .chunks_exact(3)
            .map(|value| [value[0], value[1], value[2]])
            .collect();
        let topology = PhysicsTopology::from_triangles(vertex_count, &triangles)?;
        Ok(Box::into_raw(Box::new(AaPhysicsTopology(Arc::new(
            topology,
        )))))
    })
}

/// # Safety
/// `topology` must be null or a live pointer returned by this library, and it
/// may be destroyed exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aa_physics_topology_destroy(topology: *mut AaPhysicsTopology) {
    if !topology.is_null() {
        // SAFETY: required by the function contract.
        drop(unsafe { Box::from_raw(topology) });
    }
}

/// Returns the canonical topology vertex count, or zero for a null pointer.
///
/// # Safety
/// `topology` must be null or a live pointer returned by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aa_physics_topology_vertex_count(
    topology: *const AaPhysicsTopology,
) -> usize {
    if topology.is_null() {
        0
    } else {
        // SAFETY: required by the function contract.
        unsafe { &*topology }.0.vertex_count()
    }
}

/// Returns the canonical unique edge count, or zero for a null pointer.
///
/// # Safety
/// `topology` must be null or a live pointer returned by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aa_physics_topology_edge_count(
    topology: *const AaPhysicsTopology,
) -> usize {
    if topology.is_null() {
        0
    } else {
        // SAFETY: required by the function contract.
        unsafe { &*topology }.0.edges().len()
    }
}

/// Constructs a stateful solver. Kernel values are 0=auto, 1=scalar,
/// 2=stable-auto-vectorized, and 3=NEON.
///
/// # Safety
/// `topology` must be live and `motion_weights` must reference `weight_count`
/// readable floats for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aa_physics_simulator_create(
    topology: *const AaPhysicsTopology,
    motion_weights: *const f32,
    weight_count: usize,
    config: AaPhysicsConfig,
    threads: usize,
    kernel: u32,
) -> *mut AaPhysicsSimulator {
    ffi_pointer(|| {
        if topology.is_null() || motion_weights.is_null() {
            return Err(PhysicsError::InvalidInput(
                "topology or weights pointer is null".into(),
            ));
        }
        let request = match kernel {
            0 => MathKernelRequest::Auto,
            1 => MathKernelRequest::Scalar,
            2 => MathKernelRequest::StableAutoVectorized,
            3 => MathKernelRequest::Neon,
            _ => {
                return Err(PhysicsError::InvalidInput(format!(
                    "unknown kernel value {kernel}"
                )));
            }
        };
        // SAFETY: pointers and ranges are guaranteed by the caller contract.
        let topology = unsafe { &*topology }.0.clone();
        let weights = unsafe { slice::from_raw_parts(motion_weights, weight_count) }.to_vec();
        let simulator = Simulator::new(topology, weights, config.into(), threads, request)?;
        Ok(Box::into_raw(Box::new(AaPhysicsSimulator(simulator))))
    })
}

/// # Safety
/// `simulator` must be null or a live pointer returned by this library, and it
/// may be destroyed exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aa_physics_simulator_destroy(simulator: *mut AaPhysicsSimulator) {
    if !simulator.is_null() {
        // SAFETY: required by the function contract.
        drop(unsafe { Box::from_raw(simulator) });
    }
}

/// Simulates a chunk into caller-owned memory. Returns 0 on success and -1 on
/// failure. A null acceleration pointer selects no forcing.
///
/// # Safety
/// The simulator must be live, target and optional acceleration ranges must be
/// readable, and output must reference `output_count` writable floats. Ranges
/// must not alias the mutable simulator or each other incompatibly.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aa_physics_simulate_chunk(
    simulator: *mut AaPhysicsSimulator,
    targets: *const f32,
    target_count: usize,
    accelerations: *const f32,
    acceleration_count: usize,
    output: *mut f32,
    output_count: usize,
) -> i32 {
    ffi_status(|| {
        if simulator.is_null() || targets.is_null() || output.is_null() {
            return Err(PhysicsError::InvalidInput(
                "simulator, targets, or output pointer is null".into(),
            ));
        }
        if output_count != target_count {
            return Err(PhysicsError::InvalidInput(format!(
                "output_count is {output_count}, expected {target_count}"
            )));
        }
        // SAFETY: pointer ranges are guaranteed by the caller contract.
        let simulator = unsafe { &mut *simulator };
        let targets = unsafe { slice::from_raw_parts(targets, target_count) };
        let accelerations = if accelerations.is_null() {
            if acceleration_count != 0 {
                return Err(PhysicsError::InvalidInput(
                    "null acceleration pointer requires acceleration_count 0".into(),
                ));
            }
            None
        } else {
            Some(unsafe { slice::from_raw_parts(accelerations, acceleration_count) })
        };
        let result = simulator.0.simulate_chunk(targets, accelerations)?;
        // SAFETY: output has exactly result.len() writable elements.
        unsafe { ptr::copy_nonoverlapping(result.as_ptr(), output, result.len()) };
        Ok(())
    })
}

/// Writes a NUL-terminated JSON report and returns the required byte count,
/// including the terminator. Pass a null buffer to query the required size.
/// Returns 0 on failure.
///
/// # Safety
/// The simulator must be live. A non-null buffer must reference `capacity`
/// writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aa_physics_report_json(
    simulator: *const AaPhysicsSimulator,
    buffer: *mut c_char,
    capacity: usize,
) -> usize {
    match catch_unwind(AssertUnwindSafe(|| {
        if simulator.is_null() {
            return Err("simulator pointer is null".to_string());
        }
        // SAFETY: guaranteed by the caller contract.
        let report = unsafe { &*simulator }.0.report();
        let json = serde_json::to_string(&report).map_err(|error| error.to_string())?;
        let required = json.len() + 1;
        if !buffer.is_null() {
            if capacity < required {
                return Err(format!(
                    "report buffer needs {required} bytes, received {capacity}"
                ));
            }
            // SAFETY: capacity has been checked and JSON contains valid bytes.
            unsafe {
                ptr::copy_nonoverlapping(json.as_ptr().cast::<c_char>(), buffer, json.len());
                *buffer.add(json.len()) = 0;
            }
        }
        Ok(required)
    })) {
        Ok(Ok(required)) => required,
        Ok(Err(error)) => {
            set_error(error);
            0
        }
        Err(_) => {
            set_error("panic caught at C ABI boundary");
            0
        }
    }
}

fn ffi_pointer<T>(operation: impl FnOnce() -> Result<*mut T, PhysicsError>) -> *mut T {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(pointer)) => pointer,
        Ok(Err(error)) => {
            set_error(error.to_string());
            ptr::null_mut()
        }
        Err(_) => {
            set_error("panic caught at C ABI boundary");
            ptr::null_mut()
        }
    }
}

fn ffi_status(operation: impl FnOnce() -> Result<(), PhysicsError>) -> i32 {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_error(error.to_string());
            -1
        }
        Err(_) => {
            set_error("panic caught at C ABI boundary");
            -1
        }
    }
}

fn set_error(message: impl AsRef<str>) {
    let sanitized = message.as_ref().replace('\0', "�");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(sanitized).expect("NUL characters were replaced");
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_abi_smoke_test_and_report() {
        let triangles = [0_u32, 1, 2];
        let topology = unsafe { aa_physics_topology_create(3, triangles.as_ptr(), 1) };
        assert!(!topology.is_null());
        assert_eq!(unsafe { aa_physics_topology_vertex_count(topology) }, 3);
        assert_eq!(unsafe { aa_physics_topology_edge_count(topology) }, 3);
        let weights = [1.0_f32; 3];
        let simulator = unsafe {
            aa_physics_simulator_create(
                topology,
                weights.as_ptr(),
                weights.len(),
                aa_physics_default_config(),
                2,
                1,
            )
        };
        assert!(!simulator.is_null());
        let target = [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut output = [f32::NAN; 9];
        let status = unsafe {
            aa_physics_simulate_chunk(
                simulator,
                target.as_ptr(),
                target.len(),
                ptr::null(),
                0,
                output.as_mut_ptr(),
                output.len(),
            )
        };
        assert_eq!(status, 0);
        assert_eq!(output, target);
        let required = unsafe { aa_physics_report_json(simulator, ptr::null_mut(), 0) };
        assert!(required > 1);
        let mut report = vec![0_i8; required];
        assert_eq!(
            unsafe { aa_physics_report_json(simulator, report.as_mut_ptr(), report.len()) },
            required
        );
        unsafe {
            aa_physics_simulator_destroy(simulator);
            aa_physics_topology_destroy(topology);
        }
    }

    #[test]
    fn c_abi_null_and_length_errors_are_contained() {
        let topology = unsafe { aa_physics_topology_create(3, ptr::null(), 1) };
        assert!(topology.is_null());
        assert!(!aa_physics_last_error_message().is_null());
        assert_eq!(unsafe { aa_physics_topology_vertex_count(ptr::null()) }, 0);
        assert_eq!(unsafe { aa_physics_topology_edge_count(ptr::null()) }, 0);
        assert_eq!(
            unsafe {
                aa_physics_simulate_chunk(
                    ptr::null_mut(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    0,
                    ptr::null_mut(),
                    0,
                )
            },
            -1
        );
        assert_eq!(
            unsafe { aa_physics_report_json(ptr::null(), ptr::null_mut(), 0) },
            0
        );
        unsafe {
            aa_physics_topology_destroy(ptr::null_mut());
            aa_physics_simulator_destroy(ptr::null_mut());
        }
    }
}
