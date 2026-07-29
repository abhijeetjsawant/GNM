use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{PhysicsError, Vec3, add, dot, length, lerp, scale, sub};

const EPSILON: f32 = 1.0e-12;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoftContactConfig {
    pub frames_per_second: f32,
    pub substeps: u32,
    pub iterations: u32,
    pub edge_compliance: f32,
    pub volume_compliance: f32,
    pub tether_compliance: f32,
    pub contact_compliance: f32,
    pub contact_thickness_m: f32,
    pub contact_activation_distance_m: f32,
    pub max_displacement_m: f32,
}

impl Default for SoftContactConfig {
    fn default() -> Self {
        Self {
            frames_per_second: 30.0,
            substeps: 8,
            iterations: 16,
            edge_compliance: 2.0e-7,
            volume_compliance: 0.0,
            tether_compliance: 2.0e-5,
            contact_compliance: 1.0e-9,
            contact_thickness_m: 0.001,
            contact_activation_distance_m: 0.006,
            max_displacement_m: 0.012,
        }
    }
}

impl SoftContactConfig {
    pub fn validate(&self) -> Result<(), PhysicsError> {
        finite_positive("frames_per_second", self.frames_per_second, 480.0)?;
        if !(1..=32).contains(&self.substeps) {
            return Err(PhysicsError::InvalidConfig(
                "soft-contact substeps must be in 1..=32".into(),
            ));
        }
        if !(1..=128).contains(&self.iterations) {
            return Err(PhysicsError::InvalidConfig(
                "soft-contact iterations must be in 1..=128".into(),
            ));
        }
        finite_nonnegative("edge_compliance", self.edge_compliance)?;
        finite_nonnegative("volume_compliance", self.volume_compliance)?;
        finite_nonnegative("tether_compliance", self.tether_compliance)?;
        finite_nonnegative("contact_compliance", self.contact_compliance)?;
        finite_positive("contact_thickness_m", self.contact_thickness_m, 0.02)?;
        finite_positive(
            "contact_activation_distance_m",
            self.contact_activation_distance_m,
            0.03,
        )?;
        if self.contact_activation_distance_m <= self.contact_thickness_m {
            return Err(PhysicsError::InvalidConfig(
                "contact_activation_distance_m must exceed contact_thickness_m".into(),
            ));
        }
        finite_positive("max_displacement_m", self.max_displacement_m, 0.1)?;
        Ok(())
    }
}

fn finite_positive(name: &str, value: f32, maximum: f32) -> Result<(), PhysicsError> {
    if !value.is_finite() || value <= 0.0 || value > maximum {
        return Err(PhysicsError::InvalidConfig(format!(
            "{name} must be finite, greater than zero, and at most {maximum}"
        )));
    }
    Ok(())
}

fn finite_nonnegative(name: &str, value: f32) -> Result<(), PhysicsError> {
    if !value.is_finite() || value < 0.0 {
        return Err(PhysicsError::InvalidConfig(format!(
            "{name} must be finite and nonnegative"
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct Edge {
    a: usize,
    b: usize,
}

#[derive(Clone, Copy, Debug)]
struct Tetrahedron {
    vertices: [usize; 4],
}

#[derive(Clone, Copy, Debug)]
struct ContactPair {
    point: usize,
    triangle: [usize; 3],
    side: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoftContactReport {
    pub schema_version: u32,
    pub backend: String,
    pub frame_count: u64,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub tetrahedron_count: usize,
    pub contact_pair_count: usize,
    pub substeps: u32,
    pub iterations: u32,
    pub input_sha256: String,
    pub output_sha256: String,
    pub contact_projection_count: u64,
    pub minimum_contact_separation_m: f32,
    pub maximum_displacement_m: f32,
    pub minimum_tetrahedron_volume_ratio: f32,
    pub maximum_tetrahedron_volume_ratio: f32,
    pub inverted_tetrahedron_samples: u64,
    pub finite: bool,
    pub target_relative: bool,
    pub continuous_collision_detection: bool,
}

/// Deterministic, target-relative XPBD solver for one deformable volume and a
/// deformable contact surface.
pub struct SoftContactSimulator {
    config: SoftContactConfig,
    rest_positions: Vec<Vec3>,
    inverse_masses: Vec<f32>,
    edges: Vec<Edge>,
    tetrahedra: Vec<Tetrahedron>,
    contacts: Vec<ContactPair>,
    positions: Vec<Vec3>,
    last_target: Vec<Vec3>,
    current_target: Vec<Vec3>,
    interpolated_target: Vec<Vec3>,
    previous_interpolated_target: Vec<Vec3>,
    edge_lambdas: Vec<f32>,
    volume_lambdas: Vec<f32>,
    tether_lambdas: Vec<Vec3>,
    contact_lambdas: Vec<f32>,
    initialized: bool,
    input_hasher: Sha256,
    output_hasher: Sha256,
    frame_count: u64,
    contact_projection_count: u64,
    minimum_contact_separation: f32,
    maximum_displacement: f32,
    minimum_volume_ratio: f32,
    maximum_volume_ratio: f32,
    inverted_tetrahedron_samples: u64,
    finite: bool,
}

impl SoftContactSimulator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rest_positions: Vec<Vec3>,
        surface_triangles: Vec<[u32; 3]>,
        tetrahedra: Vec<[u32; 4]>,
        contact_pairs: Vec<[u32; 4]>,
        inverse_masses: Vec<f32>,
        config: SoftContactConfig,
    ) -> Result<Self, PhysicsError> {
        config.validate()?;
        let vertex_count = rest_positions.len();
        if vertex_count == 0 || vertex_count > u32::MAX as usize {
            return Err(PhysicsError::InvalidTopology(
                "soft-contact vertex count must be in 1..=u32::MAX".into(),
            ));
        }
        validate_positions("rest_positions", &rest_positions)?;
        if inverse_masses.len() != vertex_count {
            return Err(PhysicsError::InvalidInput(format!(
                "inverse_masses has length {}, expected {vertex_count}",
                inverse_masses.len()
            )));
        }
        for (index, value) in inverse_masses.iter().copied().enumerate() {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(PhysicsError::InvalidInput(format!(
                    "inverse mass {index} must be finite and in 0..=1"
                )));
            }
        }
        if surface_triangles.is_empty() {
            return Err(PhysicsError::InvalidTopology(
                "at least one soft-contact surface triangle is required".into(),
            ));
        }
        let mut unique_edges = BTreeSet::new();
        for (index, triangle) in surface_triangles.iter().copied().enumerate() {
            validate_distinct_indices("surface triangle", index, &triangle, vertex_count)?;
            for [a, b] in [
                [triangle[0], triangle[1]],
                [triangle[1], triangle[2]],
                [triangle[2], triangle[0]],
            ] {
                unique_edges.insert(if a < b { [a, b] } else { [b, a] });
            }
        }
        if tetrahedra.is_empty() {
            return Err(PhysicsError::InvalidTopology(
                "at least one soft-contact tetrahedron is required".into(),
            ));
        }
        let mut validated_tetrahedra = Vec::with_capacity(tetrahedra.len());
        for (index, tetrahedron) in tetrahedra.iter().copied().enumerate() {
            validate_distinct_indices("tetrahedron", index, &tetrahedron, vertex_count)?;
            let vertices = tetrahedron.map(|value| value as usize);
            let signed = signed_six_volume(
                rest_positions[vertices[0]],
                rest_positions[vertices[1]],
                rest_positions[vertices[2]],
                rest_positions[vertices[3]],
            );
            if !signed.is_finite() || signed.abs() <= EPSILON {
                return Err(PhysicsError::InvalidTopology(format!(
                    "tetrahedron {index} has zero or non-finite rest volume"
                )));
            }
            validated_tetrahedra.push(Tetrahedron { vertices });
        }
        if contact_pairs.is_empty() {
            return Err(PhysicsError::InvalidTopology(
                "at least one soft-contact pair is required".into(),
            ));
        }
        let mut validated_contacts = Vec::with_capacity(contact_pairs.len());
        for (index, pair) in contact_pairs.iter().copied().enumerate() {
            let [point, a, b, c] = pair;
            validate_distinct_indices("contact triangle", index, &[a, b, c], vertex_count)?;
            if point as usize >= vertex_count {
                return Err(PhysicsError::InvalidTopology(format!(
                    "contact pair {index} references point {point}, outside 0..{vertex_count}"
                )));
            }
            if point == a || point == b || point == c {
                return Err(PhysicsError::InvalidTopology(format!(
                    "contact pair {index} uses its point in its triangle"
                )));
            }
            let triangle = [a as usize, b as usize, c as usize];
            let normal = cross(
                sub(rest_positions[triangle[1]], rest_positions[triangle[0]]),
                sub(rest_positions[triangle[2]], rest_positions[triangle[0]]),
            );
            if length(normal) <= EPSILON {
                return Err(PhysicsError::InvalidTopology(format!(
                    "contact pair {index} has a degenerate rest triangle"
                )));
            }
            validated_contacts.push(ContactPair {
                point: point as usize,
                triangle,
                // The tongue can legitimately travel around the open lip edge
                // and change triangle half-space without intersection. Lock
                // the permitted side only when this pair first enters the
                // contact activation shell.
                side: 0.0,
            });
        }
        validated_contacts.sort_by_key(|contact| {
            (
                contact.point,
                contact.triangle[0],
                contact.triangle[1],
                contact.triangle[2],
            )
        });
        let edges = unique_edges
            .into_iter()
            .map(|[a, b]| Edge {
                a: a as usize,
                b: b as usize,
            })
            .collect::<Vec<_>>();
        let edge_count = edges.len();
        let tetrahedron_count = validated_tetrahedra.len();
        let contact_count = validated_contacts.len();
        Ok(Self {
            config,
            rest_positions: rest_positions.clone(),
            inverse_masses,
            edges,
            tetrahedra: validated_tetrahedra,
            contacts: validated_contacts,
            positions: rest_positions.clone(),
            last_target: rest_positions.clone(),
            current_target: rest_positions.clone(),
            interpolated_target: rest_positions.clone(),
            previous_interpolated_target: rest_positions,
            edge_lambdas: vec![0.0; edge_count],
            volume_lambdas: vec![0.0; tetrahedron_count],
            tether_lambdas: vec![[0.0; 3]; vertex_count],
            contact_lambdas: vec![0.0; contact_count],
            initialized: false,
            input_hasher: Sha256::new(),
            output_hasher: Sha256::new(),
            frame_count: 0,
            contact_projection_count: 0,
            minimum_contact_separation: f32::INFINITY,
            maximum_displacement: 0.0,
            minimum_volume_ratio: f32::INFINITY,
            maximum_volume_ratio: 0.0,
            inverted_tetrahedron_samples: 0,
            finite: true,
        })
    }

    pub fn simulate_chunk(&mut self, targets: &[f32]) -> Result<Vec<f32>, PhysicsError> {
        let frame_width = self.rest_positions.len() * 3;
        if targets.is_empty() || !targets.len().is_multiple_of(frame_width) {
            return Err(PhysicsError::InvalidInput(format!(
                "soft-contact targets must contain a nonzero whole number of {frame_width}-float frames"
            )));
        }
        if targets.iter().any(|value| !value.is_finite()) {
            return Err(PhysicsError::InvalidInput(
                "soft-contact targets must contain only finite values".into(),
            ));
        }
        let frame_count = targets.len() / frame_width;
        let mut output = Vec::with_capacity(targets.len());
        for frame_index in 0..frame_count {
            let flat = &targets[frame_index * frame_width..(frame_index + 1) * frame_width];
            hash_f32_slice(&mut self.input_hasher, flat);
            for (position, value) in self.current_target.iter_mut().zip(flat.chunks_exact(3)) {
                *position = [value[0], value[1], value[2]];
            }
            self.simulate_frame();
            let output_start = output.len();
            for position in &self.positions {
                output.extend_from_slice(position);
            }
            hash_f32_slice(&mut self.output_hasher, &output[output_start..]);
            self.update_metrics();
            self.last_target.copy_from_slice(&self.current_target);
            self.frame_count += 1;
        }
        Ok(output)
    }

    fn simulate_frame(&mut self) {
        if !self.initialized {
            self.positions.copy_from_slice(&self.current_target);
            self.last_target.copy_from_slice(&self.current_target);
            self.interpolated_target
                .copy_from_slice(&self.current_target);
            self.previous_interpolated_target
                .copy_from_slice(&self.current_target);
            self.initialized = true;
        }
        let dt = 1.0 / self.config.frames_per_second / self.config.substeps as f32;
        let dt_squared = dt * dt;
        let edge_alpha = self.config.edge_compliance / dt_squared;
        let volume_alpha = self.config.volume_compliance / dt_squared;
        let tether_alpha = self.config.tether_compliance / dt_squared;
        let contact_alpha = self.config.contact_compliance / dt_squared;

        for substep in 0..self.config.substeps {
            self.previous_interpolated_target
                .copy_from_slice(&self.interpolated_target);
            let phase = (substep + 1) as f32 / self.config.substeps as f32;
            for ((target, &before), &now) in self
                .interpolated_target
                .iter_mut()
                .zip(&self.last_target)
                .zip(&self.current_target)
            {
                *target = lerp(before, now, phase);
            }
            for ((position, &before), &now) in self
                .positions
                .iter_mut()
                .zip(&self.previous_interpolated_target)
                .zip(&self.interpolated_target)
            {
                *position = add(*position, sub(now, before));
            }
            self.edge_lambdas.fill(0.0);
            self.volume_lambdas.fill(0.0);
            self.tether_lambdas.fill([0.0; 3]);
            self.contact_lambdas.fill(0.0);

            for _ in 0..self.config.iterations {
                self.solve_edges(edge_alpha);
                self.solve_tethers(tether_alpha);
                self.clamp_displacements();
                self.solve_contacts(contact_alpha);
                // Volume is last so a contact projection cannot leave a
                // tetrahedron inverted at the published substep boundary.
                // The 1 mm contact shell provides room for this final,
                // typically sub-0.1 mm, volume correction.
                self.solve_volumes(volume_alpha);
            }
        }
    }

    fn solve_edges(&mut self, alpha: f32) {
        for (index, edge) in self.edges.iter().copied().enumerate() {
            let delta = sub(self.positions[edge.a], self.positions[edge.b]);
            let current_length = length(delta);
            if current_length <= EPSILON {
                continue;
            }
            let target_length = length(sub(
                self.interpolated_target[edge.a],
                self.interpolated_target[edge.b],
            ));
            let constraint = current_length - target_length;
            let wa = self.inverse_masses[edge.a];
            let wb = self.inverse_masses[edge.b];
            let denominator = wa + wb + alpha;
            if denominator <= EPSILON {
                continue;
            }
            let delta_lambda = (-constraint - alpha * self.edge_lambdas[index]) / denominator;
            self.edge_lambdas[index] += delta_lambda;
            let gradient = scale(delta, 1.0 / current_length);
            self.positions[edge.a] =
                add(self.positions[edge.a], scale(gradient, wa * delta_lambda));
            self.positions[edge.b] =
                add(self.positions[edge.b], scale(gradient, -wb * delta_lambda));
        }
    }

    fn solve_volumes(&mut self, alpha: f32) {
        for (index, tetrahedron) in self.tetrahedra.iter().copied().enumerate() {
            let [i0, i1, i2, i3] = tetrahedron.vertices;
            let [p0, p1, p2, p3] = [
                self.positions[i0],
                self.positions[i1],
                self.positions[i2],
                self.positions[i3],
            ];
            let target_volume = signed_six_volume(
                self.interpolated_target[i0],
                self.interpolated_target[i1],
                self.interpolated_target[i2],
                self.interpolated_target[i3],
            );
            let constraint = signed_six_volume(p0, p1, p2, p3) - target_volume;
            let g1 = cross(sub(p2, p0), sub(p3, p0));
            let g2 = cross(sub(p3, p0), sub(p1, p0));
            let g3 = cross(sub(p1, p0), sub(p2, p0));
            let g0 = scale(add(add(g1, g2), g3), -1.0);
            let gradients = [g0, g1, g2, g3];
            let indices = [i0, i1, i2, i3];
            let denominator =
                indices
                    .iter()
                    .zip(gradients)
                    .fold(alpha, |sum, (&vertex, gradient)| {
                        sum + self.inverse_masses[vertex] * dot(gradient, gradient)
                    });
            if denominator <= EPSILON {
                continue;
            }
            let delta_lambda = (-constraint - alpha * self.volume_lambdas[index]) / denominator;
            self.volume_lambdas[index] += delta_lambda;
            for (&vertex, gradient) in indices.iter().zip(gradients) {
                self.positions[vertex] = add(
                    self.positions[vertex],
                    scale(gradient, self.inverse_masses[vertex] * delta_lambda),
                );
            }
        }
    }

    fn solve_tethers(&mut self, alpha: f32) {
        for index in 0..self.positions.len() {
            let weight = self.inverse_masses[index];
            if weight <= 0.0 {
                self.positions[index] = self.interpolated_target[index];
                continue;
            }
            let denominator = weight + alpha;
            for axis in 0..3 {
                let constraint =
                    self.positions[index][axis] - self.interpolated_target[index][axis];
                let delta_lambda =
                    (-constraint - alpha * self.tether_lambdas[index][axis]) / denominator;
                self.tether_lambdas[index][axis] += delta_lambda;
                self.positions[index][axis] += weight * delta_lambda;
            }
        }
    }

    fn clamp_displacements(&mut self) {
        for index in 0..self.positions.len() {
            let displacement = sub(self.positions[index], self.interpolated_target[index]);
            let magnitude = length(displacement);
            if magnitude > self.config.max_displacement_m {
                self.positions[index] = add(
                    self.interpolated_target[index],
                    scale(displacement, self.config.max_displacement_m / magnitude),
                );
            }
        }
    }

    fn solve_contacts(&mut self, alpha: f32) {
        let mut group_start = 0;
        while group_start < self.contacts.len() {
            let point = self.contacts[group_start].point;
            let mut group_end = group_start + 1;
            while group_end < self.contacts.len() && self.contacts[group_end].point == point {
                group_end += 1;
            }
            let mut selected: Option<(usize, f32)> = None;
            for index in group_start..group_end {
                let contact = self.contacts[index];
                let [ia, ib, ic] = contact.triangle;
                let (closest, _) = closest_point_barycentric(
                    self.positions[point],
                    self.positions[ia],
                    self.positions[ib],
                    self.positions[ic],
                );
                let distance = length(sub(self.positions[point], closest));
                if distance <= self.config.contact_activation_distance_m
                    && selected.is_none_or(|(_, best)| distance < best)
                {
                    selected = Some((index, distance));
                }
            }
            let Some((index, _)) = selected else {
                for lambda in &mut self.contact_lambdas[group_start..group_end] {
                    *lambda = 0.0;
                }
                group_start = group_end;
                continue;
            };
            for other in group_start..group_end {
                if other != index {
                    self.contact_lambdas[other] = 0.0;
                }
            }
            let mut contact = self.contacts[index];
            let [ia, ib, ic] = contact.triangle;
            let a = self.positions[ia];
            let b = self.positions[ib];
            let c = self.positions[ic];
            let raw_normal = cross(sub(b, a), sub(c, a));
            let normal_length = length(raw_normal);
            if normal_length <= EPSILON {
                group_start = group_end;
                continue;
            }
            if contact.side == 0.0 {
                let signed = dot(sub(self.positions[contact.point], a), raw_normal);
                contact.side = if signed < 0.0 { -1.0 } else { 1.0 };
                self.contacts[index].side = contact.side;
            }
            let normal = scale(raw_normal, contact.side / normal_length);
            let (closest, barycentric) =
                closest_point_barycentric(self.positions[contact.point], a, b, c);
            debug_assert!(
                length(sub(self.positions[contact.point], closest))
                    <= self.config.contact_activation_distance_m + 1.0e-6
            );
            let plane_distance = dot(sub(self.positions[contact.point], a), normal);
            let constraint = plane_distance - self.config.contact_thickness_m;
            if constraint >= 0.0 && self.contact_lambdas[index] <= 0.0 {
                group_start = group_end;
                continue;
            }
            let wp = self.inverse_masses[contact.point];
            let wa = self.inverse_masses[ia];
            let wb = self.inverse_masses[ib];
            let wc = self.inverse_masses[ic];
            let denominator = wp
                + wa * barycentric[0] * barycentric[0]
                + wb * barycentric[1] * barycentric[1]
                + wc * barycentric[2] * barycentric[2]
                + alpha;
            if denominator <= EPSILON {
                group_start = group_end;
                continue;
            }
            let unconstrained = self.contact_lambdas[index]
                + (-constraint - alpha * self.contact_lambdas[index]) / denominator;
            let new_lambda = unconstrained.max(0.0);
            let delta_lambda = new_lambda - self.contact_lambdas[index];
            self.contact_lambdas[index] = new_lambda;
            if delta_lambda <= 0.0 {
                group_start = group_end;
                continue;
            }
            self.contact_projection_count += 1;
            self.positions[contact.point] = add(
                self.positions[contact.point],
                scale(normal, wp * delta_lambda),
            );
            for (vertex, weight, barycentric_weight) in [
                (ia, wa, barycentric[0]),
                (ib, wb, barycentric[1]),
                (ic, wc, barycentric[2]),
            ] {
                self.positions[vertex] = add(
                    self.positions[vertex],
                    scale(normal, -weight * barycentric_weight * delta_lambda),
                );
            }
            group_start = group_end;
        }
    }

    fn update_metrics(&mut self) {
        for (&position, &target) in self.positions.iter().zip(&self.current_target) {
            self.maximum_displacement =
                self.maximum_displacement.max(length(sub(position, target)));
            self.finite &= position.iter().all(|value| value.is_finite());
        }
        let mut group_start = 0;
        while group_start < self.contacts.len() {
            let point = self.contacts[group_start].point;
            let mut group_end = group_start + 1;
            while group_end < self.contacts.len() && self.contacts[group_end].point == point {
                group_end += 1;
            }
            let mut selected: Option<(ContactPair, f32)> = None;
            for contact in &self.contacts[group_start..group_end] {
                let [a, b, c] = contact.triangle.map(|index| self.positions[index]);
                let (closest, _) = closest_point_barycentric(self.positions[point], a, b, c);
                let distance = length(sub(self.positions[point], closest));
                if distance <= self.config.contact_activation_distance_m
                    && selected.is_none_or(|(_, best)| distance < best)
                {
                    selected = Some((*contact, distance));
                }
            }
            if let Some((contact, _)) = selected {
                if contact.side == 0.0 {
                    group_start = group_end;
                    continue;
                }
                let [a, b, c] = contact.triangle.map(|index| self.positions[index]);
                let raw_normal = cross(sub(b, a), sub(c, a));
                let normal_length = length(raw_normal);
                if normal_length > EPSILON {
                    let separation = dot(
                        sub(self.positions[point], a),
                        scale(raw_normal, contact.side / normal_length),
                    );
                    self.minimum_contact_separation =
                        self.minimum_contact_separation.min(separation);
                }
            }
            group_start = group_end;
        }
        for tetrahedron in &self.tetrahedra {
            let [i0, i1, i2, i3] = tetrahedron.vertices;
            let target = signed_six_volume(
                self.current_target[i0],
                self.current_target[i1],
                self.current_target[i2],
                self.current_target[i3],
            );
            let current = signed_six_volume(
                self.positions[i0],
                self.positions[i1],
                self.positions[i2],
                self.positions[i3],
            );
            if target.abs() > EPSILON {
                let ratio = current / target;
                self.minimum_volume_ratio = self.minimum_volume_ratio.min(ratio);
                self.maximum_volume_ratio = self.maximum_volume_ratio.max(ratio);
                if ratio <= 0.0 {
                    self.inverted_tetrahedron_samples += 1;
                }
            }
        }
    }

    pub fn report(&self) -> SoftContactReport {
        SoftContactReport {
            schema_version: 1,
            backend: "rust-xpbd-volumetric-soft-contact".into(),
            frame_count: self.frame_count,
            vertex_count: self.rest_positions.len(),
            edge_count: self.edges.len(),
            tetrahedron_count: self.tetrahedra.len(),
            contact_pair_count: self.contacts.len(),
            substeps: self.config.substeps,
            iterations: self.config.iterations,
            input_sha256: format!("{:x}", self.input_hasher.clone().finalize()),
            output_sha256: format!("{:x}", self.output_hasher.clone().finalize()),
            contact_projection_count: self.contact_projection_count,
            minimum_contact_separation_m: if self.minimum_contact_separation.is_finite() {
                self.minimum_contact_separation
            } else {
                0.0
            },
            maximum_displacement_m: self.maximum_displacement,
            minimum_tetrahedron_volume_ratio: if self.minimum_volume_ratio.is_finite() {
                self.minimum_volume_ratio
            } else {
                0.0
            },
            maximum_tetrahedron_volume_ratio: self.maximum_volume_ratio,
            inverted_tetrahedron_samples: self.inverted_tetrahedron_samples,
            finite: self.finite,
            target_relative: true,
            continuous_collision_detection: false,
        }
    }
}

fn validate_positions(name: &str, positions: &[Vec3]) -> Result<(), PhysicsError> {
    if positions
        .iter()
        .flat_map(|position| position.iter())
        .any(|value| !value.is_finite())
    {
        return Err(PhysicsError::InvalidInput(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}

fn validate_distinct_indices<const N: usize>(
    kind: &str,
    element_index: usize,
    indices: &[u32; N],
    vertex_count: usize,
) -> Result<(), PhysicsError> {
    for &vertex in indices {
        if vertex as usize >= vertex_count {
            return Err(PhysicsError::InvalidTopology(format!(
                "{kind} {element_index} references vertex {vertex}, outside 0..{vertex_count}"
            )));
        }
    }
    let distinct = indices.iter().copied().collect::<BTreeSet<_>>();
    if distinct.len() != N {
        return Err(PhysicsError::InvalidTopology(format!(
            "{kind} {element_index} is degenerate"
        )));
    }
    Ok(())
}

#[inline]
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1].mul_add(b[2], -a[2] * b[1]),
        a[2].mul_add(b[0], -a[0] * b[2]),
        a[0].mul_add(b[1], -a[1] * b[0]),
    ]
}

#[inline]
fn signed_six_volume(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f32 {
    dot(sub(b, a), cross(sub(c, a), sub(d, a)))
}

fn closest_point_barycentric(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, [f32; 3]) {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(point, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }

    let bp = sub(point, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (add(a, scale(ab, v)), [1.0 - v, v, 0.0]);
    }

    let cp = sub(point, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (add(a, scale(ac, w)), [1.0 - w, 0.0, w]);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let edge = sub(c, b);
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (add(b, scale(edge, w)), [0.0, 1.0 - w, w]);
    }

    let inverse = 1.0 / (va + vb + vc);
    let v = vb * inverse;
    let w = vc * inverse;
    (add(a, add(scale(ab, v), scale(ac, w))), [1.0 - v - w, v, w])
}

fn hash_f32_slice(hasher: &mut Sha256, values: &[f32]) {
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_solver() -> SoftContactSimulator {
        // One deformable tetrahedron above a deformable triangular lip.
        let positions = vec![
            [-0.1, 0.02, -0.1],
            [0.1, 0.02, -0.1],
            [0.0, 0.02, 0.1],
            [0.0, 0.08, 0.0],
            [-0.2, 0.0, -0.2],
            [0.2, 0.0, -0.2],
            [0.0, 0.0, 0.2],
        ];
        SoftContactSimulator::new(
            positions,
            vec![[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0], [4, 6, 5]],
            vec![[0, 1, 2, 3]],
            vec![[0, 4, 6, 5]],
            vec![1.0; 7],
            SoftContactConfig {
                substeps: 4,
                iterations: 12,
                contact_thickness_m: 0.005,
                contact_activation_distance_m: 0.025,
                tether_compliance: 1.0e-3,
                max_displacement_m: 0.1,
                ..SoftContactConfig::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn two_way_contact_keeps_volume_positive() {
        let mut solver = synthetic_solver();
        let mut frame = solver.rest_positions.clone();
        let rest = frame.iter().flatten().copied().collect::<Vec<_>>();
        solver.simulate_chunk(&rest).unwrap();
        for position in &mut frame[..4] {
            position[1] -= 0.04;
        }
        let flat = frame.into_iter().flatten().collect::<Vec<_>>();
        let output = solver.simulate_chunk(&flat).unwrap();
        let report = solver.report();
        assert!(report.contact_projection_count > 0);
        assert!(report.minimum_contact_separation_m > 0.0);
        assert_eq!(report.inverted_tetrahedron_samples, 0);
        assert!(report.minimum_tetrahedron_volume_ratio > 0.5);
        assert!(report.maximum_displacement_m > 0.0);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn chunks_are_deterministic() {
        let mut whole = synthetic_solver();
        let frame = whole
            .rest_positions
            .iter()
            .flat_map(|position| position.iter().copied())
            .collect::<Vec<_>>();
        let two_frames = [frame.clone(), frame.clone()].concat();
        let expected = whole.simulate_chunk(&two_frames).unwrap();
        let mut chunked = synthetic_solver();
        let actual = [
            chunked.simulate_chunk(&frame).unwrap(),
            chunked.simulate_chunk(&frame).unwrap(),
        ]
        .concat();
        assert_eq!(actual, expected);
        assert_eq!(chunked.report().output_sha256, whole.report().output_sha256);
    }

    #[test]
    fn malformed_topology_is_rejected() {
        let error = SoftContactSimulator::new(
            vec![[0.0; 3]; 4],
            vec![[0, 1, 2]],
            vec![[0, 1, 2, 9]],
            vec![[0, 1, 2, 3]],
            vec![1.0; 4],
            SoftContactConfig::default(),
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("outside"));
    }
}
