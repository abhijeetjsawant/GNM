use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::PhysicsError;

/// A canonical undirected edge. `a` is always less than `b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Edge {
    pub a: u32,
    pub b: u32,
}

/// One fixed-order CSR entry. `sign` is +1 at edge `a` and -1 at edge `b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Neighbor {
    pub edge_index: u32,
    pub sign: i8,
}

/// Immutable canonical topology shared by solver instances.
#[derive(Clone, Debug)]
pub struct PhysicsTopology {
    vertex_count: usize,
    edges: Vec<Edge>,
    offsets: Vec<u32>,
    neighbors: Vec<Neighbor>,
    digest: String,
}

impl PhysicsTopology {
    pub fn from_triangles(
        vertex_count: usize,
        triangles: &[[u32; 3]],
    ) -> Result<Self, PhysicsError> {
        if triangles.is_empty() {
            return Err(PhysicsError::InvalidTopology(
                "at least one triangle is required".into(),
            ));
        }
        validate_vertex_count(vertex_count)?;

        let mut unique = BTreeSet::new();
        for (triangle_index, triangle) in triangles.iter().enumerate() {
            let [a, b, c] = *triangle;
            for vertex in [a, b, c] {
                if vertex as usize >= vertex_count {
                    return Err(PhysicsError::InvalidTopology(format!(
                        "triangle {triangle_index} references vertex {vertex}, but vertex_count is {vertex_count}"
                    )));
                }
            }
            if a == b || b == c || c == a {
                return Err(PhysicsError::InvalidTopology(format!(
                    "triangle {triangle_index} is degenerate"
                )));
            }
            unique.insert(canonical(a, b));
            unique.insert(canonical(b, c));
            unique.insert(canonical(c, a));
        }
        Self::build(vertex_count, unique.into_iter().collect())
    }

    pub fn from_edges(vertex_count: usize, edges: &[[u32; 2]]) -> Result<Self, PhysicsError> {
        validate_vertex_count(vertex_count)?;
        if edges.is_empty() {
            return Err(PhysicsError::InvalidTopology(
                "at least one edge is required".into(),
            ));
        }
        let mut unique = BTreeSet::new();
        for (edge_index, [a, b]) in edges.iter().copied().enumerate() {
            if a as usize >= vertex_count || b as usize >= vertex_count {
                return Err(PhysicsError::InvalidTopology(format!(
                    "edge {edge_index} references a vertex outside 0..{vertex_count}"
                )));
            }
            if a == b {
                return Err(PhysicsError::InvalidTopology(format!(
                    "edge {edge_index} is degenerate"
                )));
            }
            unique.insert(canonical(a, b));
        }
        Self::build(vertex_count, unique.into_iter().collect())
    }

    fn build(vertex_count: usize, edges: Vec<Edge>) -> Result<Self, PhysicsError> {
        if edges.len() > u32::MAX as usize / 2 {
            return Err(PhysicsError::InvalidTopology("too many edges".into()));
        }

        let mut degree = vec![0_u32; vertex_count];
        for edge in &edges {
            degree[edge.a as usize] = degree[edge.a as usize]
                .checked_add(1)
                .ok_or_else(|| PhysicsError::InvalidTopology("vertex degree overflow".into()))?;
            degree[edge.b as usize] = degree[edge.b as usize]
                .checked_add(1)
                .ok_or_else(|| PhysicsError::InvalidTopology("vertex degree overflow".into()))?;
        }

        let mut offsets = Vec::with_capacity(vertex_count + 1);
        offsets.push(0);
        for value in degree {
            let next = offsets
                .last()
                .copied()
                .unwrap_or(0_u32)
                .checked_add(value)
                .ok_or_else(|| PhysicsError::InvalidTopology("CSR size overflow".into()))?;
            offsets.push(next);
        }
        let mut neighbors = vec![
            Neighbor {
                edge_index: 0,
                sign: 0
            };
            edges.len() * 2
        ];
        let mut cursor = offsets[..vertex_count].to_vec();
        // Edges are sorted, so each vertex's CSR entries are ordered by edge index.
        for (edge_index, edge) in edges.iter().enumerate() {
            let edge_index = u32::try_from(edge_index)
                .map_err(|_| PhysicsError::InvalidTopology("edge index overflow".into()))?;
            for (vertex, sign) in [(edge.a as usize, 1), (edge.b as usize, -1)] {
                let slot = cursor[vertex] as usize;
                neighbors[slot] = Neighbor { edge_index, sign };
                cursor[vertex] += 1;
            }
        }

        let mut hasher = Sha256::new();
        hasher.update((vertex_count as u64).to_le_bytes());
        hasher.update((edges.len() as u64).to_le_bytes());
        for edge in &edges {
            hasher.update(edge.a.to_le_bytes());
            hasher.update(edge.b.to_le_bytes());
        }
        let digest = format!("{:x}", hasher.finalize());

        Ok(Self {
            vertex_count,
            edges,
            offsets,
            neighbors,
            digest,
        })
    }

    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    pub fn neighbors(&self) -> &[Neighbor] {
        &self.neighbors
    }

    pub fn neighbors_of(&self, vertex: usize) -> &[Neighbor] {
        let start = self.offsets[vertex] as usize;
        let end = self.offsets[vertex + 1] as usize;
        &self.neighbors[start..end]
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }
}

fn validate_vertex_count(vertex_count: usize) -> Result<(), PhysicsError> {
    if vertex_count == 0 || vertex_count > u32::MAX as usize {
        return Err(PhysicsError::InvalidTopology(
            "vertex_count must be in 1..=u32::MAX".into(),
        ));
    }
    Ok(())
}

fn canonical(a: u32, b: u32) -> Edge {
    if a < b {
        Edge { a, b }
    } else {
        Edge { a: b, b: a }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_edges_and_fixed_csr() {
        let topology = PhysicsTopology::from_triangles(4, &[[0, 1, 2], [2, 1, 3]]).unwrap();
        assert_eq!(
            topology.edges(),
            &[
                Edge { a: 0, b: 1 },
                Edge { a: 0, b: 2 },
                Edge { a: 1, b: 2 },
                Edge { a: 1, b: 3 },
                Edge { a: 2, b: 3 },
            ]
        );
        assert_eq!(topology.offsets(), &[0, 2, 5, 8, 10]);
        assert_eq!(
            topology.neighbors_of(1),
            &[
                Neighbor {
                    edge_index: 0,
                    sign: -1
                },
                Neighbor {
                    edge_index: 2,
                    sign: 1
                },
                Neighbor {
                    edge_index: 3,
                    sign: 1
                },
            ]
        );
    }

    #[test]
    fn reversed_and_duplicate_edges_are_unique() {
        let topology = PhysicsTopology::from_edges(3, &[[1, 0], [0, 1], [1, 2]]).unwrap();
        assert_eq!(
            topology.edges(),
            &[Edge { a: 0, b: 1 }, Edge { a: 1, b: 2 }]
        );
    }

    #[test]
    fn gnm_style_quad_grid_has_expected_unique_edges() {
        let rows = 37;
        let cols = 41;
        let mut triangles = Vec::new();
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
        let topology = PhysicsTopology::from_triangles(rows * cols, &triangles).unwrap();
        let expected = rows * (cols - 1) + (rows - 1) * cols + (rows - 1) * (cols - 1);
        assert_eq!(topology.edges().len(), expected);
        assert_eq!(topology.neighbors().len(), expected * 2);
    }

    #[test]
    fn invalid_topology_is_rejected() {
        assert!(PhysicsTopology::from_triangles(3, &[]).is_err());
        assert!(PhysicsTopology::from_triangles(3, &[[0, 1, 1]]).is_err());
        assert!(PhysicsTopology::from_triangles(3, &[[0, 1, 3]]).is_err());
        assert!(PhysicsTopology::from_edges(2, &[[0, 0]]).is_err());
    }
}
