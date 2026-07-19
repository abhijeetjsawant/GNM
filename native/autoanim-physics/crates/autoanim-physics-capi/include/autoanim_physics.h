#ifndef AUTOANIM_PHYSICS_H
#define AUTOANIM_PHYSICS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AaPhysicsTopology AaPhysicsTopology;
typedef struct AaPhysicsSimulator AaPhysicsSimulator;

typedef struct AaPhysicsConfig {
  float frames_per_second;
  uint32_t substeps;
  uint32_t iterations;
  float velocity_retention;
  float stretch_compliance;
  float tether_compliance;
  float max_displacement_m;
  float jacobi_relaxation;
} AaPhysicsConfig;

AaPhysicsConfig aa_physics_default_config(void);
const char *aa_physics_last_error_message(void);
AaPhysicsTopology *aa_physics_topology_create(size_t vertex_count,
                                               const uint32_t *triangles_xyz,
                                               size_t triangle_count);
void aa_physics_topology_destroy(AaPhysicsTopology *topology);
size_t aa_physics_topology_vertex_count(const AaPhysicsTopology *topology);
size_t aa_physics_topology_edge_count(const AaPhysicsTopology *topology);
AaPhysicsSimulator *aa_physics_simulator_create(
    const AaPhysicsTopology *topology, const float *motion_weights,
    size_t weight_count, AaPhysicsConfig config, size_t threads,
    uint32_t kernel);
void aa_physics_simulator_destroy(AaPhysicsSimulator *simulator);
int32_t aa_physics_simulate_chunk(AaPhysicsSimulator *simulator,
                                  const float *targets, size_t target_count,
                                  const float *accelerations,
                                  size_t acceleration_count, float *output,
                                  size_t output_count);
size_t aa_physics_report_json(const AaPhysicsSimulator *simulator, char *buffer,
                              size_t capacity);

#ifdef __cplusplus
}
#endif

#endif
