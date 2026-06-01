# GNM Head Model

**GNM** is a state-of-the-art parametric 3D statistical model of the human
head, learned from a large dataset of 3D scans. It provides fine-grained control
over facial identity, expressions, and head pose. This repository contains the
core NumPy and Jax based GNM model implementation, and tools for visualization
and semantic sampling of parameters.

![GNM Teaser Image (Placeholder - Replace with an actual image)]

## Features

*   **Detailed 3D Face Geometry:** Generates a dense 3D face mesh comprised of the skin, the eyes, the teeth, and the tongue.
*   **Disentangled Control:** Offers separate parameters for:
    *   **Identity:** Controls subject-specific facial features.
    *   **Expression:** Animates the face with a rich set of expression blendshapes.
    *   **Head Pose:** Controls the rotation of the neck and eyeballs.
    *   **Translation:** Controls the global position.
*   **Semantic Parameter Sampling:** Includes pre-trained models to generate identity and expression parameters from semantic labels:
    *   `ExpressionSampler`: Generate expressions like "happy", "surprise", or blend them.
    *   `IdentitySampler`: Generate identities based on attributes like gender and ethnicity.
*   **NumPy Implementation:** Easy to integrate into existing Python pipelines.
*   **Permissive License:** Apache 2.0.

## Project Structure

```text
.
├── CONTRIBUTING.md                 # Guidelines for contributing to the project
├── LICENSE                         # Project license
├── README.md                       # Main documentation
└── gnm/
    └── shape/                      # Main package containing model logic
        ├── data/                   # GNM model assets and versions
        │   ├── textures/           # Model textures (.jpg, .png)
        │   ├── semantic_sampler/   # Pre-trained .h5 semantic sampling models
        │   └── versions/
        │       └── v3_0/           # Contains v3 GNM model files (.npz)
        ├── fitting_utils/          # Shared optimization helper functions
        ├── demos/                  # Interactive demo notebooks (.ipynb)
        ├── visualization/          # Rendering and camera projection utilities
        ├── gnm_base.py             # Base GNM class definitions
        ├── gnm_colab_viewer.py     # Colab 3D face model visualization tool
        ├── gnm_data_loader.py      # Dynamic model loaders and checkers
        ├── gnm_data_schema.py      # Input/output data validation schemas
        ├── gnm_jax.py              # JAX implementation of GNM
        ├── gnm_numpy.py            # NumPy implementation of GNM (primary)
        ├── requirements.txt        # Package dependencies
        └── semantic_sampler.py     # Semantic parameter sampling (identities/expressions)
```

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/google/gnm.git
    cd gnm/shape
    ```

2.  Create virtual environment and install dependencies:

    ```bash
    conda create -n myenv python=3.12
    conda activate myenv
    pip install -r requirements.txt
    pip install -e .
    ```

## Getting Started

### Loading the GNM Model

The core model can be loaded as follows. The necessary model data (`gnm.npz`)
is included in this repository.

```python
from gnm.shape import gnm_numpy
from gnm.shape import semantic_sampler
import numpy as np
import trimesh # For visualization

# Load the GNM head model.
gnm = gnm_numpy.GNM.from_local(
    version=gnm_numpy.GNMMajorVersion.V3,
    variant=gnm_numpy.GNMVariant.HEAD,
)

# Get the template (average) face mesh.
template_vertices = gnm.template_vertex_positions
faces = gnm.triangles

# Save or visualize the mesh (example using trimesh).
mesh = trimesh.Trimesh(vertices=template_vertices, faces=faces, process=False)
# mesh.show()
mesh.export("template_face.obj")
```

### Basic Parameter Manipulation
You can generate a mesh by providing parameters for identity, expression,
joint rotations, and translation.

```python
import trimesh

# Zero parameters result in the template face.
identity = np.zeros(gnm.identity_dim)
expression = np.zeros(gnm.expression_dim)
rotations = np.zeros((gnm.num_joints, 3)) # Axis-angle
translation = np.zeros((3,))

vertices = gnm(identity, expression, rotations, translation)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
mesh.show()
```

### Demo

To experiment with generating a human head mesh from custom identity,
expression, joint rotations and global translation, please see
`gnm/shape/demos/gnm_head_demo.ipynb`.

![sampling](assets/readme/gnm_head_demo.gif)

## Using the Semantic Sampler
Generate meaningful identity and expression parameters using the
`ExpressionSampler` and `IdentitySampler`.

### Expression Sampling
```python
expr_sampler = semantic_sampler.ExpressionSampler()

# Available expression labels.
print(expr_sampler.expression_label_mapping)

# Sample a 'happy' expression.
happy_expression = expr_sampler.sample_expression(
    semantic_sampler.Expression.HAPPY, num_samples=1
)[0]

vertices_happy = gnm(expression=happy_expression)
mesh_happy = trimesh.Trimesh(vertices=vertices_happy, faces=faces)
mesh_happy.show()
mesh_happy.export("happy_face.obj")
```

### Identity Sampling
```python
id_sampler = semantic_sampler.IdentitySampler()

# Explain available classes.
print(id_sampler.explain_classes())

# Sample a specific identity.
identity_sample = id_sampler.sample_identity(
    semantic_sampler.Gender.FEMALE,
    semantic_sampler.Ethnicity.ASIAN,
    num_samples=1
)[0]

vertices_identity = gnm(identity=identity_sample)
mesh_identity = trimesh.Trimesh(vertices=vertices_identity, faces=faces)
mesh_identity.show()
mesh_identity.export("sampled_identity_face.obj")
```

### Demo
To experiment with identity and expression sampling and blending, please see
`gnm/shape/demos/semantic_gnm_demo.ipynb`.

![sampling](assets/readme/semantic_gnm_demo.gif)

## Generating a Complete Face
Combine identity and expression parameters from the samplers.

```python
# Get a random identity and expression.
random_identity = id_sampler.randomize_identities(num_samples=1)[0]
random_expression = expr_sampler.randomize_expressions(num_samples=1)[0]

# Optionally add some head rotation (e.g., slight neck bend).
rotations = np.zeros((gnm.num_joints, 3))
# Example: gnm.joint_names to find the index for 'head'.
head_index = gnm.joint_names.index('head')
# Random rotations along each axis.
rotations[head_index] = [-0.2, 0.3, 0.2]

vertices_random = gnm(identity=random_identity, expression=random_expression, rotations=rotations)
mesh_random = trimesh.Trimesh(vertices=vertices_random, faces=faces)
# mesh_random.show()
mesh_random.export("random_face.obj")
```

## Model Parameters

The GNM model is controlled by two primary sets of coefficients that determine
the identity and expression of the generated face. The following dimensions are
relevant for the GNM v3.x.

### Identity Parameters

*   **Shape:** `[batch_size, 253]`
*   **Description:** Controls the unique physical characteristics of the individual. These are divided into:
    *   **170** Head components
    *   **3** Eyeball components
    *   **80** Teeth components
*   **Total:** 253 identity components.
*   **Typical Range:** -3 to +3

### Expression Parameters

*   **Shape:** `[batch_size, 383]`
*   **Description:** Controls the facial movement and blendshape weights. These are divided into:
    *   **100** Left eye components
    *   **100** Right eye components
    *   **150** Lower face components
    *   **32** Tongue components
    *   **1** Iris component
*   **Total:** 383 expression components.
*   **Typical Range:** -3 to +3.

### Joint Parameters

*   **Shape:** rotations: `[batch_size, 4x3 Rotation matrix]`, global translation: `[batch_size, 3]`
*   **Description:** Controls the global head position and joint angles for head pose and eyeball orientation.

## Model Data
The GNM model data (e.g., `gnm_head.npz`) contains the template shape, identity
basis, expression basis, skinning weights, and UV layout. This file is provided
within the `gnm/shape/data/versions/v{MAJOR}_{MINOR}` directory.

The Semantic Sampler models
(`expression_decoder_model.h5`, `identity_decoder_model.h5`) are located as
in `gnm/shape/data/semantic_sampler`.

## Model Limitations Regarding Gender Representation
This 3D Morphable Model (3DMM) is trained on datasets that categorize human head
data into binary gender categories: male and female. We acknowledge that this
binary classification does not encompass the full spectrum of human gender
identities and is not representative of all individuals. The use of these
categories stems from the limitations of available large-scale 3D scan datasets
and follows common practice in the historical development of 3DMMs. Users should
be aware of this limitation and consider the potential implications for fairness
and representation in their specific applications. We encourage the community to
explore and develop more inclusive datasets and models as the field progresses.

## Citation
If you use GNM in your research, please cite:

```
(Note: Placeholder - a BibTeX entry should be provided for the relevant publication.)
```

## Contributing
We'd love to accept your patches and contributions to this project! See
[CONTRIBUTING.md](CONTRIBUTING.md) for more information on how to get started
and how we handle external contributions.

## License
This project is licensed under the Apache License, Version 2.0. See the
[LICENSE](LICENSE) file for details.