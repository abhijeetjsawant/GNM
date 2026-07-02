# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GNM data loader."""

# from collections.abc import Mapping, Sequence
from collections.abc import Sequence
import functools
from typing import Any

from absl import logging
from etils import epath
from gnm.shape import gnm_data_schema
from gnm.shape.data.versions import gnm_catalog
from gnm.shape.data.versions import gnm_specs
import numpy as np

_pkg = __package__ or 'gnm.shape'
_MODELS_VERSIONS_DIR = epath.resource_path(f'{_pkg}.data.versions')
_VARIANT_TO_MODEL_FILE_NAME_MAP = gnm_catalog.VARIANT_TO_MODEL_FILE_NAME_MAP


class GNMModelDataNotLinkedError(Exception):
  """Raised when a GNM model data is not linked into the binary."""

  pass


def _get_model_path_from_version_and_variant(
    version: gnm_specs.GNMMajorVersion,
    variant: gnm_specs.GNMVariant,
) -> epath.Path:
  """Returns the GNM model runfiles path for given variant and version."""
  version_value = major_to_newest_full_version(version).value.replace('.', '_')
  version_dir_name = f'v{version_value}'
  model_file_name = f'{_VARIANT_TO_MODEL_FILE_NAME_MAP[variant]}.npz'
  return _MODELS_VERSIONS_DIR / version_dir_name / model_file_name


def major_to_newest_full_version(
    major: gnm_specs.GNMMajorVersion,
) -> gnm_specs.GNMVersion:
  """Returns the newest GNMVersion for a given GNMMajorVersion."""
  minors = [
      e for e in gnm_specs.GNMVersion if e.value.split('.')[0] == major.value
  ]
  return sorted(minors, key=lambda e: int(e.value.split('.')[1]))[-1]


def full_version_to_major(
    version: gnm_specs.GNMVersion,
) -> gnm_specs.GNMMajorVersion:
  """Returns the major version of a GNMVersion."""
  return gnm_specs.GNMMajorVersion(version.value.split('.')[0])


@functools.lru_cache
def load_model_from_runfile(
    version: gnm_specs.GNMMajorVersion, variant: gnm_specs.GNMVariant
) -> dict[str, Any]:
  """Loads GNM model data from a runfile for the given version/variant."""
  model_file = _get_model_path_from_version_and_variant(version, variant)

  logging.info(
      'Loading GNM model version %s, variant %s from runfiles: %s',
      version,
      variant,
      model_file,
  )
  with model_file.open('rb') as f:
    data_dict = dict(np.load(f))

  # Validate the data.
  valid, missing, extra = _validate_gnm_data(data_dict)
  if not valid:
    raise ValueError(
        f'Validation failed for version {version}, variant {variant}.'
        f' Missing: {missing}, Extra: {extra}'
    )

  return _standardize_gnm_data_types(data_dict)


def _validate_gnm_data(
    data: dict[str, Any],
) -> tuple[bool, Sequence[str], Sequence[str]]:
  """Validates the GNM data dict.

  It returns any extra or missing fields and a boolean indicating if the data
  dict has exactly the expected fields.

  Args:
    data: The GNM data dict to validate.

  Returns:
    A tuple of (bool, Sequence[str], Sequence[str]) indicating if the data dict
    has exactly the expected fields, the missing fields and the extra fields.
  """
  expected_fields = gnm_data_schema.GNM_DATA_ATTRIBUTES
  missing_fields = list(set(expected_fields) - set(data.keys()))
  extra_fields = list(set(data.keys()) - set(expected_fields))
  return not missing_fields and not extra_fields, missing_fields, extra_fields


def _standardize_gnm_data_types(data: dict[str, Any]) -> dict[str, Any]:
  """Standardizes the GNM data data types in-place.

  The data loaded from the .npz model files are defined as Numpy arrays. This
  function converts the items to their expected Python types.

  Args:
    data: The GNM data dict to standardize.

  Returns:
    The GNM data dict with standardized data types.
  """
  keys_to_standardize = (
      'version',
      'variant',
      'identity_names',
      'joint_names',
      'expression_names',
      'mesh_component_names',
      'vertex_group_names',
  )
  for k in keys_to_standardize:
    if k not in data:
      raise ValueError(f'Required attribute {k} not found in GNM data.')

  try:
    data['version'] = gnm_specs.GNMVersion(str(data['version']))
  except ValueError as e:
    version_val = data['version']
    raise ValueError(f'Unknown GNM version: {version_val}') from e
  try:
    data['variant'] = gnm_specs.GNMVariant(str(data['variant']))
  except ValueError as e:
    variant_val = data['variant']
    raise ValueError(f'Unknown GNM variant: {variant_val}') from e
  for key in (
      'identity_names',
      'joint_names',
      'expression_names',
      'mesh_component_names',
      'vertex_group_names',
  ):
    data[key] = [str(v) for v in data[key]]

  return data
