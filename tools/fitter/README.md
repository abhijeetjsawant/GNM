# Fitter integration probes

Groundwork for `docs/FITTER_PLAN.md` — the build that closes the lane's largest
structural gap: **we triangulate points and drive a fixed rig; nothing ever fits a
body to the performer.**

These run against Meta's **momentum** (MIT) and the **MHR** release assets
(Apache-2.0, verified down to `assets/LICENSE.txt`). Neither is installed into the
project environment — create a throwaway one:

```bash
uv venv /tmp/momenv --python 3.12
uv pip install --python /tmp/momenv/bin/python pymomentum-cpu
curl -sL -o assets.zip \
  https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip
```

**Consume MHR from its own release, never via the SOMA third-party redistribution** —
that copy also carries a 28-dimension scale PCA which is under the **SAM licence, not
Apache**. Do not lift it; fit the 68 raw scale channels with our own regulariser.

| script | what it establishes |
|---|---|
| `probe_mhr_rest.py` | momentum loads MHR + `compact_v6_1.model`: 127 joints, 204 parameters (68 scaling, 136 pose), units in **centimetres**, and its rest-pose segment lengths |
| `compare_rest_skeletons.py` | MHR's **untuned mean** body against our canonical rig, both scored on the same landmark definition against the two measured performers |

Result of the second, and it is the argument for the route: **mean absolute error
23 mm for MHR's mean body against 49 mm for our canonical rig — 2.2x closer before a
single parameter is fitted**, and on shoulder width 8 mm against 185.
