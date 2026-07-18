# Audio2Face3D native runner

Small macOS 15+ command-line wrapper around speech-swift's
`Audio2Face3D` product. `Package.swift` pins speech-swift **exactly to
v0.0.23**. The runner loads a WAV as mono 16 kHz audio and writes one Codable
`Audio2Face3DFrame` per JSONL line. Coefficient layouts are model-specific
(James/Claire: 169 values; Mark: 301 values) and require a downstream
retargeting step before they can drive GNM.

```bash
cd native/a2f-runner
swift build -c release --product a2f-runner

# MLX requires its Metal shader library beside the executable.
env BUILD_DIR="$PWD/.build" \
  .build/checkouts/speech-swift/scripts/build_mlx_metallib.sh release

.build/release/a2f-runner \
  --input ../../.cache/autoanim_gnm/fixtures/libri-human-speech-8s.wav \
  --output /tmp/libri-a2f.jsonl \
  --model aufklarer/Audio2Face-3D-v2.3.1-Claire-MLX \
  --emotion joy --emotion-strength 0.65 \
  --verbose
```

The metallib build requires Xcode's Metal toolchain. If `xcrun metal` reports
that it is missing, install it with `xcodebuild -downloadComponent
MetalToolchain`, then repeat the metallib command. The runner will fail during
model loading when no compatible `mlx.metallib` is colocated with the binary.

To use a previously downloaded/exported bundle without network access:

```bash
.build/release/a2f-runner \
  --input speech.wav \
  --output motion.jsonl \
  --model-dir /path/to/audio2face3d-mlx-james
```

`--model` and `--model-dir` are mutually exclusive. Add `--offline` with
`--model` to require an already cached Hugging Face bundle. Argument and path
validation failures exit with status 2; model/audio/inference failures exit
with status 1. Run `swift test --filter RunnerOptionsTests` for the lightweight
parser and validation tests.

The optional acting direction is passed into Audio2Face's native ten-channel
explicit emotion input; it does not replace or retime the acoustic mouth
motion. Supported names are `neutral`, `surprise`, `anger`, `contempt`,
`disgust`, `fear`, `grief`, `joy`, `outofbreath`, `pain`, and `sad`.

Audio2Face3D inference loads a large MLX model. Do not overlap the build,
tests, or inference command with another Swift/MLX model-loading process.
