#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CACHE_DIR=${AUTOANIM_CACHE_DIR:-"$PROJECT_DIR/.cache/autoanim_gnm"}
FIXTURE_DIR=${AUTOANIM_TEST_FIXTURES:-"$CACHE_DIR/fixtures"}
mkdir -p "$FIXTURE_DIR"

download_checked() {
  local url=$1 path=$2 sha=$3
  if [[ ! -f "$path" ]]; then curl -L --fail --silent --show-error "$url" -o "$path.partial"; mv "$path.partial" "$path"; fi
  echo "$sha  $path" | shasum -a 256 -c -
}

download_checked \
  "https://librosa.org/data/audio/5703-47212-0000.ogg" \
  "$FIXTURE_DIR/libri-human-speech.ogg" \
  "a284612b46af0535f7e1873758c4387bb8369f6dbbe192ffdec1f171108f98dd"
ffmpeg -y -v error -i "$FIXTURE_DIR/libri-human-speech.ogg" -t 8 -ac 1 -ar 16000 -c:a pcm_s16le "$FIXTURE_DIR/libri-human-speech-8s.wav"
echo "f298d9abc89993008cd4711e1400ee84e5d4bcd01c55672eb514f33b65dc996b  $FIXTURE_DIR/libri-human-speech-8s.wav" | shasum -a 256 -c -

download_checked \
  "https://raw.githubusercontent.com/scikit-image/scikit-image/main/src/_skimage2/data/astronaut.png" \
  "$FIXTURE_DIR/astronaut.png" \
  "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5"
download_checked \
  "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg" \
  "$FIXTURE_DIR/official-portrait.jpg" \
  "744dd848fbb0584229169e01c4944664957c62495fb9e8af514a088ebca43e19"

if [[ "${AUTOANIM_FETCH_RAVDESS:-0}" == "1" ]]; then
  RAVDESS_ZIP="$CACHE_DIR/Audio_Speech_Actors_01-24.zip"
  download_checked \
    "https://zenodo.org/api/records/1188976/files/Audio_Speech_Actors_01-24.zip/content" \
    "$RAVDESS_ZIP" \
    "5d208e01632cc3e5242106fa2af3273e6dc5239fb8143131979ac74c4aa40657"
  unzip -j -o "$RAVDESS_ZIP" "Actor_01/03-01-05-02-01-01-01.wav" -d "$FIXTURE_DIR" >/dev/null
  echo "ba4d1f678784d9656239885413edd9699696f08f9a6e800619eed975ad7a98d6  $FIXTURE_DIR/03-01-05-02-01-01-01.wav" | shasum -a 256 -c -
else
  echo "RAVDESS skipped. Set AUTOANIM_FETCH_RAVDESS=1 for the CC BY-NC-SA emotional fixture."
fi

if [[ "${AUTOANIM_FETCH_CREMA_D:-0}" == "1" ]]; then
  download_checked \
    "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/1658cd342dff90010aa843eaeebd53610a08b1dc/VideoFlash/1001_DFA_ANG_XX.flv" \
    "$FIXTURE_DIR/crema-d-1001-dfa-ang.flv" \
    "10dc3fd1f2bc8203657431598bd7dc9312462008f93d08fda786043ae6a8d2f4"
else
  echo "CREMA-D skipped. Set AUTOANIM_FETCH_CREMA_D=1 after reviewing docs/TEST_FIXTURES.md."
fi

echo "Fixtures ready in $FIXTURE_DIR"
