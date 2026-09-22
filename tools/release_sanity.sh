#!/usr/bin/env bash
# Hardware-free release-image sanity gate (Task 8.5).
# Usage: release_sanity.sh <image-ref> <vendor-manifest-sha256>
# Runs the SAME checks pre-push (local tag) and post-push (digest),
# so a bad image can never reach :latest. No NPU, no secrets, no network
# except the local container runtime. Exit nonzero on any mismatch.
set -euo pipefail

IMG="${1:?usage: release_sanity.sh <image-ref> <manifest-sha256>}"
WANT_MANIFEST="${2:?usage: release_sanity.sh <image-ref> <manifest-sha256>}"
RUN="podman run --rm --security-opt apparmor=unconfined"

echo "== $IMG: CLI =="
$RUN --entrypoint fxdna "$IMG" --help > /dev/null
$RUN --user 10001:10001 --read-only \
  --entrypoint fxdna "$IMG" health

echo "== $IMG: legal notices =="
$RUN --entrypoint sh "$IMG" -c \
  'test -f /opt/fxdna/legal/component-map.json && test -f /opt/fxdna/legal/THIRD_PARTY_NOTICES.md'

echo "== $IMG: native worker linkage =="
$RUN --entrypoint sh "$IMG" -c '
  set -eu
  test -x /opt/fxdna/native/fxdna-worker
  if ldd /opt/fxdna/native/fxdna-worker | grep -iE "onnxruntime|/voe/|vaiml|xcompiler|pyflexmlrt"; then
    echo "forbidden linkage in native worker"
    exit 1
  fi'

echo "== $IMG: no model bytes, no credentials =="
# Upstream test fixtures ship inside site-packages (tiny public graphs,
# not weights): onnxruntime/datasets and onnx/backend/test. Anything
# else in .onnx/.rai form fails the gate.
FOUND=$($RUN --entrypoint sh "$IMG" -c '
  find /opt /usr/local /root \( -name "*.onnx" -o -name "*.rai" \) 2>/dev/null | grep -v "onnxruntime/datasets/" | grep -v "onnx/backend/test/" || true')
# shellcheck disable=SC2086
[[ -z "$FOUND" ]] || { echo "$FOUND"; echo "unexpected model bytes"; exit 1; }
CREDS=$($RUN --entrypoint sh "$IMG" -c '
  grep -rIlE "PLUS_API_KEY=.{4,}|ghp_|github_pat_|xox[bap]-" /opt/fxdna/manager 2>/dev/null || true')
[[ -z "$CREDS" ]] || { echo "$CREDS"; echo "credential material"; exit 1; }

echo "== $IMG: vendor manifest matches release record =="
GOT=$($RUN --entrypoint sha256sum "$IMG" \
  /opt/fxdna/recipes/vendor-files.manifest.json | cut -d" " -f1)
[[ "$GOT" == "$WANT_MANIFEST" ]] || {
  echo "in-image $GOT != record $WANT_MANIFEST"; exit 1; }
echo "manifest match: $GOT"
echo "SANITY PASS: $IMG"
