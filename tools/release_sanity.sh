#!/usr/bin/env bash
# Hardware-free release-image sanity gate (Task 8.5, identity: v0.1.1 C2).
# Usage: release_sanity.sh <image-ref> <vendor-manifest-sha256> <version> <revision>
# Runs the SAME checks pre-push (local tag) and post-push (digest),
# so a bad image can never reach :latest. No NPU, no secrets, no network
# except the local container runtime. Exit nonzero on any mismatch.
set -euo pipefail

IMG="${1:?usage: release_sanity.sh <image-ref> <manifest-sha256> <version> <revision>}"
WANT_MANIFEST="${2:?usage: release_sanity.sh <image-ref> <manifest-sha256> <version> <revision>}"
WANT_VERSION="${3:?usage: release_sanity.sh <image-ref> <manifest-sha256> <version> <revision>}"
WANT_REVISION="${4:?usage: release_sanity.sh <image-ref> <manifest-sha256> <version> <revision>}"
# Container runtime override (podman locally, docker on stock runners).
CR="${CONTAINER_RUNTIME:-podman}"
if [[ "$CR" == "docker" ]]; then
  RUN="docker run --rm"
else
  RUN="podman run --rm --security-opt apparmor=unconfined"
fi

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

echo "== $IMG: reported build identity =="
GOT_IDENTITY=$($RUN --user 10001:10001 --read-only \
  --entrypoint fxdna "$IMG" --version)
echo "$GOT_IDENTITY"
[[ "$GOT_IDENTITY" == "fxdna $WANT_VERSION revision=$WANT_REVISION channel="* ]] || {
  echo "identity mismatch: want version=$WANT_VERSION revision=$WANT_REVISION";
  exit 1; }
if [[ "$GOT_IDENTITY" == *" channel=development" ]]; then
  echo "release image reports a development build"
  exit 1
fi
echo "identity match: $WANT_VERSION @ $WANT_REVISION"

echo "== $IMG: no test fixtures or fake backends =="
# Scripted-vendor fixtures (fake workers, phase dummies) carry the
# FXDNA-TEST-FIXTURE marker; the appliance must not contain it, nor
# any test tree. (compiler/fake.py ships but is unreachable in images:
# backend selection pins the audited backend and the artifact manifest
# backend check invalidates anything else.)
FOUND_FIX=$($RUN --entrypoint sh "$IMG" -c '
  grep -rIl "FXDNA-TEST-FIXTURE" /opt/fxdna 2>/dev/null || true')
# shellcheck disable=SC2086
[[ -z "$FOUND_FIX" ]] || { echo "$FOUND_FIX"; echo "test fixture in image"; exit 1; }
HAS_TESTS=$($RUN --entrypoint sh "$IMG" -c '
  ls -d /opt/fxdna/manager/tests /opt/fxdna/manager/fixtures 2>/dev/null || true')
[[ -z "$HAS_TESTS" ]] || { echo "$HAS_TESTS"; echo "test tree in image"; exit 1; }

echo "== $IMG: vendor manifest matches release record =="
GOT=$($RUN --entrypoint sha256sum "$IMG" \
  /opt/fxdna/recipes/vendor-files.manifest.json | cut -d" " -f1)
[[ "$GOT" == "$WANT_MANIFEST" ]] || {
  echo "in-image $GOT != record $WANT_MANIFEST"; exit 1; }
echo "manifest match: $GOT"
echo "SANITY PASS: $IMG"
