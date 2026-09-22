# Development expectations

## Supported modes

1. CPU-only unit/contract development: no AMD binaries, NPU or Plus credentials. Use fake compiler/native/HTTP services.
2. Native Linux development: same supervisor/CLI and explicit runtime/compiler prefixes as Docker; no dependence on historical `/root/xdna/env.sh`.
3. Appliance build/hardware acceptance: exact audited vendor payload supplied as a private build input, full hash/licence verification, explicit exclusive device ownership.

macOS can run suitable pure-Python/unit tests, but is not an XDNA execution platform for this project. Do not imply that a Docker image substitutes for the host's NPU driver.

## Developer commands

Implemented in the Makefile (Tasks 01–04 landed):

```
make test-unit                # hardware-free unit tests
make test-contract            # stock Frigate 0.18.0 contract tests (pinned)
make test-integration         # fake-Plus / fake-backend integration
make test                     # all of the above, hardware-free
make coverage                 # all suites with coverage + fail_under floor
make build-native             # cmake build of fxdna-worker (needs XRT_ROOT)
make test-hardware            # explicit opt-in/ownership guard (refuses by default)

Unavailable placeholders (exit 3 until Task 05 implements them):
make image, make test-image-offline, make test-frigate-e2e
```

Environment: `pip install -r requirements.lock` (pinned runtime + dev
pins: contract deps, pyzmq, coverage, ruff). `ruff check` covers
`src recipes tools packaging tests schemas`; CI runs the same gates.
Pin manager and compiler dependencies separately. CI must not download
mutable `latest` dependencies or run tests from Frigate `dev` and
describe them as 0.18.0 compatibility.

## Vendor/build inputs

Maintain a file-level `vendor.lock.json` including complete digest, source package/version, path, size, governing notice and purpose. The audited compiler tar can be a private CI input; do not publish it standalone. Only the incorporated appliance follows the accepted distribution route.

Use BuildKit secrets/private build inputs for maintainer acquisition credentials. Do not pass secrets through build arguments or COPY them into an intermediate layer. Inspect final and intermediate/cache publication settings before release.

The image may include source code plus non-open vendor binaries under their separate applicable terms. Do not put the project source under an AMD-wide EULA simply because the image includes those binaries.

## Benchmark discipline

Keep saturation capacity and paced service latency separate. Report model bytes/hash, resolution, original precision, compile recipe, native runtime, CPU provider, thread limits, throughput, latency distributions, memory, queue delay and deadline misses.

Do not compare ORT CPU fallback to a historical OpenVINO number as if they were the same backend. Do not convert marketing TOPS into supported-camera claims. Use the fixed tensor/label contract and actual model metadata throughout.
