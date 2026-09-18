# Target repository structure

This spec kit contains documentation/examples/tasks, not an implemented service. Task 01 creates the code structure below.

Implementation status (Tasks 01–02 landed): manager, CLI, cache, daemon,
contract/integration/unit tests and fixtures exist. Compiler, native
worker, Docker image and hardware gates remain future work (see
docs/WORKQUEUE.md).

```
frigate-xdna/
  README.md
  AGENTS.md
  SPEC.md
  LICENSE                         # original project code; owner supplies copyright
  THIRD_PARTY_NOTICES.md
  pyproject.toml
  requirements.lock              # manager only
  CMakeLists.txt
  src/frigate_xdna/
    __main__.py
    cli.py
    config.py
    supervisor.py
    admin.py                     # local Unix socket; no public control API
    errors.py
    plus/client.py
    models/refs.py
    models/inspect.py
    models/contracts.py
    cache/keys.py
    cache/registry.py
    cache/store.py
    cache/gc.py
    compiler/launcher.py
    compiler/recipe.py
    compiler/jobs.py
    runtime/supervisor.py
    runtime/ipc.py
    runtime/device_lease.py
    runtime/safety.py
    transport/frigate_zmq.py
    transport/sessions.py
    observability/logging.py
    observability/metrics.py
  native/
    CMakeLists.txt
    src/main.cpp
    src/model.cpp                # proven FlexML API and mmap ownership
    src/tensors.cpp
    src/protocol.cpp             # bounded private IPC framing
    src/postprocess_yolo.cpp
    include/...
  recipes/bf16-vaiml-v1/
    recipe.json
    prepare.py
    compile.py
    validate.py
    requirements.lock           # compiler environment only
  packaging/
    Dockerfile
    vendor.lock.json             # complete imported digests, no placeholders in a release
    vendor-files.manifest.json
    verify_payload.py
    compiler-launcher/
    legal/component-map.json
    legal/README.md
  schemas/
    model-descriptor.schema.json
    artifact.schema.json
    status.schema.json
  examples/
    compose.yaml
    compose.host-port.yaml
    frigate-plus.yaml
    frigate-local.yaml
    .env.example
  tests/
    unit/
    contract/                    # exact rc2 plugin; loopback tests
    integration/                 # mocks and offline container tests
    hardware/                    # opt-in, serialized, safe named fixtures
    acceptance/                  # full Frigate synthetic-camera tests
    fixtures/                    # no private model/credentials
    upstream/upstream.lock.json
  tools/
    import_proofs.py
    prepare_test_stream.py
    collect_report.py
  docs/
    INTERFACES.md
    CACHE.md
    ACCEPTANCE.md
    SOURCES.md
    OPERATIONS.md
    DEVELOPMENT.md
  agent-tasks/
    01-foundation.md
    02-model-manager.md
    03-compiler-appliance.md
    04-native-zmq.md
    05-docker-acceptance.md
```

## Dependency policy

Manager: Python standard library plus a small pinned set such as pyzmq, HTTP client, data validation and ONNX inspection dependencies. Prefer stdlib argparse/logging/sqlite3 over adding frameworks for a small CLI. Do not import the whole Frigate application into the deployed sidecar.

Inference: C++17 and the proven runtime libraries. Keep Python vendor bindings out of this process. Use the existing successful runner and model lifetime semantics before refactoring for style.

Compiler: exact audited prefix/payload and an isolated recipe-specific Python environment. Audit reconstruction is a packaging exercise; do not replace it with unpinned pip dependencies or a fresh toolchain upgrade.

Fixtures: pin and include applicable MIT notices for copied Frigate contract code. Fake Plus authentication/download servers and a fake native worker must exercise most logic without hardware. Private Plus model tests use runtime secrets and never upload their artifacts.

## Repository working conventions

Each agent task is a small reviewable series of commits. Every task report identifies its commit, tests run, tests not run, new risks and next task. No host/provisioning mutations without explicit approval. Do not push releases, create public binaries or change production while implementing this spec.

Hardware operations are never implicit side effects of test discovery, `doctor`, `status`, importing a module or starting the CLI parser.
