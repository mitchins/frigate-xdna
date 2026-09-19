# Task 8.3.5 — Repository hardening record

Bounded housekeeping/security pass over the Task 8.3 product. No features,
no architecture changes, no AMD-stack upgrades, no hardware experiments.

## 1. Analysis scope

| Contents | Sonar treatment | Rationale |
|---|---|---|
| `src/**`, `native/src/**` | Full analysis + coverage | First-party product code |
| `packaging/*.py`, `schemas/` | Full analysis + coverage | First-party build/contract code |
| `tools/**`, `recipes/**` | Analysis, no import coverage | Operator-run offline scripts, verified by execution (`--help` smoke + appliance build), not by import |
| `native/**` (C++) | Analysis, no line coverage | Covered by contract/hardware tests (§4), never faked into Python-style numbers |
| `native/reference/**` | Excluded | Byte-identical Phase-5/7 proof sources, retained for provenance (`native/reference/README.md`); not compiled (`CMakeLists.txt` builds only `native/src/*`), not shipped; must not be modified |
| `tests/upstream/**` | Excluded | Pinned stock Frigate `v0.18.0-rc2` sources (`LICENSE`, `NOTICES.md`, `upstream.lock.json`); third-party, must not be reformatted |
| `tests/**` | Test scope | S8997/S5778 style rules assume pytest; suite is unittest — documented residual, not churned |
| `packaging/vendor-input/**`, `packaging/vendor-files.manifest.json`, `**/__pycache__/**` | Excluded | Staged binary payload input / generated manifest / bytecode |
| `tests/fixtures/**` | Duplication-excluded | Binary/text evidence fixtures |

Exclusions live in `sonar-project.properties` (commented) and
`pyproject.toml` (`[tool.ruff.lint.per-file-ignores]` for upstream).

## 2. Security triage (21 findings, rating E → re-run pending)

| # | Location | Rule | Verdict | Action |
|---|---|---|---|---|
| 1 | `packaging/verify_payload.py:25` | S2083 BLOCKER path traversal | **Confirmed** (manifest-driven join could escape root) | **Fixed**: `safe_join()` rejects absolute/`..`-escaping entries with `TRAVERSAL` error; unit-tested |
| 2 | `packaging/verify_payload.py:46` | S8707 CLI path | Accepted risk | Operator's own dirs; tool purpose is checking them; manifest side now constrained |
| 3-5 | `packaging/verify_payload.py:70,74` | S6549 filesystem oracle | By design (false positive) | Presence/size reporting IS the tool's job |
| 6-7 | `recipes/*/compile.py:53,77` | S8707 CLI path | Accepted risk | Offline operator tools reading operator-named paths; no network, no privilege boundary |
| 8 | `recipes/*/prepare.py:88` | S8707 CLI path | Accepted risk | Same as above |
| 9 | `recipes/*/validate.py:40` | S8707 CLI path | Accepted risk | Same as above |
| 10-13 | `tools/build_vendor_manifest.py` | S8707 CLI path ×4 | Accepted risk | Maintainer-side manifest builder; local paths only |
| 14-15 | `tools/prepare_vendor_input.py:50,55` | S8707 CLI path | Accepted risk | Maintainer-side staging tool; local paths only |
| 16 | `tools/prepare_vendor_input.py:65` | S8705 command injection | **False positive** | List-form `subprocess.run`, no shell; ruff PLW1510 likewise not applicable (exit code is checked) |
| 17 | `native/reference/worker-phase7.cc:161` | S2612 perms | Scoped, not waived | Non-shipped proof code (`mkdir workdir, 0755`); triaged, retained byte-identical by policy |
| 18 | `pyproject.toml` | S8565 lock file | Accepted risk | pip-based project; transitive closure pinned in `requirements.lock` + `packaging/constraints-*.txt`, not uv/poetry |
| 19 | `src/frigate_xdna/supervisor.py:300` | S8707 descriptor path | Accepted risk | Local admin CLI reads an operator-named `--descriptor` file; `OSError`/`ValueError` handled; no privilege boundary crossed |
| 20-21 | `tools/import_proofs.py:86,87` | S8707 CLI path | Accepted risk | One-shot local evidence-import tool |

Zero unresolved confirmed High/Critical defects. Nothing waived to
improve the score: every accepted item names the missing privilege
boundary or the tool's explicit purpose.

## 3. Reliability + maintainability fixes

Fixed (were failing the quality gate on new code):

- `runtime/ipc.py` S1764: intentional `v != v` NaN check rewritten as
  `math.isfinite` (clearer, rule-clean).
- `transport/frigate_zmq.py` S7497 ×4: serve/worker loops no longer
  convert `CancelledError` to `break`; `stop()` joins via
  `asyncio.wait(timeout=5.0)` instead of swallowing foreign
  cancellations. Bounded, structured shutdown.
- `transport/frigate_zmq.py` S7493 (+ ruff ASYNC230): artifact
  validation moved off the event loop (`asyncio.to_thread`); lease
  still serializes activation.
- Expired queue entries now reply (`TIMEOUT` / zero frame) before
  `late_discard`, so stock REQ clients cannot hang
  (`docs/INTERFACES.md` deadline rules extended by one bullet).

Also fixed: dead `parsed` dict (`supervisor.py`), unused `binding`
(`frigate_zmq.py`), unused `art` (test), `E741 l` → `line`, UP041
timeout alias, ~40 E501 wraps, 50 autofixed import/unused findings
(ruff `E,F,I,UP` clean), Dockerfile package sort (S7018), unclosed
manifest file in `verify_payload.py`.

Residuals (reviewed, intentionally left):

- `native/reference/**` smells: provenance, not modified by policy.
- `tests/**` S8997/S5778: pytest-specific style rules on a unittest
  suite; churn not justified.
- `verify_payload.py` S3776 (complexity 73): linear manifest checker,
  now unit-tested; splitting it would obscure the check sequence.
- `build_vendor_manifest.py` S3776 (16 vs 15): one point over, churn
  not justified in a pinned manifest builder.
- C++ smells in `native/src` (S6494/S5945 class): product code kept
  warning-clean under `-Wall -Wextra -Werror`; Sonar C++ style
  residuals are readability-level, no lifetime/bound findings open.

## 4. Coverage

`make coverage` (unittest suites ×3 with append) → `coverage.xml` +
terminal report; Sonar ingests via
`sonar.python.coverage.reportPaths`. Floor `fail_under = 70`
(`pyproject.toml`), CI-enforced.

| Area | Before | After | Note |
|---|---|---|---|
| Total | 59% | **76%** | |
| transport/sessions | 0% | 100% | New: A/B isolation, quiescence |
| runtime/ipc | 0% | 94% | New: framing round-trips, bounds |
| runtime/device_lease | 0% | 85% | New: contention, self-fd guard |
| runtime/safety | 0% | 82% | New: journal, inhibition, rotation |
| transport/frigate_zmq | 0% | 60% | New: wire + dispatch + activation gating |
| packaging/verify_payload | — | 67% | New: traversal/symlink/hash guards |
| plus/client | 83% | 84% | New: SSRF URL gates, retry delays |

Critical Python areas now covered: model acquisition gates
(`test_plus_gates`), cache/registry recovery (`test_manager`,
existing), compiler job state machine (`test_launcher`, existing),
device lease, ZMQ identity/generation binding, activation/quarantine
gating (`_try_activate` fake refusal), secret filtering
(`test_plus`, existing — bearer/redaction paths), error paths.

Known-uncovered by policy: `launcher._run_locked` (needs audited
payload subprocesses), Plus download pinning (needs network),
`cli.py` daemon flows (43% — serve-loop coverage is Task 05 e2e
territory), native C++ (contract/hardware tests, §1).

## 5. CI gates + lint

`.github/workflows/build.yml` (jobs bounded `timeout-minutes: 10`):

- `test`: pinned install → `make coverage` (floor) → ruff →
  `compileall` → packaging sanity (manifest JSON, `--help` smokes,
  CMake XRT guard refuses without vendored root) → proprietary-file
  guard (no `.rai/.onnx/.pdf/.tgz` committed) → uploads `coverage.xml`.
- `sonarqube`: full checkout + coverage artifact → official scan
  action (pinned SHAs), `SONAR_TOKEN` secret. CI scanning owns
  analysis; SonarCloud Automatic Analysis disabled via API
  (`POST api/autoscan/activation?enable=false`) after the first run
  proved the conflict (exit 3).

Lint: ruff `E,F,I,UP`, `target py312`, upstream per-file ignore.
Native: `-Wall -Wextra -Werror` unchanged; no new warnings.

## 6. Dependency / SBOM inputs (no upgrades)

Audited AMD/XDNA stack untouched. Inventory for Task 8.4 SBOM:

- `requirements.lock` — now machine-readable; runtime pins +
  dev-only pins (`coverage`, `ruff`, contract deps), all `==`.
- `packaging/constraints-manager.txt`, `constraints-quant.txt`.
- `packaging/vendor.lock.json` (17 components) +
  `vendor-files.manifest.json` + `packaging/legal/component-map.json`.
- `packaging/Dockerfile` base `ubuntu:24.04` (+ sorted apt set;
  OS packages float within the LTS — noted, not pinned).
- `THIRD_PARTY_NOTICES.md` + `tests/upstream/upstream.lock.json`.

## 7. Repo structure / docs reconciliation

- `Makefile`: wired real `build-native` (explicit `XRT_ROOT`;
  fails with guidance otherwise); added `coverage`; `PY` overridable.
- `docs/WORKQUEUE.md`: Task 04 marked landed (PR #3); numbering fixed.
- `docs/DEVELOPMENT.md`: rewritten from pre-implementation plan to
  landed commands + env setup.
- `docs/OPERATIONS.md`: intro de-staled (implemented CLI/image).
- `docs/INTERFACES.md`: late-discard reply rule documented.
- `packaging/Dockerfile`: apt sort; apparmor workaround scoped as
  **build-host-only**, runtime must use `examples/compose.yaml`.
- `examples/`: env names verified against `config.py`; Plus secret
  stays a mounted file (`compose.plus.yaml`).

## 8. Docker / security review

`examples/compose.yaml` already meets the checklist: `init: true`
(child reaping for compile workers), `user 10001:10001`, single
device `/dev/accel/accel0` via `NPU_GID` (no `/dev/kfd`), no
privileged, no socket, `cap_drop: ALL` + `no-new-privileges`,
`read_only: true` + tmpfs, no published ZMQ port (private net;
`compose.host-port.yaml` binds loopback-only with a warning),
`PLUS_API_KEY_FILE` secret mount, CPU/mem/pids bounds, healthcheck
(`fxdna health` exists, no NPU probe). Image runs as `USER
10001:10001` with data under `/data`.

Shutdown semantics reviewed: `KeyboardInterrupt` → graceful
(frontend + supervisor stop). `SIGTERM` (docker stop) terminates
without the handler — fail-safe by design: an idle stop leaves no
journal; a mid-operation stop leaves one and the next start inhibits
until explicit `fxdna recover`. No silent continuation.

## 9. Sonar re-run record

| Metric | Before (main `df1a44a`) | After (PR #4) |
|---|---|---|
| Security rating / vulns | E / 21 | _to fill_ |
| Reliability rating / bugs | C / 6 | _to fill_ |
| Maintainability / smells | A / 215 | _to fill_ |
| Coverage | — (none ingested) | _to fill_ |
| Duplication | 0.0% | _to fill_ |
| Quality gate | ERROR (new reliability) | _to fill_ |

## 10. Remaining intentional debt / next task

- Coverage floor 70 (baseline 76); raise after Task 05 e2e covers
  serve flows.
- `launcher._run_locked` + Plus download pinning covered only by
  execution-gated paths, not unit coverage (documented, not faked).
- Test-style Sonar residuals (S8997/S5778) and reference-provenance
  scope stand until a pytest migration is justified (not this task).
- Next bounded task: **Task 8.4 acceptance** (separate PR).
