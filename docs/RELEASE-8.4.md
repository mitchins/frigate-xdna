# Task 8.4 release-boundary record (staged candidate, NOT published)

Intended image name (defined now, never pushed during Task 8.4):

```text
ghcr.io/mitchins/frigate-xdna
```

No GHCR release workflow runs, no image is pushed, and no `:latest` or
version tag is assigned until Task 8.5. A subsequent automation needs
only: this file + `reports/sbom-8.4.json` + the digests below.

## 1. Candidate under test

| Field | Value |
|---|---|
| Local reference | `localhost/frigate-xdna:0.1-dev` |
| Image ID | `5e94a3835e00f92cefc722d3a52c1631b9c1accfcb7662bf4da2c98f420cef53` |
| Content digest (local) | `sha256:04d2c9172ea32d4528d1ecb8d51f01f3ebf563b0a86ec9535683e248eb3408ec` |
| Uncompressed size | 4120097569 B (3.84 GiB) |
| Compressed size | 1235358289 B gzip (`podman save … \| gzip`, measured 2026-09-19) |
| Built | 2026-09-19T05:12:59Z (Task 03 appliance) |
| Registry digest | none — never pushed; the `sha256:04d2…` digest is local content only |

## 2. Reproducing the build (private inputs stay private)

```sh
tools/prepare_vendor_input.py --payload-src <audited site-packages> \
    --xrt-src <audited xrt> --flexmlrt-lib <standalone lib> \
    --legal-src <licence texts> --calib-src <calib dir>
podman build --security-opt apparmor=unconfined \
    -f packaging/Dockerfile -t frigate-xdna:0.1-dev .
```

* Required private path: `packaging/vendor-input/` (gitignored, COPY'd at
  build; verify against `packaging/vendor-files.manifest.json`).
* Vendor manifest digest:
  `5d5e7ac38b3861fac9e8a6f3588afa50c670de0a56338dc39712d93246894275`
  (`sha256sum packaging/vendor-files.manifest.json`; +1 file vs Task 03:
  `flexmlrt/include/FlexMLClient.h` from the audited 1.8.0 wheel, whose
  `libflexmlrt.so` is byte-identical to the staged lib — same version,
  same amd-eula licence, native-worker build requirement per
  `native/CMakeLists.txt`).
* Base: `ubuntu:24.04`. The `apparmor=unconfined` flag is a build-host
  (LXC) workaround only — runtime uses `examples/compose.yaml` without it.

## 3. SBOM / licences / provenance

* `reports/sbom-8.4.json` (CycloneDX 1.5): 17 audited vendor components
  (hashes + licence IDs from `packaging/vendor.lock.json`) + 10 runtime
  + 12 dev/test PyPI pins (from `requirements.lock`; dev/test flagged,
  not shipped). Generated offline by `tools/gen_sbom.py`; no syft/trivy
  on the build host.
  `sha256=7f1981fa6cd9d42495febba81839a1464de76831862e7609af0371ef15d0a13f`
* Licence inventory: `THIRD_PARTY_NOTICES.md` + file-level mapping
  `packaging/legal/component-map.json` (EULA flow-down notices under
  `/opt/fxdna/legal` in-image).
* No standalone vendor tar, private model, or credential is published or
  committed (`packaging/vendor-input/`, `*.sqlite3`, secrets gitignored).

## 4. Runtime contract (verified in 8.3.5, re-asserted here)

* UID:GID `10001:10001`, `NPU_GID` via `group_add`, single device
  `/dev/accel/accel0`; no `/dev/kfd`, no privileged, no Docker socket,
  no GPU compute, no published ZMQ port.
* `read_only: true` + tmpfs, `cap_drop: ALL`, `no-new-privileges:true`,
  `init: true`, `cpus 4.0 / mem 8g / pids 512`, `stop_grace_period 60s`.
* Liveness `fxdna health`; model readiness is separate (never inferred
  from `service_healthy`). Secrets via `PLUS_API_KEY_FILE` mount.
* Known-good Compose: `examples/compose.yaml` (+ `compose.plus.yaml`
  overlay for Frigate+ models). Named volume `frigate-xdna-data` owned
  by image-init; bind mounts are never recursively chowned.
* End-user needs: NPU host, `/data` volume, model refs — no SDK, AMD
  account, activation, or maintainer precompile.

## 5. Acceptance evidence (Task 8.4)

| Gate (`docs/ACCEPTANCE.md`) | Status |
|---|---|
| §1 non-hardware CI | PASS — 202 hardware-free tests + ruff + Sonar gate OK (PR branch CI) |
| §1 Plus stub layer | PASS — `FakePlus` + 23 client tests incl. republish (same-ID new bytes) and A+B coexistence; bearer never reaches download host |
| §1 external-observability privacy | PASS — `fxdna diagnose` bundle + 5 sentinel tests: no raw IDs/tokens/URLs/bytes/DB/key; HMAC aliases stable; `--show-identifiers` opt-in |
| §2 compiler appliance | BANKED (Task 03 proofs) — re-run on the release candidate pending |
| §3 native runtime | BANKED bounded proofs — full gate pending device window |
| §4 full Frigate replay | HARNESS READY, RUN PENDING — `tests/fixtures/replay/` + `tools/replay_acceptance.py`; deterministic clip not yet acquired; Frigate rc2 image not yet pulled |
| §4 real Plus model (fetch→compile→activate→replay) | BLOCKED — operator key present (`/root/.env`), no model ID supplied yet |
| §5 A→B update / offline / failure | PROCEDURE READY (stub paths green), hardware run pending |
| §6 24 h soak | NOT STARTED — needs an authorised quiet-host window after the above |
| §7 release gate | THIS FILE — inputs complete, publication explicitly withheld |

## 6. Verdict

**BLOCKERS before Task 8.5** (i.e. NOT `RELEASE_AUTOMATION_READY`):

1. Real Plus model ID not supplied — fetch/compile/activate proof ungated.
2. Full-app replay not executed — clip + `ghcr.io/blakeblackshear/frigate:0.18.0-rc2` pull + device window outstanding.
3. A→B / offline / failure-matrix runs outstanding (same window as 2).
4. 24 h service soak not started (needs quiet host after 1–3).
5. Compiler-appliance re-run on the exact release candidate outstanding.

When 1–5 close, the release workflow (Task 8.5) may: rebuild with the
pinned inputs above, `podman push ghcr.io/mitchins/frigate-xdna:<version>`,
and attach `reports/sbom-8.4.json` + this file to the release.
