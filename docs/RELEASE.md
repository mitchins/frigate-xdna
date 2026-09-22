# Release process (Task 8.5)

One automated path from approved source to GHCR — no manual image
handling, no self-hosted runners. The owner merges to `main`, then
pushes a version tag; the `Release` workflow runs on stock
`ubuntu-latest` runners and does everything else, including creating
the GitHub Release with SBOM + manifest. There is exactly one
publication trigger (a pushed `v*` tag), so a version builds exactly
once. Manual dispatch is dry-run only by construction and can never
publish.

The audited vendor payload is not in the repo and never enters CI
artifacts: it ships as a content-pinned tarball in a PRIVATE Backblaze
B2 bucket (`packaging/vendor-bundle.json` records bucket, bundle name
and sha256 — no secrets). The job downloads it by the exact filename
derived from the in-tree manifest, verifies its sha256 against the
committed receipt, extracts it, and verifies every file with
`verify_payload` plus a `vendor.lock.json` consistency check before
building. Staged input is deleted in an `always()` cleanup step.

## One-time owner setup (all required before first use)

1. **Backblaze B2.** Keep `B2_APP_KEY` (repository secret) and
   `B2_KEY_ID` (repository variable) set. The key is restricted to
   the private vendor bucket; it can only read the pinned bundle.
2. **Environment.** Create the `release` environment with required
   owner approval. Publishing, attestation and release uploads all run
   under it.
3. Nothing else: stock runners provide `docker`, `python3`, `git`
   and `gh`. No `/dev/accel` needed (release building is
   hardware-free), no self-hosted runner to maintain.

## Cutting a release

- **Release candidate:** push tag `v0.1.0-rc.1` (or publish a
  prerelease). Publishes `:0.1.0-rc.1` and `:sha-<short>` only —
  never `:latest`.
- **Stable:** push tag `v0.1.0` (or publish a Release). Publishes
  `:0.1.0`, `:0.1`, `:0`, `:latest` and `:sha-<short>` — all tags
  resolve to one digest (asserted in the job; divergence fails).
- **Dry run** (recommended first): Actions → Release → Run workflow
  (optionally with a `version_tag`). Builds, generates SBOM/manifest
  and runs the image sanity gate against the local image; pushes,
  attests and uploads nothing (each job runs on a fresh runner with
  no shared image store, so cross-job validation only applies to
  pushed digests). Dispatch can never publish, so the requested
  version cannot diverge from the built SHA.
- **Pinned inputs.** The base image is digest-pinned
  (digest-only `ubuntu@sha256:008173c2…`, i.e. docker.io 24.04;
  bump deliberately with a fresh `base-packages.txt` inventory,
  which the workflow records per release). Build-tool versions are pinned in `packaging/Dockerfile`
  (`pip`/`setuptools`/`wheel`); runtime/test pins live in
  `requirements.lock` and `packaging/constraints-*.txt`. No floating
  `--upgrade` remains on the release path.

## What the workflow proves per release

- Exact tag SHA checked out (dirty tree refuses).
- Vendor payload byte-identical to the audited manifest; lock
  consistent; versions never regenerated in the job.
- `linux/amd64` build from `packaging/Dockerfile` + exact source tree
  with OCI labels (source/revision/version/created; licences name the
  split between MIT project code and AMD-EULA/third-party terms).
- CycloneDX SBOM + release manifest (source SHA, image digest, vendor
  manifest digest, recipe ID, base image) regenerated from release
  inputs — never the checked-in 8.4 report.
- OIDC build-provenance + SBOM attestations on the pushed digest.
- Post-push pull-by-digest into a clean environment: image starts,
  `fxdna --help`/`health`, legal notices present, native worker
  present with no forbidden linkage, no `.onnx`/`.rai`/credentials,
  in-image vendor manifest matches the release record.
- SBOM + manifest attached to the GitHub Release (release events);
  workflow artifacts otherwise. Vendor payload, models and keys never
  leave the builder.

## Compatibility (conservative)

Certified: Ryzen AI Max 300 / Strix Halo — the tested platform as
recorded in `docs/RELEASE-8.4.md`. Other XDNA2 platforms: not yet
certified. `latest` is only ever published for stable releases.
