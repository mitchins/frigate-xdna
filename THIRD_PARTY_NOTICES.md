# Third-party notices

This file records the governing terms of incorporated third-party material.
It is not legal advice. The project source licence (`LICENSE`, MIT) covers
original project code only.

## Frigate (contract-test sources)

Files under `tests/upstream/` that are byte-identical copies of Frigate
sources at tag `v0.18.0-rc2` are MIT-licensed, copyright their respective
authors. See `tests/upstream/NOTICES.md` for the full text pointer and the
pinned blob hashes in `tests/upstream/upstream.lock.json`.

Upstream: https://github.com/blakeblackshear/frigate (tag `v0.18.0-rc2`,
licence file blob `924cb4148cda90ee9d7f7953dcbd9fea31a05874`).

## AMD Ryzen AI (appliance-incorporated binaries)

The release appliance incorporates unmodified AMD vendor binaries from the
Ryzen AI 1.8 Linux toolchain (XRT userspace, VitisAI EP, VAIML/xcompiler
backend, peano runtime) as audited in the user's Phase-7.7 compiler-appliance
audit. Those files are governed by:

* AMD End-User License Agreement (`amd-end-user-license-agreement.pdf`), and
* Ryzen AI Third-Party Notices (`ryzen-ai-1.7.1-linux-tpn-license.pdf`,
  `ryzen-ai-1-8-0-linux-tpn-license.pdf`).

The audit's engineering reading: distribution is via the AMD EULA §2.3
"Derivative Works to customers" path with at-least-as-restrictive flow-down
terms; standalone redistribution of the vendor payload and re-licensing under
a Free Software licence are prohibited (§3.2/§3.6). The release packaging
task must reproduce the per-file licence mapping
(`packaging/legal/component-map.json`) and ship the applicable notices with
the image. Nothing in this repository re-licences those binaries, and no
vendor tarball, licence PDF, or private model is committed here.

## Test-only shims

`tests/upstream/frigate_shims/` contains minimal test-harness modules for
Frigate-internal imports that are NOT part of the pinned compatibility
surface (`const`, `plus.PlusApi`, `util.builtin`, `detection_api`). They are
original test code under the project licence, clearly marked, and must never
be mistaken for upstream Frigate behaviour. Only `zmq_ipc.py` and
`detector_config.py` in `tests/upstream/` are byte-identical pinned sources.
