# Task 8.1 — Plus acquisition, multi-model registry/cache and CLI

Depends on Task 8.0. Follow SPEC and INTERFACES; no NPU work.

## Objective

A user can register multiple Plus/local models, obtain exact source bytes/metadata, inspect preparation state and retain them safely. Compilation is still a fake backend in this task.

## Work

1. Implement `fxdna serve`, private administrative Unix socket, configuration/env parsing, secret-file handling and `prepare/status/wait/cache` commands.
2. Implement narrowly read-only Plus API behaviour from pinned rc2 source: API key -> token, get model info, signed download URL, model bytes. API host/constants must be verified from the pinned source. Do not guess endpoints or copy unrelated upload/training calls.
3. Use test servers first. Test expiry, refresh, failure, rate limits, malicious redirect, oversize download, mismatched metadata, and secret redaction. Never send bearer credentials to the signed-download destination.
4. Implement the content-addressed source/artifact store, serving-contract metadata, SQLite migrations, job deduplication, per-key lock, atomic publication and recovery as in CACHE.
5. Implement pins and dry-run prune. Adding B must not select it, replace A, or delete A. Removing a ref from the configured list must not silently destroy its cache.
6. Support local ONNX imports and explicit RAI descriptors. Reject pickle/custom executable formats, external-data traversal and unsupported ambiguous output contracts.
7. Implement fake compile-job transitions including WAITING_FOR_DEVICE, terminal errors and long preparation. No actual compiler invocation yet.
8. Add offline startup/cached behaviour and content-change handling under an existing model ID. Never assume basename is a content digest.

## Optional authorised live check

Only when the user has explicitly provided a test key/model ID: fetch that selected model and metadata. Do not print the secret, signed URL or private bytes; do not train/upload/activate/change production. Mock tests are sufficient if credentials were not supplied.

## Acceptance

* Two models coexist in cache with correct independent metadata/pins.
* Same bytes under different aliases deduplicate compilation; same alias with different bytes does not.
* Interrupted writes never become ready artifacts; low space fails cleanly.
* Online commands use the daemon as single writer; standalone mode refuses its lock.
* No NPU/runtime/Quark import anywhere in the manager.
* A fake long compile proves `wait` and status progress without fabricated percentages.

## Stop

Commit and report control-plane functionality. Do not install AMD components, execute a model or alter the stock protocol.
