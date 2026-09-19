# Schema implementation requirements

Task 8.0 must implement and test JSON Schema plus matching typed models for:

1. `model-descriptor.schema.json`: schema version; source kind/ref/digest; input tensor name/shape/dtype/layout/pixel format/normalization; output tensors; explicit decoder profile and numerical score/box semantics; class count; label map/attributes; target/runtime compatibility; provenance. RAI imports require an artifact digest. Unknown semantic fields/ambiguous combinations fail.
2. `artifact.schema.json`: compile-key inputs; source and artifact hashes/sizes; recipe/toolchain payload ID; target/RAI ABI profile; compile stats; serving-contract refs; immutable relative paths; licence/provenance manifest reference. No secrets/signed URLs/user-supplied absolute output paths.
3. `status.schema.json`: service/model/job states, active worker generation, safety inhibition, bounded diagnostic/error codes, request counters/timing and schema version. Distinguish PREPARED, VERIFIED and ACTIVE; errors are not READY.

Do not fabricate exact ABI/profile IDs, full hashes or tensor contracts from examples. Import them from the audited successful artifacts and pinned Frigate metadata, then validate representative JSON fixtures in CI.
