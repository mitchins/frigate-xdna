# Test fixtures (no private models, credentials, or weights)

## `coco-research-labelmap.txt`
Verbatim copy of the historical research labelmap
(`/mnt/downloads/scratch/frigate-src/labelmap.txt`) used in pre-product
experiments. It contains 91 lines and known discrepancies versus standard
COCO-80 (notably `car` at both id 2 and id 7, where standard COCO has
`truck` at 7; `hat` at 25). It is preserved as evidence of the discrepancy
the product must NOT silently adopt — serving contracts carry their own
explicit label maps. Do not treat this file as ground truth.

## `yolov9s-320.descriptor.json`
Serving descriptor fixture for the banked v9s-320 artifact family: raw YOLO
profile, 84x2100 `cxcywh`-pixel output, a 9-entry standard-ID label-map
subset (0 person, 2 car, 5 bus, 7 truck — standard COCO-80 numerics, not the
research file above). Any product use with a different label map requires a
new serving digest. No runtime ABI IDs are claimed (assigned at activation
validation, Task 04).

## `plus-private-example.descriptor.json`
Shape of a private Plus-style descriptor: sparse non-COCO label IDs and an
attributes map. Values are synthetic examples, not a real Plus model.

## `raw-yolo-320.npz`
Golden synthetic raw tensor: shape (1, 84, 2100), crafted boxes/scores with
known expected NMS output. Generated deterministically by
`tests/fixtures/make_raw_tensors.py` (seed pinned); the generator script is
the fixture source of truth.

## `raw-yolo-640.npz`
Same generator at 8400 anchors for the 640 geometry.

## Circularity note
`expected` arrays are produced by the fixture generator's own NMS, which
mirrors the product decoder — they guard against regressions, not against a
shared misunderstanding of the raw contract. The hand-computed single-box
test (`test_hand_computed_single_box`) is the independent anchor; a
hardware-anchored vector from banked XDNA outputs is Task 04 work.
