#pragma once
#include <cstddef>
#include <cstdint>

namespace fxdna {

// Base output profile is YOLOv9s-320: raw (84,2100) -> [20,6] [class_id, score, ymin, xmin, ymax, xmax]
// normalized, descending score, zero-filled remainder, preserve class IDs, no double NMS.

struct YoloConfig {
    int image_w = 320;
    int image_h = 320;
    int class_count = 80;
    float score_threshold = 0.25f;
    float nms_iou = 0.45f;
    int max_det = 20;
    bool class_aware_nms = true; // preserve class IDs, different classes retained
};

// raw: pointer to float32 buffer of size (4+class_count)*N in row-major (84,N) layout
// out20x6: [20][6] float32 output, little-endian normalized coords, zero-filled
int postprocess_yolo_raw(const float* raw, size_t cols, const YoloConfig& cfg, float out20x6[20][6]);

} // namespace fxdna
