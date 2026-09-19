#include "postprocess.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace fxdna {

int postprocess_yolo_raw(const float* raw, size_t cols, const YoloConfig& cfg, float out20x6[20][6]) {
    struct Det { float score; int cls; float x1,y1,x2,y2; };
    std::vector<Det> dets;
    dets.reserve(cols);

    const int C = cfg.class_count;
    const int rows = 4 + C;
    // raw layout is [rows, cols] row-major: raw[r*cols + c]
    for (size_t c=0;c<cols;++c) {
        float best = 0; int best_cls = -1;
        for (int k=0;k<C;++k) {
            float s = raw[(4+k)*cols + c];
            if (s > best) { best = s; best_cls = k; }
        }
        if (best < cfg.score_threshold || best_cls<0) continue;
        float cx = raw[0*cols + c];
        float cy = raw[1*cols + c];
        float w  = raw[2*cols + c];
        float h  = raw[3*cols + c];
        // Input contract: coordinate_units == "pixels" (SPEC §7.1), layout cxcywh
        float x1 = cx - w*0.5f;
        float y1 = cy - h*0.5f;
        float x2 = cx + w*0.5f;
        float y2 = cy + h*0.5f;
        dets.push_back({best, best_cls, x1, y1, x2, y2});
    }

    std::sort(dets.begin(), dets.end(), [](const Det& a, const Det& b){
        if (a.score != b.score) return a.score > b.score;
        return a.cls < b.cls;
    });

    // Class-aware NMS: suppress only same-class overlap (preserve different classes)
    std::vector<Det> kept;
    std::vector<char> sup(dets.size(), 0);
    std::vector<float> areas(dets.size());
    for (size_t i=0;i<dets.size();++i) areas[i]=(dets[i].x2-dets[i].x1)*(dets[i].y2-dets[i].y1);

    for (size_t i=0;i<dets.size() && (int)kept.size()<cfg.max_det; ++i) {
        if (sup[i]) continue;
        kept.push_back(dets[i]);
        for (size_t j=i+1;j<dets.size();++j) {
            if (sup[j]) continue;
            if (cfg.class_aware_nms && dets[j].cls != dets[i].cls) continue;
            float xx1 = std::max(dets[i].x1, dets[j].x1);
            float yy1 = std::max(dets[i].y1, dets[j].y1);
            float xx2 = std::min(dets[i].x2, dets[j].x2);
            float yy2 = std::min(dets[i].y2, dets[j].y2);
            float inter = std::max(0.f, xx2-xx1)*std::max(0.f, yy2-yy1);
            float uni = areas[i]+areas[j]-inter+1e-9f;
            float iou = inter/uni;
            if (iou > cfg.nms_iou) sup[j]=1;
        }
    }

    // Ensure descending score, preserve class IDs, no double NMS (caller must not re-apply)
    std::sort(kept.begin(), kept.end(), [](const Det& a, const Det& b){ return a.score > b.score; });

    std::memset(out20x6, 0, sizeof(float)*20*6);
    for (size_t i=0;i<kept.size() && i<20; ++i) {
        float y1 = kept[i].y1 / static_cast<float>(cfg.image_h);
        float x1 = kept[i].x1 / static_cast<float>(cfg.image_w);
        float y2 = kept[i].y2 / static_cast<float>(cfg.image_h);
        float x2 = kept[i].x2 / static_cast<float>(cfg.image_w);
        // Clamp to [0,1] after division, preserve finite fallback
        y1 = std::isfinite(y1) ? std::clamp(y1, 0.0f, 1.0f) : 0.0f;
        x1 = std::isfinite(x1) ? std::clamp(x1, 0.0f, 1.0f) : 0.0f;
        y2 = std::isfinite(y2) ? std::clamp(y2, 0.0f, 1.0f) : 0.0f;
        x2 = std::isfinite(x2) ? std::clamp(x2, 0.0f, 1.0f) : 0.0f;
        out20x6[i][0] = static_cast<float>(kept[i].cls); // preserve class IDs
        out20x6[i][1] = kept[i].score;
        out20x6[i][2] = y1;
        out20x6[i][3] = x1;
        out20x6[i][4] = y2;
        out20x6[i][5] = x2;
    }
    for (int i=0;i<20;++i) for(int k=2;k<6;++k) if(!std::isfinite(out20x6[i][k])) out20x6[i][k]=0.f;

    return (int)kept.size();
}

} // namespace fxdna
