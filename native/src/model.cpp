#include "model.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>

namespace fxdna {

Model::~Model() {
    unload();
}

void Model::unload() {
    // Destruction ordering: delete Model before munmap (mmap must outlive all native refs).
    if (model_) {
        delete model_;
        model_ = nullptr;
    }
    if (fbs_buffer_) {
        munmap(fbs_buffer_, fbs_buffer_size_);
        fbs_buffer_ = nullptr;
        fbs_buffer_size_ = 0;
    }
    inputs_.clear();
    outputs_.clear();
    model_path_.clear();
}

bool Model::map_rai(const char* path, uint8_t** buf, size_t* size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return false; }
    void* p = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return false;
    *buf = static_cast<uint8_t*>(p);
    *size = static_cast<size_t>(st.st_size);
    return true;
}

bool Model::load(const std::string& rai_path, std::string& error) {
    // Tear down previous model strictly before new mmap.
    if (model_) { delete model_; model_ = nullptr; }
    if (fbs_buffer_) { munmap(fbs_buffer_, fbs_buffer_size_); fbs_buffer_ = nullptr; fbs_buffer_size_ = 0; }

    uint8_t* map = nullptr;
    size_t map_size = 0;
    if (!map_rai(rai_path.c_str(), &map, &map_size)) {
        error = "mmap failed for " + rai_path;
        return false;
    }

    // FlexML Model construction with exact Options (preserve types).
    flexmlrt::client::Options options;
    options.modelPath = rai_path;
    options.executeMode = 2;
    options.deviceName = "stx";
    options.subgraphName = "vaiml_par_0";
    options.extOptions["fbs_buffer"] = map;                       // uint8_t* required
    options.extOptions["fbs_buffer_size"] = map_size;
    options.extOptions["cache_dir"] = std::string(".");
    options.extOptions["enable_preemption"] = true;

    auto* mdl = new flexmlrt::client::Model(options);
    if (!mdl->good()) {
        delete mdl;
        munmap(map, map_size);
        error = "Model creation failed for " + rai_path;
        return false;
    }

    // Input/output tensor handling via getIOTensors
    auto in = mdl->getIOTensors("input", false);
    auto out = mdl->getIOTensors("output", false);
    if (in.empty() || out.empty()) {
        delete mdl;
        munmap(map, map_size);
        error = "no io tensors for " + rai_path;
        return false;
    }

    // Commit ownership only after success: mmap outlives Model
    fbs_buffer_ = map;
    fbs_buffer_size_ = map_size;
    model_ = mdl;
    inputs_ = std::move(in);
    outputs_ = std::move(out);
    model_path_ = rai_path;
    return true;
}

bool Model::forward(std::string& error) {
    if (!model_) { error = "no model loaded"; return false; }
    try {
        model_->forward(inputs_, outputs_);
    } catch (const std::exception& e) {
        error = std::string("forward failed: ") + e.what();
        return false;
    } catch (...) {
        error = "forward failed: unknown exception";
        return false;
    }
    return true;
}

size_t Model::input_elements() const {
    if (inputs_.empty()) return 0;
    size_t n = 1;
    for (auto d : inputs_[0].getMetadata().shape) n *= d;
    return n;
}

size_t Model::output_elements() const {
    if (outputs_.empty()) return 0;
    size_t n = 1;
    for (auto d : outputs_[0].getMetadata().shape) n *= d;
    return n;
}

size_t Model::output_cols() const {
    if (outputs_.empty()) return 0;
    size_t bytes = outputs_[0].getMetadata().size;
    if (bytes < 84 * sizeof(float)) return 0;
    size_t elems = bytes / sizeof(float);
    if (elems % 84 != 0) return 0;
    return elems / 84; // 2100 @320, 8400 @640
}

} // namespace fxdna
