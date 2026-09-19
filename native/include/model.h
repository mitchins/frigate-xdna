#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "FlexMLClient.h"

namespace fxdna {

// Owns the .rai mmap and FlexML Model with strict lifetime ordering:
// mmap (uint8_t* fbs_buffer) must outlive Model; munmap only after Model destruction.
class Model {
public:
    Model() = default;
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Load one .rai artifact. Previous model (if any) is destroyed before new mmap.
    // Returns true on success, false on failure (with error string).
    bool load(const std::string& rai_path, std::string& error);

    void unload();

    bool loaded() const { return model_ != nullptr; }

    // Accessors for IO tensors (via getIOTensors)
    std::vector<flexmlrt::client::ErtTensorType>& inputs() { return inputs_; }
    std::vector<flexmlrt::client::ErtTensorType>& outputs() { return outputs_; }

    // Execute forward; caller must have bound inputs_[0].data and outputs_[0].data
    bool forward(std::string& error);

    // Metadata derived from model
    size_t input_elements() const;
    std::vector<uint32_t> input_shape() const;
    size_t output_elements() const;
    size_t output_cols() const; // N for (84,N) raw layout

private:
    static bool map_rai(const char* path, uint8_t** buf, size_t* size);

    uint8_t* fbs_buffer_ = nullptr;          // must be uint8_t* per FlexML API
    size_t fbs_buffer_size_ = 0;
    flexmlrt::client::Model* model_ = nullptr; // owned, delete before munmap
    std::vector<flexmlrt::client::ErtTensorType> inputs_;
    std::vector<flexmlrt::client::ErtTensorType> outputs_;
    std::string model_path_;
};

} // namespace fxdna
