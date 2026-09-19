#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace fxdna {

constexpr int PROTOCOL_VERSION = 1;
constexpr size_t MAX_HEADER_BYTES = 16 * 1024;      // 16 KiB
constexpr size_t MAX_TENSOR_BYTES = 16 * 1024 * 1024; // 16 MiB
constexpr size_t MAX_MODEL_BYTES = 256 * 1024 * 1024; // 256 MiB
constexpr size_t RESULT_BYTES = 20 * 6 * 4;           // 480 bytes float32 LE

enum class MessageType { LOAD, INFER, RESULT, STATUS, SHUTDOWN, UNKNOWN };

MessageType parse_message_type(const std::string& s);
std::string to_string(MessageType t);

struct TensorSpec {
    std::vector<int> shape; // e.g. [1,3,320,320]
    std::string dtype;      // "float32"
};

struct Header {
    int protocol_version = PROTOCOL_VERSION;
    MessageType message_type = MessageType::UNKNOWN;
    int64_t request_id = 0;
    int64_t worker_generation = 0;
    std::string artifact_path;   // LOAD only, supervisor-controlled
    std::string serving_digest;  // LOAD/INFER
    TensorSpec tensor_spec;      // INFER/RESULT
    size_t payload_length = 0;   // bounded length of following binary payload
    std::string error_code;      // RESULT/STATUS failures
    std::string error_message;   // bounded message
};

// Framing: [4-byte BE header_len][JSON header bytes][binary payload bytes]
// No pointers cross boundary, no public socket, private socketpair only.

bool send_message(int fd, const Header& hdr, const uint8_t* payload, size_t payload_len, std::string& err);
bool recv_message(int fd, Header& hdr, std::vector<uint8_t>& payload, std::string& err);

// JSON serialization/parsing with bounded checks
std::string header_to_json(const Header& h);
bool json_to_header(const std::string& json, Header& out, std::string& err);

// Validation helpers (native independently validates)
bool validate_header(const Header& h, std::string& err);
bool validate_tensor_spec(const TensorSpec& spec, size_t payload_len, std::string& err);
bool validate_finite(const float* data, size_t count);

} // namespace fxdna
