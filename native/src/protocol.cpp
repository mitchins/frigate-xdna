#include "protocol.h"
#include <arpa/inet.h>
#include <unistd.h>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <sstream>

namespace fxdna {

MessageType parse_message_type(const std::string& s) {
    if (s == "LOAD" || s == "load") return MessageType::LOAD;
    if (s == "INFER" || s == "infer") return MessageType::INFER;
    if (s == "RESULT" || s == "result") return MessageType::RESULT;
    if (s == "STATUS" || s == "status") return MessageType::STATUS;
    if (s == "SHUTDOWN" || s == "shutdown") return MessageType::SHUTDOWN;
    return MessageType::UNKNOWN;
}
std::string to_string(MessageType t) {
    switch (t) {
        case MessageType::LOAD: return "LOAD";
        case MessageType::INFER: return "INFER";
        case MessageType::RESULT: return "RESULT";
        case MessageType::STATUS: return "STATUS";
        case MessageType::SHUTDOWN: return "SHUTDOWN";
        default: return "UNKNOWN";
    }
}

// Minimal JSON helpers without external deps; bounded and strict.

static std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size()+2);
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o.push_back('\\'); o.push_back((char)c); }
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\b') o += "\\b";
        else if (c == '\f') o += "\\f";
        else if (c == '\t') o += "\\t";
        else if (c < 0x20) {
            char buf[7];
            std::snprintf(buf, sizeof(buf), "\\u%04X", c);
            o += buf;
        } else o.push_back((char)c);
    }
    return o;
}

std::string header_to_json(const Header& h) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"protocol_version\":" << h.protocol_version << ",";
    oss << "\"message_type\":\"" << to_string(h.message_type) << "\",";
    oss << "\"request_id\":" << h.request_id << ",";
    oss << "\"worker_generation\":" << h.worker_generation;
    if (!h.artifact_path.empty()) oss << ",\"artifact_path\":\"" << json_escape(h.artifact_path) << "\"";
    if (!h.serving_digest.empty()) oss << ",\"serving_digest\":\"" << json_escape(h.serving_digest) << "\"";
    if (!h.tensor_spec.shape.empty() || !h.tensor_spec.dtype.empty()) {
        oss << ",\"tensor_spec\":{";
        oss << "\"shape\":[";
        for (size_t i=0;i<h.tensor_spec.shape.size();++i) {
            if (i) oss << ",";
            oss << h.tensor_spec.shape[i];
        }
        oss << "],";
        oss << "\"dtype\":\"" << json_escape(h.tensor_spec.dtype) << "\"";
        oss << "}";
    }
    oss << ",\"payload_length\":" << h.payload_length;
    oss << ",\"class_count\":" << h.class_count;
    if (!h.error_code.empty()) oss << ",\"error_code\":\"" << json_escape(h.error_code) << "\"";
    if (!h.error_message.empty()) oss << ",\"error_message\":\"" << json_escape(h.error_message) << "\"";
    oss << "}";
    return oss.str();
}

// Very small JSON field extractor (flat + one nested tensor_spec) – bounded, no eval.
static std::string extract_string(const std::string& js, const std::string& key) {
    std::string q = "\"" + key + "\"";
    auto i = js.find(q);
    if (i == std::string::npos) return "";
    i = js.find(':', i);
    if (i == std::string::npos) return "";
    ++i; while (i<js.size() && isspace((unsigned char)js[i])) ++i;
    if (i>=js.size() || js[i]!='"') return "";
    ++i; std::string out;
    while (i<js.size() && js[i]!='"') {
        if (js[i]=='\\' && i+1<js.size()) { out.push_back(js[i+1]); i+=2; } else { out.push_back(js[i]); ++i; }
    }
    return out;
}
static bool extract_int64(const std::string& js, const std::string& key, int64_t& out) {
    std::string q = "\"" + key + "\"";
    auto i = js.find(q);
    if (i==std::string::npos) return false;
    i = js.find(':', i); if (i==std::string::npos) return false;
    ++i; while (i<js.size() && isspace((unsigned char)js[i])) ++i;
    auto j = js.find_first_of(",}", i);
    if (j==std::string::npos) return false;
    std::string tok = js.substr(i, j-i);
    // trim
    size_t a=0; while(a<tok.size()&&isspace((unsigned char)tok[a])) ++a;
    size_t b=tok.size(); while(b>a&&isspace((unsigned char)tok[b-1])) --b;
    tok = tok.substr(a,b-a);
    try { out = std::stoll(tok); return true; } catch(...) { return false; }
}
static bool extract_uint64(const std::string& js, const std::string& key, uint64_t& out) {
    int64_t tmp=0; if(!extract_int64(js,key,tmp)) return false; if(tmp<0) return false; out=(uint64_t)tmp; return true;
}

bool json_to_header(const std::string& js, Header& h, std::string& err) {
    if (js.size() > MAX_HEADER_BYTES) { err="header exceeds 16 KiB"; return false; }
    int64_t pv=0;
    if (!extract_int64(js,"protocol_version",pv)) { err="missing protocol_version"; return false; }
    h.protocol_version = (int)pv;
    std::string mt = extract_string(js,"message_type");
    if (mt.empty()) mt = extract_string(js,"type");
    h.message_type = parse_message_type(mt);
    if (h.message_type==MessageType::UNKNOWN) { err="unknown message_type"; return false; }
    if (!extract_int64(js,"request_id",h.request_id)) h.request_id=0;
    if (!extract_int64(js,"worker_generation",h.worker_generation)) h.worker_generation=0;
    h.artifact_path = extract_string(js,"artifact_path");
    h.serving_digest = extract_string(js,"serving_digest");
    h.error_code = extract_string(js,"error_code");
    h.error_message = extract_string(js,"error_message");
    uint64_t pl=0;
    if (extract_uint64(js,"payload_length",pl)) h.payload_length=(size_t)pl;
    else h.payload_length=0;
    if (int64_t cc=0; extract_int64(js,"class_count",cc)) h.class_count=(int)cc;
    else h.class_count=0;

    // tensor_spec nested
    h.tensor_spec.shape.clear();
    h.tensor_spec.dtype.clear();
    auto ti = js.find("\"tensor_spec\"");
    if (ti != std::string::npos) {
        auto si = js.find("\"shape\"", ti);
        if (si != std::string::npos) {
            auto lb = js.find('[', si); auto rb = js.find(']', lb);
            if (lb!=std::string::npos && rb!=std::string::npos) {
                std::string inside = js.substr(lb+1, rb-lb-1);
                std::istringstream iss(inside);
                std::string tok;
                while (std::getline(iss, tok, ',')) {
                    // trim
                    size_t a=0; while(a<tok.size()&&isspace((unsigned char)tok[a])) ++a;
                    size_t b=tok.size(); while(b>a&&isspace((unsigned char)tok[b-1])) --b;
                    if (a>=b) continue;
                    try { h.tensor_spec.shape.push_back(std::stoi(tok.substr(a,b-a))); } catch(...) {}
                }
            }
        }
        auto di = js.find("\"dtype\"", ti);
        if (di != std::string::npos) {
            // find next quoted value after dtype
            auto c = js.find(':', di); if(c!=std::string::npos){
                auto q1=js.find('"',c); auto q2=js.find('"',q1+1);
                if(q1!=std::string::npos&&q2!=std::string::npos) h.tensor_spec.dtype=js.substr(q1+1,q2-q1-1);
            }
        }
    }
    return true;
}

bool validate_header(const Header& h, std::string& err) {
    if (h.protocol_version != PROTOCOL_VERSION) { err="bad protocol_version"; return false; }
    if (h.message_type==MessageType::UNKNOWN) { err="unknown message_type"; return false; }
    if (h.request_id < 0) { err="negative request_id"; return false; }
    if (h.worker_generation < 0) { err="negative worker_generation"; return false; }
    size_t limit = (h.message_type==MessageType::LOAD) ? MAX_MODEL_BYTES : MAX_TENSOR_BYTES;
    if (h.message_type==MessageType::RESULT) limit = RESULT_BYTES;
    if (h.payload_length > limit) { err="payload_length exceeds bound"; return false; }
    if (h.error_message.size() > 1024) { err="error_message too long"; return false; }
    // No pointers cross boundary – artifact_path must be supervisor-controlled absolute path, no traversal
    if (h.message_type==MessageType::LOAD && h.artifact_path.empty()) { err="LOAD requires artifact_path"; return false; }
    if (!h.artifact_path.empty()) {
        if (h.artifact_path.size() > 4096) { err="artifact_path too long"; return false; }
        if (h.artifact_path.find("..") != std::string::npos) { err="artifact_path traversal"; return false; }
    }
    return true;
}

bool validate_tensor_spec(const TensorSpec& spec, size_t payload_len, std::string& err) {
    if (spec.dtype != "float32") { err="unsupported dtype (only float32)"; return false; }
    if (spec.shape.size()!=4) { err="rank must be 4"; return false; }
    if (spec.shape[0]!=1 || spec.shape[1]!=3) { err="shape must be [1,3,H,W]"; return false; }
    for (int d: spec.shape) if (d<=0 || d>4096) { err="invalid tensor dimension"; return false; }
    size_t elems=1;
    for (int d: spec.shape) {
        if (elems > MAX_TENSOR_BYTES/4) { err="tensor exceeds 16 MiB"; return false; }
        elems *= (size_t)d;
    }
    size_t want = elems * 4;
    if (want > MAX_TENSOR_BYTES) { err="tensor exceeds 16 MiB bound"; return false; }
    if (payload_len != want) { err="byte count != shape product"; return false; }
    return true;
}

bool validate_finite(const float* data, size_t count) {
    for (size_t i=0;i<count;++i) if (!std::isfinite(data[i])) return false;
    return true;
}

// Low-level IO helpers: read/write exactly N bytes
static bool write_n(int fd, const void* buf, size_t n) {
    const uint8_t* p=(const uint8_t*)buf;
    while(n){
        ssize_t w=write(fd,p,n);
        if(w<0){ if(errno==EINTR) continue; return false; }
        if(w==0) return false;
        p+=w; n-=w;
    }
    return true;
}
static bool read_n(int fd, void* buf, size_t n) {
    uint8_t* p=(uint8_t*)buf;
    while(n){
        ssize_t r=read(fd,p,n);
        if(r<0){ if(errno==EINTR) continue; return false; }
        if(r==0) return false;
        p+=r; n-=r;
    }
    return true;
}

bool send_message(int fd, const Header& hdr, const uint8_t* payload, size_t payload_len, std::string& err) {
    std::string js = header_to_json(hdr);
    if (js.size() > MAX_HEADER_BYTES) { err="header too large"; return false; }
    if (payload_len> MAX_TENSOR_BYTES && hdr.message_type!=MessageType::RESULT) { err="payload too large"; return false; }
    uint32_t be_len = htonl((uint32_t)js.size());
    if (!write_n(fd,&be_len,4)) { err="write header_len failed"; return false; }
    if (!write_n(fd,js.data(),js.size())) { err="write header failed"; return false; }
    if (payload_len && payload) {
        if (!write_n(fd,payload,payload_len)) { err="write payload failed"; return false; }
    }
    return true;
}

bool recv_message(int fd, Header& hdr, std::vector<uint8_t>& payload, std::string& err) {
    uint32_t be_len=0;
    if (!read_n(fd,&be_len,4)) { err="read header_len failed"; return false; }
    uint32_t hlen = ntohl(be_len);
    if (hlen==0 || hlen>MAX_HEADER_BYTES) { err="header length out of bounds"; return false; }
    std::string js(hlen,'\0');
    if (!read_n(fd,js.data(),hlen)) { err="read header failed"; return false; }
    if (!json_to_header(js,hdr,err)) return false;
    if (!validate_header(hdr,err)) return false;
    payload.clear();
    if (hdr.payload_length){
        size_t limit = (hdr.message_type==MessageType::LOAD) ? MAX_MODEL_BYTES : MAX_TENSOR_BYTES;
        if (hdr.message_type==MessageType::RESULT) limit = RESULT_BYTES;
        if (hdr.payload_length > limit) { err="payload_length exceeds bound"; return false; }
        payload.resize(hdr.payload_length);
        if (!read_n(fd,payload.data(),hdr.payload_length)) { err="read payload failed"; return false; }
    }
    return true;
}

} // namespace fxdna
