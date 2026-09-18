// xdna-zmq-worker: persistent XDNA inference worker speaking stock Frigate
// ZmqIpcDetector protocol (REQ/REP). Loads one .rai via standalone FlexMLRT,
// serves model mgmt + inference with Frigate [20,6] float32 responses.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>
#include <zmq.h>

#include "FlexMLClient.h"

namespace {
constexpr int kW = 640, kH = 640;
constexpr int kMaxDet = 20;
constexpr float kConf = 0.25f, kIou = 0.45f;

uint8_t* g_map = nullptr;
size_t g_map_size = 0;
std::string g_model_path;
flexmlrt::client::Model* g_runner = nullptr;
std::vector<flexmlrt::client::ErtTensorType> g_in, g_out;
std::vector<float> g_outbuf;
long g_requests = 0;
double g_infer_ms_total = 0;

bool map_rai(const char* path, uint8_t** buf, size_t* size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) || st.st_size <= 0) { close(fd); return false; }
    void* p = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return false;
    *buf = (uint8_t*)p; *size = st.st_size;
    return true;
}

bool load_model(const std::string& rai_path) {
    if (g_runner) { delete g_runner; g_runner = nullptr; }
    if (g_map) { munmap(g_map, g_map_size); g_map = nullptr; }
    if (!map_rai(rai_path.c_str(), &g_map, &g_map_size)) {
        std::fprintf(stderr, "worker: mmap failed for %s\n", rai_path.c_str());
        return false;
    }
    flexmlrt::client::Options options;
    options.modelPath = rai_path;
    options.executeMode = 2;
    options.deviceName = "stx";
    options.subgraphName = "vaiml_par_0";
    options.extOptions["enable_preemption"] = true;
    options.extOptions["fbs_buffer"] = g_map;
    options.extOptions["fbs_buffer_size"] = g_map_size;
    options.extOptions["cache_dir"] = std::string(".");
    g_runner = new flexmlrt::client::Model(options);
    if (!g_runner->good()) {
        std::fprintf(stderr, "worker: Model creation failed for %s\n", rai_path.c_str());
        delete g_runner; g_runner = nullptr;
        return false;
    }
    g_in = g_runner->getIOTensors("input", false);
    g_out = g_runner->getIOTensors("output", false);
    if (g_in.empty() || g_out.empty()) return false;
    g_model_path = rai_path;
    g_outbuf.assign(84 * 8400, 0.0f);
    std::fprintf(stderr, "worker: loaded %s in=%zu out=%zu\n", rai_path.c_str(),
                 g_in.size(), g_out.size());
    return true;
}

std::string basename_of(const std::string& p) {
    auto i = p.find_last_of('/');
    return i == std::string::npos ? p : p.substr(i + 1);
}

// Minimal JSON field extraction (flat string/bool values only).
std::string jfield(const std::string& js, const char* key) {
    std::string q = std::string("\"") + key + "\"";
    auto i = js.find(q);
    if (i == std::string::npos) return "";
    i = js.find(':', i);
    if (i == std::string::npos) return "";
    ++i;
    while (i < js.size() && isspace((unsigned char)js[i])) ++i;
    if (i < js.size() && js[i] == '"') {
        auto j = js.find('"', i + 1);
        return j == std::string::npos ? "" : js.substr(i + 1, j - i - 1);
    }
    auto j = js.find_first_of(",}", i);
    return js.substr(i, j == std::string::npos ? j : j - i);
}

// YOLOv8 decode (84,8400): rows 0-3 cxcywh px, rows 4-83 class scores.
int postprocess(const float* p, float out20x6[kMaxDet][6]) {
    struct Det { float s; int c; float x1, y1, x2, y2; };
    std::vector<Det> ds;
    for (int i = 0; i < 8400; ++i) {
        float best = 0; int bc = -1;
        for (int c = 0; c < 80; ++c) {
            float s = p[(4 + c) * 8400 + i];
            if (s > best) { best = s; bc = c; }
        }
        if (best < kConf || bc < 0) continue;
        float cx = p[0 * 8400 + i], cy = p[1 * 8400 + i];
        float w = p[2 * 8400 + i], h = p[3 * 8400 + i];
        ds.push_back({best, bc, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2});
    }
    std::sort(ds.begin(), ds.end(), [](const Det& a, const Det& b) { return a.s > b.s; });
    std::vector<Det> keep;
    std::vector<float> areas(ds.size());
    for (size_t i = 0; i < ds.size(); ++i)
        areas[i] = (ds[i].x2 - ds[i].x1) * (ds[i].y2 - ds[i].y1);
    std::vector<char> sup(ds.size(), 0);
    for (size_t i = 0; i < ds.size() && (int)keep.size() < kMaxDet; ++i) {
        if (sup[i]) continue;
        keep.push_back(ds[i]);
        for (size_t j = i + 1; j < ds.size(); ++j) {
            if (sup[j]) continue;
            float xx1 = std::max(ds[i].x1, ds[j].x1), yy1 = std::max(ds[i].y1, ds[j].y1);
            float xx2 = std::min(ds[i].x2, ds[j].x2), yy2 = std::min(ds[i].y2, ds[j].y2);
            float ov = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1) /
                       (areas[i] + areas[j] - std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1) + 1e-9f);
            if (ov > kIou) sup[j] = 1;
        }
    }
    memset(out20x6, 0, sizeof(float) * kMaxDet * 6);
    for (size_t i = 0; i < keep.size(); ++i) {
        out20x6[i][0] = (float)keep[i].c;
        out20x6[i][1] = keep[i].s;
        out20x6[i][2] = keep[i].y1 / kH;
        out20x6[i][3] = keep[i].x1 / kW;
        out20x6[i][4] = keep[i].y2 / kH;
        out20x6[i][5] = keep[i].x2 / kW;
    }
    return (int)keep.size();
}

void send_json(void* sock, const std::string& js) {
    zmq_send(sock, js.data(), js.size(), 0);
}
void send_zeros(void* sock) {
    static float z[kMaxDet][6] = {};
    zmq_send(sock, z, sizeof(z), 0);
}
}  // namespace

int main(int argc, char** argv) {
    const char* endpoint = argc > 1 ? argv[1] : "ipc:///tmp/cache/zmq_detector";
    const char* rai = argc > 2 ? argv[2] : "/root/xdna/yolo_cache/yolov8n/yolov8n.rai";
    const char* workdir = argc > 3 ? argv[3] : "/root/xdna/phase7/work";
    mkdir(workdir, 0755);

    if (!load_model(rai)) return 1;

    void* ctx = zmq_ctx_new();
    void* sock = zmq_socket(ctx, ZMQ_REP);
    if (zmq_bind(sock, endpoint) != 0) {
        std::fprintf(stderr, "worker: bind %s failed: %s\n", endpoint, zmq_strerror(zmq_errno()));
        return 1;
    }
    std::fprintf(stderr, "worker: listening on %s model=%s\n", endpoint, rai);

    float out20x6[kMaxDet][6];
    while (true) {
        zmq_msg_t hmsg, dmsg;
        zmq_msg_init(&hmsg);
        if (zmq_msg_recv(&hmsg, sock, 0) < 0) { zmq_msg_close(&hmsg); continue; }
        std::string header((char*)zmq_msg_data(&hmsg), zmq_msg_size(&hmsg));
        bool more = zmq_msg_more(&hmsg);
        std::string data;
        if (more) {
            zmq_msg_init(&dmsg);
            if (zmq_msg_recv(&dmsg, sock, 0) < 0) { zmq_msg_close(&hmsg); zmq_msg_close(&dmsg); continue; }
            data.assign((char*)zmq_msg_data(&dmsg), zmq_msg_size(&dmsg));
            zmq_msg_close(&dmsg);
        }
        zmq_msg_close(&hmsg);

        std::string mr = jfield(header, "model_request");
        std::string md = jfield(header, "model_data");
        if (!mr.empty() && mr.find("true") != std::string::npos) {
            std::string want = jfield(header, "model_name");
            bool avail = !want.empty() &&
                         (want == basename_of(g_model_path) || access((std::string(workdir) + "/" + want).c_str(), R_OK) == 0);
            bool loaded = avail && want == basename_of(g_model_path) && g_runner && g_runner->good();
            char r[256];
            std::snprintf(r, sizeof(r), "{\"model_available\": %s, \"model_loaded\": %s}",
                          avail ? "true" : "false", loaded ? "true" : "false");
            std::fprintf(stderr, "worker: model_request name=%s -> avail=%d loaded=%d\n",
                         want.c_str(), avail, loaded);
            send_json(sock, r);
        } else if (!md.empty() && md.find("true") != std::string::npos) {
            std::string name = jfield(header, "model_name");
            std::string dest = std::string(workdir) + "/" + (name.empty() ? "model.rai" : name);
            bool ok = false, lok = false;
            if (!name.empty() && !data.empty() && name.find('/') == std::string::npos &&
                name.find('\\') == std::string::npos) {
                FILE* f = fopen(dest.c_str(), "wb");
                if (f) {
                    ok = fwrite(data.data(), 1, data.size(), f) == data.size();
                    fclose(f);
                }
                if (ok) {
                    std::fprintf(stderr, "worker: saved %zu bytes to %s, loading\n",
                                 data.size(), dest.c_str());
                    lok = load_model(dest);
                }
            } else {
                std::fprintf(stderr, "worker: rejected model transfer (bad name/empty)\n");
            }
            char r[256];
            std::snprintf(r, sizeof(r), "{\"model_saved\": %s, \"model_loaded\": %s}",
                          ok ? "true" : "false", lok ? "true" : "false");
            send_json(sock, r);
        } else {
            // inference: header {shape, dtype, model_type}
            bool valid = false;
            float* ten = nullptr;
            if (header.find("\"shape\"") != std::string::npos &&
                header.find("float32") != std::string::npos &&
                data.size() == (size_t)1 * 3 * 640 * 640 * 4) {
                bool finite = true;
                ten = (float*)data.data();
                for (size_t i = 0; i < 1228800; i += 4096)
                    if (!std::isfinite(ten[i])) { finite = false; break; }
                valid = finite;
                if (!valid) std::fprintf(stderr, "worker: non-finite tensor values\n");
            } else {
                std::fprintf(stderr, "worker: malformed inference request hdr=%.80s bytes=%zu\n",
                             header.c_str(), data.size());
            }
            if (!valid || !g_runner) { send_zeros(sock); continue; }
            g_in[0].data = ten;
            g_out[0].data = g_outbuf.data();
            auto t0 = std::chrono::steady_clock::now();
            try {
                g_runner->forward(g_in, g_out);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "worker: forward failed: %s\n", e.what());
                send_zeros(sock);
                continue;
            }
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
            ++g_requests;
            g_infer_ms_total += ms;
            int n = postprocess(g_outbuf.data(), out20x6);
            if (g_requests % 200 == 0)
                std::fprintf(stderr, "worker: req=%ld avg_infer=%.2fms last_n=%d\n",
                             g_requests, g_infer_ms_total / g_requests, n);
            zmq_send(sock, out20x6, sizeof(out20x6), 0);
        }
    }
    return 0;
}
