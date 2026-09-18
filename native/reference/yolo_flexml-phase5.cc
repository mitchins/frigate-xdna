// Standalone FlexMLRT YOLO runner (no ORT/VOE/VAIML).
// Follows upstream whisper.cpp's VitisAI integration pattern:
// mmap .rai -> Options{fbs_buffer, fbs_buffer_size, cache_dir} -> Model ->
// getIOTensors -> bind user buffers -> forward.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <numeric>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <vector>

#include "FlexMLClient.h"

static uint8_t* g_map = nullptr;
static size_t g_map_size = 0;

static bool map_rai(const char* path, uint8_t** buf, size_t* size) {
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

static void print_tensors(const char* tag,
                          const std::vector<flexmlrt::client::ErtTensorType>& ts) {
    std::printf("%s: %zu tensor(s)\n", tag, ts.size());
    for (size_t i = 0; i < ts.size(); ++i) {
        const auto& meta = ts[i].getMetadata();
        std::printf("  [%zu] name='%s' size=%zu dtype=%d shape=",
                    i, meta.name.c_str(), meta.size, (int)meta.type);
        for (auto d : meta.shape) std::printf("%u ", d);
        std::printf("\n");
    }
}

int main(int argc, char** argv) {
    const char* rai = argc > 1 ? argv[1] : "/root/xdna/yolo_cache/yolov8n/yolov8n.rai";
    const char* inbin = argc > 2 ? argv[2] : "/root/xdna/phase5/ref_input.npy";
    const char* outbin = argc > 3 ? argv[3] : "/root/xdna/phase5/flexml_out0.bin";
    int bench_n = argc > 4 ? std::atoi(argv[4]) : 0;

    // load reference input (skip 128-byte .npy header)
    FILE* f = std::fopen(inbin, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", inbin); return 1; }
    std::fseek(f, 0, SEEK_END);
    long fsize = std::ftell(f);
    std::fseek(f, 128, SEEK_SET);
    std::vector<float> input((fsize - 128) / sizeof(float));
    if (std::fread(input.data(), 1, fsize - 128, f) != (size_t)(fsize - 128)) {
        std::fprintf(stderr, "short read\n");
        return 1;
    }
    std::fclose(f);
    std::printf("input floats=%zu\n", input.size());

    if (!map_rai(rai, &g_map, &g_map_size)) {
        std::fprintf(stderr, "mmap rai failed\n");
        return 1;
    }
    std::printf("rai mapped %zu bytes\n", g_map_size);

    flexmlrt::client::Options options;
    options.modelPath = rai;
    options.debug = true;
    options.executeMode = 2;
    options.deviceName = "stx";
    options.subgraphName = "vaiml_par_0";
    options.extOptions["enable_preemption"] = true;
    options.extOptions["fbs_buffer"] = g_map;
    options.extOptions["fbs_buffer_size"] = g_map_size;
    options.extOptions["cache_dir"] = std::string(".");

    auto t0 = std::chrono::steady_clock::now();
    flexmlrt::client::Model runner(options);
    if (!runner.good()) {
        std::fprintf(stderr, "Runner creation failed\n");
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    std::printf("LOAD_OK %.2fs\n",
                std::chrono::duration<double>(t1 - t0).count());

    auto in_tensors = runner.getIOTensors("input", false);
    auto out_tensors = runner.getIOTensors("output", false);
    print_tensors("input", in_tensors);
    print_tensors("output", out_tensors);
    if (in_tensors.empty() || out_tensors.empty()) {
        std::fprintf(stderr, "no io tensors\n");
        return 1;
    }
    size_t in_elems = 1;
    for (auto d : in_tensors[0].getMetadata().shape) in_elems *= d;
    if (in_elems != input.size())
        std::fprintf(stderr, "warn: model input elems %zu != provided %zu\n",
                     in_elems, input.size());
    in_tensors[0].data = input.data();

    size_t out_elems = 1;
    for (auto d : out_tensors[0].getMetadata().shape) out_elems *= d;
    std::vector<float> output(out_elems);
    out_tensors[0].data = output.data();

    t0 = std::chrono::steady_clock::now();
    runner.forward(in_tensors, out_tensors);
    t1 = std::chrono::steady_clock::now();
    std::printf("RUN_OK %.3fs\n", std::chrono::duration<double>(t1 - t0).count());

    FILE* fo = std::fopen(outbin, "wb");
    std::fwrite(output.data(), 1, output.size() * sizeof(float), fo);
    std::fclose(fo);
    std::printf("wrote %s (%zu floats)\n", outbin, output.size());

    const char* sus = std::getenv("FLEXML_SUSTAIN_SECS");
    if (sus) {
        double secs = std::atof(sus);
        auto end = std::chrono::steady_clock::now() +
                   std::chrono::duration<double>(secs);
        long n = 0, errs = 0;
        auto t0s = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() < end) {
            try {
                runner.forward(in_tensors, out_tensors);
                ++n;
            } catch (...) { ++errs; }
        }
        double dt =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0s).count();
        std::printf("SUSTAINED secs=%.1f inferences=%ld fps=%.1f errors=%ld\n",
                    dt, n, n / dt, errs);
    }

    if (bench_n > 0) {
        std::vector<double> ts;
        for (int i = 0; i < bench_n; ++i) {
            t0 = std::chrono::steady_clock::now();
            runner.forward(in_tensors, out_tensors);
            t1 = std::chrono::steady_clock::now();
            ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(ts.begin(), ts.end());
        std::printf("bench n=%d p50=%.2f p90=%.2f fps=%.1f\n", bench_n,
                    ts[bench_n / 2], ts[(int)(bench_n * 0.9)],
                    1000.0 / (std::accumulate(ts.begin(), ts.end(), 0.0) / bench_n));
    }
    return 0;
}
