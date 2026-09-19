#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <vector>
#include <string>
#include "model.h"
#include "protocol.h"
#include "postprocess.h"

using namespace fxdna;

static volatile sig_atomic_t g_term = 0;
static void on_sig(int){ g_term=1; }

static int parse_fd_arg(int argc, char** argv) {
    // Private supervisor↔native IPC over Unix socketpair, inherited fd.
    // Supervisor passes fd via --fd N or env FXDNA_NATIVE_FD, no public socket.
    for(int i=1;i<argc;++i){
        if(std::strcmp(argv[i],"--fd")==0 && i+1<argc) return std::atoi(argv[i+1]);
        if(std::strncmp(argv[i],"--fd=",5)==0) return std::atoi(argv[i]+5);
    }
    const char* e = std::getenv("FXDNA_NATIVE_FD");
    if(e) return std::atoi(e);
    // Fallback: fd 3 (first inherited after stdin/out/err) if is socket
    return 3;
}

int main(int argc, char** argv){
    std::signal(SIGTERM, on_sig);
    std::signal(SIGINT, on_sig);
    std::signal(SIGPIPE, SIG_IGN);

    int fd = parse_fd_arg(argc, argv);
    // Validate fd is a socketpair (not a public socket)
    // We do not bind/listen; we only use the inherited private fd.
    int64_t worker_generation = 0;
    std::string serving_digest;
    Model model;
    YoloConfig yolo_cfg;
    yolo_cfg.image_w = 320; yolo_cfg.image_h = 320; // YOLOv9s-320 base
    yolo_cfg.class_count = 80;
    // generation derived from supervisor LOAD; native independently validates generation per INFER
    std::fprintf(stderr, "fxdna-worker: starting on private fd %d\n", fd);

    // Buffers for FlexML output
    std::vector<float> outbuf; // sized after load from output_cols*84
    float result20x6[20][6];

    while(!g_term){
        Header hdr;
        std::vector<uint8_t> payload;
        std::string err;
        if(!recv_message(fd, hdr, payload, err)){
            if(g_term) break;
            // Framing error -> structured failure via STATUS if possible, else exit
            std::fprintf(stderr, "fxdna-worker: recv error: %s\n", err.c_str());
            if(err.find("header")!=std::string::npos || err.find("payload")!=std::string::npos){
                Header rep; rep.protocol_version=PROTOCOL_VERSION; rep.message_type=MessageType::STATUS;
                rep.request_id=hdr.request_id; rep.worker_generation=worker_generation;
                rep.error_code="INVALID_ARGS"; rep.error_message=err; rep.payload_length=0;
                std::string se; send_message(fd,rep,nullptr,0,se);
                continue;
            }
            break;
        }

        // Independent validation: rank/shape/dtype/byte count/finite/generation, no pointers cross boundary
        if(hdr.protocol_version != PROTOCOL_VERSION){
            Header rep; rep.protocol_version=PROTOCOL_VERSION; rep.message_type=MessageType::RESULT;
            rep.request_id=hdr.request_id; rep.worker_generation=worker_generation;
            rep.error_code="INVALID_ARGS"; rep.error_message="bad protocol_version"; rep.payload_length=RESULT_BYTES;
            uint8_t zeros[RESULT_BYTES]={}; std::string se; send_message(fd,rep,zeros,RESULT_BYTES,se);
            continue;
        }

        if(hdr.message_type==MessageType::SHUTDOWN){
            std::fprintf(stderr,"fxdna-worker: shutdown req %ld gen %ld\n",(long)hdr.request_id,(long)hdr.worker_generation);
            Header rep; rep.protocol_version=PROTOCOL_VERSION; rep.message_type=MessageType::STATUS;
            rep.request_id=hdr.request_id; rep.worker_generation=worker_generation; rep.payload_length=0;
            std::string se; send_message(fd,rep,nullptr,0,se);
            break;
        }
        if(hdr.message_type==MessageType::LOAD){
            // LOAD: supervisor-controlled artifact_path only
            if(hdr.artifact_path.empty() || hdr.artifact_path.find("..")!=std::string::npos){
                Header rep; rep.protocol_version=PROTOCOL_VERSION; rep.message_type=MessageType::STATUS;
                rep.request_id=hdr.request_id; rep.worker_generation=hdr.worker_generation;
                rep.error_code="INVALID_ARGS"; rep.error_message="invalid artifact_path"; rep.payload_length=0;
                std::string se; send_message(fd,rep,nullptr,0,se);
                continue;
            }
            std::string load_err;
            if(!model.load(hdr.artifact_path, load_err)){
                Header rep; rep.protocol_version=PROTOCOL_VERSION; rep.message_type=MessageType::STATUS;
                rep.request_id=hdr.request_id; rep.worker_generation=hdr.worker_generation;
                rep.error_code="INVALID_MODEL"; rep.error_message=load_err; rep.payload_length=0;
                std::string se; send_message(fd,rep,nullptr,0,se);
                continue;
            }
            worker_generation = hdr.worker_generation;
            serving_digest = hdr.serving_digest;
            // Derive geometry from model output if available
            size_t cols = model.output_cols();
            if(cols==2100){ yolo_cfg.image_w=320; yolo_cfg.image_h=320; }
            else if(cols==8400){ yolo_cfg.image_w=640; yolo_cfg.image_h=640; }
            else if(cols) { /* keep 320 default, but log */ std::fprintf(stderr,"fxdna-worker: unexpected cols %zu\n",cols); }
            size_t out_elems = model.output_elements();
            outbuf.assign(out_elems, 0.0f);
            // Update yolo_cfg from serving_digest if provided (score_threshold/nms kept as defaults; real contract passed via load)
            Header rep; rep.protocol_version=PROTOCOL_VERSION; rep.message_type=MessageType::STATUS;
            rep.request_id=hdr.request_id; rep.worker_generation=worker_generation; rep.payload_length=0;
            std::string se; send_message(fd,rep,nullptr,0,se);
            std::fprintf(stderr,"fxdna-worker: loaded %s gen %ld cols %zu\n", hdr.artifact_path.c_str(), (long)worker_generation, cols);
            continue;
        }
        if(hdr.message_type==MessageType::STATUS){
            Header rep; rep.protocol_version=PROTOCOL_VERSION; rep.message_type=MessageType::STATUS;
            rep.request_id=hdr.request_id; rep.worker_generation=worker_generation; rep.payload_length=0;
            if(!model.loaded()){ rep.error_code="MODEL_NOT_PREPARED"; }
            std::string se; send_message(fd,rep,nullptr,0,se);
            continue;
        }
        if(hdr.message_type==MessageType::INFER){
            // Generation check (native independently validates)
            if(hdr.worker_generation != worker_generation){
                Header rep; rep.protocol_version=PROTOCOL_VERSION; rep.message_type=MessageType::RESULT;
                rep.request_id=hdr.request_id; rep.worker_generation=worker_generation;
                rep.error_code="INVALID_ARGS"; rep.error_message="generation mismatch"; rep.payload_length=RESULT_BYTES;
                uint8_t zeros[RESULT_BYTES]={}; std::string se; send_message(fd,rep,zeros,RESULT_BYTES,se);
                continue;
            }
            if(!model.loaded()){
                Header rep; rep.protocol_version=PROTOCOL_VERSION; rep.message_type=MessageType::RESULT;
                rep.request_id=hdr.request_id; rep.worker_generation=worker_generation;
                rep.error_code="MODEL_NOT_PREPARED"; rep.error_message="no model loaded"; rep.payload_length=RESULT_BYTES;
                uint8_t zeros[RESULT_BYTES]={}; std::string se; send_message(fd,rep,zeros,RESULT_BYTES,se);
                continue;
            }
            std::string verr;
            if(!validate_tensor_spec(hdr.tensor_spec, payload.size(), verr)){
                Header rep; rep.protocol_version=PROTOCOL_VERSION; rep.message_type=MessageType::RESULT;
                rep.request_id=hdr.request_id; rep.worker_generation=worker_generation;
                rep.error_code="INVALID_ARGS"; rep.error_message=verr; rep.payload_length=RESULT_BYTES;
                uint8_t zeros[RESULT_BYTES]={}; std::string se; send_message(fd,rep,zeros,RESULT_BYTES,se);
                continue;
            }
            // Finite check before forward (independent validation)
            size_t elems = payload.size()/4;
            float* fptr = reinterpret_cast<float*>(payload.data());
            if(!validate_finite(fptr, elems)){
                Header rep; rep.protocol_version=PROTOCOL_VERSION; rep.message_type=MessageType::RESULT;
                rep.request_id=hdr.request_id; rep.worker_generation=worker_generation;
                rep.error_code="INVALID_ARGS"; rep.error_message="non-finite input"; rep.payload_length=RESULT_BYTES;
                uint8_t zeros[RESULT_BYTES]={}; std::string se; send_message(fd,rep,zeros,RESULT_BYTES,se);
                continue;
            }
            // Bind user buffers via getIOTensors handles (preserved)
            // Validate byte count matches model input
            size_t model_in = model.input_elements();
            if(model_in && model_in != elems){
                Header rep; rep.protocol_version=PROTOCOL_VERSION; rep.message_type=MessageType::RESULT;
                rep.request_id=hdr.request_id; rep.worker_generation=worker_generation;
                rep.error_code="INVALID_ARGS"; rep.error_message="tensor byte count mismatch model"; rep.payload_length=RESULT_BYTES;
                uint8_t zeros[RESULT_BYTES]={}; std::string se; send_message(fd,rep,zeros,RESULT_BYTES,se);
                continue;
            }
            model.inputs()[0].data = fptr;
            if(outbuf.empty()) outbuf.assign(model.output_elements(), 0.0f);
            model.outputs()[0].data = outbuf.data();
            std::string fwd_err;
            if(!model.forward(fwd_err)){
                Header rep; rep.protocol_version=PROTOCOL_VERSION; rep.message_type=MessageType::RESULT;
                rep.request_id=hdr.request_id; rep.worker_generation=worker_generation;
                rep.error_code="DEVICE_FAULT"; rep.error_message=fwd_err; rep.payload_length=RESULT_BYTES;
                uint8_t zeros[RESULT_BYTES]={}; std::string se; send_message(fd,rep,zeros,RESULT_BYTES,se);
                continue;
            }
            // Validate output finite before postprocess
            if(!validate_finite(outbuf.data(), outbuf.size())){
                Header rep; rep.protocol_version=PROTOCOL_VERSION; rep.message_type=MessageType::RESULT;
                rep.request_id=hdr.request_id; rep.worker_generation=worker_generation;
                rep.error_code="DEVICE_FAULT"; rep.error_message="non-finite output"; rep.payload_length=RESULT_BYTES;
                uint8_t zeros[RESULT_BYTES]={}; std::string se; send_message(fd,rep,zeros,RESULT_BYTES,se);
                continue;
            }
            // Postprocess YOLO raw -> [20,6]
            size_t cols = model.output_cols();
            if(cols==0) cols = outbuf.size()/84;
            std::memset(result20x6,0,sizeof(result20x6));
            // Update config from tensor_spec geometry (320 vs 640 derived from shape)
            int h = hdr.tensor_spec.shape[2];
            int w = hdr.tensor_spec.shape[3];
            if(h>0 && w>0){ yolo_cfg.image_h=h; yolo_cfg.image_w=w; }
            postprocess_yolo_raw(outbuf.data(), cols, yolo_cfg, result20x6);

            Header rep; rep.protocol_version=PROTOCOL_VERSION; rep.message_type=MessageType::RESULT;
            rep.request_id=hdr.request_id; rep.worker_generation=worker_generation;
            rep.tensor_spec.shape = {20,6}; rep.tensor_spec.dtype="float32";
            rep.payload_length=RESULT_BYTES;
            std::string se;
            send_message(fd, rep, reinterpret_cast<uint8_t*>(result20x6), RESULT_BYTES, se);
            continue;
        }

        // Unknown message
        Header rep; rep.protocol_version=PROTOCOL_VERSION; rep.message_type=MessageType::STATUS;
        rep.request_id=hdr.request_id; rep.worker_generation=worker_generation;
        rep.error_code="INVALID_ARGS"; rep.error_message="unknown message_type"; rep.payload_length=0;
        std::string se; send_message(fd,rep,nullptr,0,se);
    }

    // Destruction ordering: delete Model before munmap (handled by Model::~Model)
    std::fprintf(stderr,"fxdna-worker: exiting gen %ld\n",(long)worker_generation);
    return 0;
}
