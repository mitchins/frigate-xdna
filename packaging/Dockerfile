# frigate-xdna compiler appliance candidate (Task 03).
#
# Build (from the repo root; vendor input is private and COPY'd):
#   tools/prepare_vendor_input.py --payload-src <audited site-packages> \
#       --xrt-src <audited xrt> --flexmlrt-lib <standalone lib> \
#       --legal-src <licence texts> --calib-src <calib dir>
#   podman build --security-opt apparmor=unconfined \
#       -f packaging/Dockerfile -t frigate-xdna:0.1-dev .
#   # docker alternative (LXC needs the same apparmor bypass via buildx):
#   # docker buildx build --allow security.insecure ... --security=insecure
#
# Prefixes stay separate by construction:
#   /opt/fxdna/manager        supervisor/CLI (Tasks 01-02 source)
#   /opt/compile-venv         audited compiler payload (venv site-packages)
#   /opt/quant-venv           public-PyPI quant stack (Quark/torch/ORT-CPU)
#   /opt/xilinx-xrt           minimal XRT userspace + shim
#   /opt/flexmlrt             standalone FlexMLRT runtime lib
#   /opt/fxdna/legal          EULA/TPN flow-down notices + component map
#   /opt/fxdna/recipes       pinned recipe + manifests
#   /data                     persistent registry/cache/work (volume)

ARG UBUNTU_TAG=24.04
FROM ubuntu:${UBUNTU_TAG} AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3.12-venv libpython3.12 \
      python3.12 \
      libstdc++6 libgomp1 libncurses6 libexpat1 zlib1g \
      ca-certificates \
      build-essential python3.12-dev ninja-build \
    && rm -rf /var/lib/apt/lists/*
# build-essential/python3.12-dev/ninja are ordinary prerequisites for the
# Quark JIT custom-op build (audited Phase-7.7 requirement); no AMD fetch.

FROM base AS manager
COPY src /opt/fxdna/manager/src
COPY pyproject.toml /opt/fxdna/manager/
COPY packaging/constraints-manager.txt /tmp/constraints-manager.txt
RUN python3.12 -m venv /opt/manager-venv \
 && /opt/manager-venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
 && /opt/manager-venv/bin/pip install --no-cache-dir \
      -c /tmp/constraints-manager.txt \
      requests==2.34.2 onnx==1.23.0 \
 && /opt/manager-venv/bin/pip install --no-cache-dir --no-deps \
      --no-build-isolation /opt/fxdna/manager
ENV PATH=/opt/manager-venv/bin:$PATH
ENV PYTHONPATH=/opt/fxdna/manager/src

FROM base AS compiler
RUN python3.12 -m venv /opt/compile-venv
COPY packaging/vendor-files.manifest.json /tmp/
COPY packaging/verify_payload.py /tmp/
COPY packaging/vendor-input/compile-site-packages/ /opt/compile-venv/lib/python3.12/site-packages/
COPY packaging/vendor-input/xrt/ /opt/xilinx-xrt/
RUN ln -sf libxrt_core.so.2.25.37 /opt/xilinx-xrt/lib/libxrt_core.so.2 \
 && ln -sf libxrt_coreutil.so.2.25.37 /opt/xilinx-xrt/lib/libxrt_coreutil.so.2 \
 && ln -sf libxrt_driver_xdna.so.2.25.260102.56.release \
      /opt/xilinx-xrt/lib/libxrt_driver_xdna.so.2 \
 && /opt/compile-venv/bin/pip install --no-cache-dir onnx==1.23.0 numpy==2.5.3 \
 && python3 /tmp/verify_payload.py \
      --manifest /tmp/vendor-files.manifest.json \
      --payload /opt/compile-venv/lib/python3.12/site-packages \
      --xrt /opt/xilinx-xrt

FROM base AS quant
COPY packaging/constraints-quant.txt /tmp/constraints-quant.txt
RUN python3.12 -m venv /opt/quant-venv \
  && /opt/quant-venv/bin/pip install --no-cache-dir --upgrade pip \
  && /opt/quant-venv/bin/pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       -c /tmp/constraints-quant.txt torch==2.14.0+cpu \
  && /opt/quant-venv/bin/pip install --no-cache-dir \
       -c /tmp/constraints-quant.txt \
       amd-quark==0.12.post1 onnx==1.22.0 onnxruntime==1.30.0 \
       pillow==12.3.0 numpy==2.5.3 \
  && /opt/quant-venv/bin/python -c \
       "from quark.onnx import ModelQuantizer; print('quark import OK')"

FROM base AS native
# Build resident native worker (Task 04): single Model/context per process,
# mmap lifetime preserved, private IPC, no compiler libs.
# Uses the same FlexMLRT/XRT userspace as the proven runner.
COPY native /tmp/native
COPY packaging/vendor-input/flexmlrt /tmp/flexmlrt
COPY packaging/vendor-input/xrt /tmp/xrt
RUN apt-get update && apt-get install -y --no-install-recommends cmake g++ pkg-config \
    && mkdir -p /tmp/native/build \
    && cmake -S /tmp/native -B /tmp/native/build \
        -DFLEXMLRT_INCLUDE_DIR=/tmp/flexmlrt/include \
        -DFLEXMLRT_LIB=/tmp/flexmlrt/lib/libflexmlrt.so \
        -DXRT_ROOT=/tmp/xrt \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/native/build -j4 \
    && mkdir -p /opt/fxdna/native \
    && cp /tmp/native/build/fxdna-worker /opt/fxdna/native/ \
    && rm -rf /var/lib/apt/lists/* /tmp/native/build

FROM base AS final
COPY --from=manager /opt/manager-venv /opt/manager-venv
COPY --from=manager /opt/fxdna/manager /opt/fxdna/manager
COPY --from=compiler /opt/compile-venv /opt/compile-venv
COPY --from=compiler /opt/xilinx-xrt /opt/xilinx-xrt
COPY --from=quant /opt/quant-venv /opt/quant-venv
COPY --from=native /opt/fxdna/native /opt/fxdna/native
# Vendor groups are copied and then verified byte-for-byte; build fails
# if any shipped file is missing, extra, or hash-mismatched.
COPY packaging/vendor-input/flexmlrt/ /opt/flexmlrt/lib/
COPY packaging/vendor-input/calib/ /opt/fxdna/calib/
COPY packaging/vendor-input/legal/ /opt/fxdna/legal/
COPY packaging/vendor-files.manifest.json /tmp/vendor-files.manifest.json
COPY packaging/verify_payload.py /tmp/verify_payload.py
RUN python3 /tmp/verify_payload.py \
      --manifest /tmp/vendor-files.manifest.json \
      --payload /opt/compile-venv/lib/python3.12/site-packages \
      --xrt /opt/xilinx-xrt \
      --flexmlrt /opt/flexmlrt/lib \
      --calib /opt/fxdna/calib \
      --legal /opt/fxdna/legal
COPY packaging/compile-env.sh /opt/fxdna/
COPY recipes/bf16-vaiml-v1/recipe.json /opt/fxdna/recipes/bf16-vaiml-v1/
COPY recipes/bf16-vaiml-v1/prepare.py recipes/bf16-vaiml-v1/compile.py recipes/bf16-vaiml-v1/validate.py /opt/fxdna/recipes/bf16-vaiml-v1/
COPY packaging/vendor.lock.json packaging/vendor-files.manifest.json /opt/fxdna/recipes/
COPY packaging/legal/component-map.json /opt/fxdna/legal/
COPY THIRD_PARTY_NOTICES.md /opt/fxdna/legal/
COPY schemas /opt/fxdna/schemas
RUN useradd -u 10001 -m -d /home/fxdna -s /usr/sbin/nologin fxdna \
 && mkdir -p /data /home/fxdna && chown 10001:10001 /data /home/fxdna
ENV PATH=/opt/manager-venv/bin:$PATH \
    PYTHONPATH=/opt/fxdna/manager/src \
    HOME=/home/fxdna
USER 10001:10001
VOLUME /data
ENTRYPOINT ["fxdna"]
CMD ["serve"]
