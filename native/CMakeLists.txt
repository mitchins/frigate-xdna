cmake_minimum_required(VERSION 3.16)
project(fxdna-worker CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Final binary must link only libflexmlrt, XRT/XDNA userspace, ordinary system libs.
# Do NOT link pyflexmlrt, ORT, VOE, VAIML, xcompiler.

find_package(Threads REQUIRED)

# FlexMLRT headers/libs: use packaged vendor-input or system install.
# Expected locations:
#   packaging/vendor-input/flexmlrt/include  (if extracted)
#   /opt/xilinx/flexmlrt/include
#   /usr/include
set(FLEXMLRT_SEARCH_PATHS
    ${CMAKE_SOURCE_DIR}/../packaging/vendor-input/flexmlrt
    ${CMAKE_SOURCE_DIR}/packaging/vendor-input/flexmlrt
    /opt/xilinx/flexmlrt
    /opt/amd/flexmlrt
    /usr
)

find_path(FLEXMLRT_INCLUDE_DIR FlexMLClient.h
    PATHS ${FLEXMLRT_SEARCH_PATHS}
    PATH_SUFFIXES include flexmlrt/include
    NO_DEFAULT_PATH
)
if(NOT FLEXMLRT_INCLUDE_DIR)
    # fallback to system search
    find_path(FLEXMLRT_INCLUDE_DIR FlexMLClient.h)
endif()
if(NOT FLEXMLRT_INCLUDE_DIR)
    message(WARNING "FlexMLClient.h not found; set FLEXMLRT_INCLUDE_DIR")
    set(FLEXMLRT_INCLUDE_DIR "/opt/xilinx/flexmlrt/include")
endif()

find_library(FLEXMLRT_LIB flexmlrt
    PATHS ${FLEXMLRT_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
)
if(NOT FLEXMLRT_LIB)
    find_library(FLEXMLRT_LIB flexmlrt)
endif()
if(NOT FLEXMLRT_LIB)
    message(STATUS "libflexmlrt not found at configure time; linking by name")
    set(FLEXMLRT_LIB flexmlrt)
endif()

# XRT (XDNA userspace) – vendored runtime via explicit XRT root, fail if missing
set(XRT_ROOT "/tmp/xrt" CACHE PATH "Vendored XRT root (from packaging/vendor-input/xrt)")
find_library(XRT_CORE_LIB xrt_core
    PATHS ${XRT_ROOT}/lib /tmp/xrt/lib ${CMAKE_SOURCE_DIR}/../packaging/vendor-input/xrt/lib
    NO_DEFAULT_PATH
)
find_library(XRT_COREUTIL_LIB xrt_coreutil
    PATHS ${XRT_ROOT}/lib /tmp/xrt/lib ${CMAKE_SOURCE_DIR}/../packaging/vendor-input/xrt/lib
    NO_DEFAULT_PATH
)
if(NOT XRT_CORE_LIB)
    message(FATAL_ERROR "XRT_CORE_LIB not found in vendored ${XRT_ROOT}/lib; pass -DXRT_ROOT=/path/to/xrt")
endif()
if(NOT XRT_COREUTIL_LIB)
    message(FATAL_ERROR "XRT_COREUTIL_LIB not found in vendored ${XRT_ROOT}/lib; pass -DXRT_ROOT=/path/to/xrt")
endif()

add_executable(fxdna-worker
    src/main.cpp
    src/model.cpp
    src/protocol.cpp
    src/postprocess_yolo.cpp
)

target_include_directories(fxdna-worker PRIVATE
    include
    ${FLEXMLRT_INCLUDE_DIR}
)

target_link_libraries(fxdna-worker PRIVATE
    ${FLEXMLRT_LIB}
    ${XRT_CORE_LIB}
    ${XRT_COREUTIL_LIB}
    Threads::Threads
)

# Enforce no disallowed linkage
# Do NOT add: pyflexmlrt, onnxruntime, voe, vaiml, xcompiler
# Verified via: ldd fxdna-worker | egrep -i "onnx|voe|vaiml|xcompiler|pyflexml"

target_compile_options(fxdna-worker PRIVATE -Wall -Wextra -Werror -O2)
target_compile_definitions(fxdna-worker PRIVATE _GNU_SOURCE)

install(TARGETS fxdna-worker DESTINATION bin)
