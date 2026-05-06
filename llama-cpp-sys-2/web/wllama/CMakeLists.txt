cmake_minimum_required(VERSION 3.14)
project("wllama")

option(GGML_WEBGPU "Enable GGML WebGPU backend" ON)
option(GGML_WEBGPU_JSPI "Enable GGML WebGPU JSPI support" ON)
option(LLAMA_WASM_MEM64 "Enable 64-bit memory for WebAssembly builds" OFF)

set(WLLAMA_LLAMA_CPP_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../llama.cpp"
    CACHE PATH "Repo-owned llama.cpp source directory")
add_subdirectory("${WLLAMA_LLAMA_CPP_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp")

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(WLLAMA_MTMD_SRC
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/clip.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/mtmd.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/mtmd-helper.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/mtmd-image.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/mtmd-audio.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/cogvlm.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/conformer.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/deepseekocr.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/dotsocr.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/gemma4a.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/gemma4v.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/glm4v.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/hunyuanocr.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/internvl.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/kimik25.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/kimivl.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/llama4.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/llava.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/minicpmv.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/mobilenetv5.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/nemotron-v2-vl.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/paddleocr.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/pixtral.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/qwen2vl.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/qwen3a.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/qwen3vl.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/siglip.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/step3vl.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/whisper-enc.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/youtuvl.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models/models.h)

set(WLLAMA_SRC cpp/wllama.cpp
    cpp/actions.hpp
    cpp/glue.hpp
    cpp/helpers/wlog.cpp
    cpp/helpers/wcommon.cpp
    cpp/helpers/wsampling.cpp
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/include/llama.h
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/mtmd.h
    ${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/mtmd-helper.h
    ${WLLAMA_MTMD_SRC})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp/helpers)
include_directories(${WLLAMA_LLAMA_CPP_SOURCE_DIR}/include)
include_directories(${WLLAMA_LLAMA_CPP_SOURCE_DIR}/common)
include_directories(${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd)
include_directories(${WLLAMA_LLAMA_CPP_SOURCE_DIR}/tools/mtmd/models)
include_directories(${WLLAMA_LLAMA_CPP_SOURCE_DIR}/vendor)

add_executable(wllama ${WLLAMA_SRC})
target_compile_definitions(wllama PRIVATE WLLAMA_MTMD_BUILD_PATCH_SCAFFOLD=1)
target_link_libraries(wllama PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
