# Uni Core Web wllama Runtime

This directory is the repo-owned Web wllama runtime used by `uni-core` smoke
tests and the Web dev server.

The runtime sources were derived from the local wllama reference checkout used
before Phase 81, but this copy builds against the single repo-owned llama.cpp
source at `../../llama.cpp`. The reference checkout is no longer a default
runtime dependency; use an explicit `LOCAL_LLM_WEB_WLLAMA_DIR` or `--wllama-dir`
override when comparing against another checkout.

Tracked runtime artifacts:

- `src/asyncify-single-thread/wllama.{js,wasm}`
- `src/jspi-single-thread/wllama.{js,wasm}`
- `src/webgpu-single-thread/wllama.{js,wasm}`

The C++ glue in `cpp/` and the generated glue schema in `src/glue/` are kept
with those artifacts so source contract checks can audit which actions the Web
runtime exposes.
