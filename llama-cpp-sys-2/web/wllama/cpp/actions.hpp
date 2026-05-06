#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdio.h>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "helpers/wcommon.h"
#include "helpers/wsampling.h"

#include "glue.hpp"

static void wllama_native_breadcrumb(const char *text);

#define PARSE_REQ(msg_typename) \
  msg_typename req;             \
  glue_inbuf inbuf(req_raw);    \
  req.handler.deserialize(inbuf);

using mtmd_context_ptr = mtmd::context_ptr;
using mtmd_input_chunks_ptr = mtmd::input_chunks_ptr;

struct app_t
{
  ggml_backend_dev_t device = nullptr;
  llama_model *model;
  llama_context *ctx;
  const llama_vocab *vocab;
  wcommon_sampler *ctx_sampling = nullptr;
  llama_batch batch = llama_batch_init(512, 0, 1);
  llama_tokens tokens;
  int32_t seed = LLAMA_DEFAULT_SEED;
  int32_t last_logits_index = -1;
  std::vector<std::string> mmproj_paths;
  std::vector<std::vector<char>> media_bytes;
  std::vector<mtmd::bitmap> mtmd_bitmaps;
  mtmd_context_ptr mtmd_context;
  mtmd_input_chunks_ptr mtmd_input_chunks;
  std::vector<const mtmd_bitmap *> mtmd_bitmap_ptrs;
};

inline std::vector<char> convert_string_to_buf(std::string &input)
{
  std::vector<char> output;
  output.reserve(input.size());
  output.insert(output.end(), input.begin(), input.end());
  return output;
}

inline static ggml_type kv_cache_type_from_str(const std::string &s)
{
  if (s == "f32")
    return GGML_TYPE_F32;
  if (s == "f16")
    return GGML_TYPE_F16;
  if (s == "q8_0")
    return GGML_TYPE_Q8_0;
  if (s == "q4_0")
    return GGML_TYPE_Q4_0;
  if (s == "q4_1")
    return GGML_TYPE_Q4_1;
  if (s == "q5_0")
    return GGML_TYPE_Q5_0;
  if (s == "q5_1")
    return GGML_TYPE_Q5_1;
  throw std::runtime_error("Invalid cache type: " + s);
}

inline static enum llama_pooling_type pooling_type_from_str(const std::string &s)
{
  if (s == "LLAMA_POOLING_TYPE_UNSPECIFIED")
    return LLAMA_POOLING_TYPE_UNSPECIFIED;
  if (s == "LLAMA_POOLING_TYPE_NONE")
    return LLAMA_POOLING_TYPE_NONE;
  if (s == "LLAMA_POOLING_TYPE_MEAN")
    return LLAMA_POOLING_TYPE_MEAN;
  if (s == "LLAMA_POOLING_TYPE_CLS")
    return LLAMA_POOLING_TYPE_CLS;
  throw std::runtime_error("Invalid pooling type: " + s);
}

inline static llama_rope_scaling_type rope_scaling_type_from_str(const std::string &s)
{
  if (s == "LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED")
    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
  if (s == "LLAMA_ROPE_SCALING_TYPE_NONE")
    return LLAMA_ROPE_SCALING_TYPE_NONE;
  if (s == "LLAMA_ROPE_SCALING_TYPE_LINEAR")
    return LLAMA_ROPE_SCALING_TYPE_LINEAR;
  if (s == "LLAMA_ROPE_SCALING_TYPE_YARN")
    return LLAMA_ROPE_SCALING_TYPE_YARN;
  throw std::runtime_error("Invalid RoPE scaling type: " + s);
}

class app_exception : public std::exception
{
public:
  app_exception(const std::string &msg) throw() : message(msg) {}
  virtual ~app_exception() throw() {}
  const char *what() const throw() { return message.c_str(); }

private:
  std::string message;
};

void free_all(app_t &app)
{
  app.mtmd_input_chunks.reset();
  app.mtmd_bitmap_ptrs.clear();
  app.mtmd_bitmaps.clear();
  app.media_bytes.clear();
  app.mtmd_context.reset();
  app.mmproj_paths.clear();
  if (app.ctx != nullptr)
    llama_free(app.ctx);
  if (app.model != nullptr)
    llama_model_free(app.model);
  if (app.ctx_sampling != nullptr)
    wcommon_sampler_free(app.ctx_sampling);
}

struct kv_dump
{
  std::vector<std::string> keys;
  std::vector<std::string> vals;
};

kv_dump dump_metadata(app_t &app)
{
  kv_dump output;
  int count = llama_model_meta_count(app.model);
  std::string key;
  std::string val;
  std::vector<char> buf(1024);
  int res = 0;
  for (int i = 0; i < count; i++)
  {
    res = llama_model_meta_val_str_by_index(app.model, i, buf.data(), buf.size());
    if (res < 0)
      continue;
    if (res > buf.size())
    {
      buf.resize(res + 1);
      res = llama_model_meta_val_str_by_index(app.model, i, buf.data(), buf.size());
    }
    val = std::string(buf.data(), res);
    res = llama_model_meta_key_by_index(app.model, i, buf.data(), buf.size());
    if (res < 0)
      continue;
    if (res > buf.size())
    {
      buf.resize(res + 1);
      res = llama_model_meta_key_by_index(app.model, i, buf.data(), buf.size());
    }
    key = std::string(buf.data(), res);
    output.keys.push_back(std::move(key));
    output.vals.push_back(std::move(val));
  }
  return output;
}

//////////////////////////////////////////
//////////////////////////////////////////
//////////////////////////////////////////

glue_msg_load_res action_load(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_load_req);
  free_all(app);
  std::vector<std::string> &model_paths = req.model_paths.arr;
  bool n_ctx_auto = req.n_ctx_auto.value;

  auto mparams = llama_model_default_params();
  if (req.use_mmap.not_null())
    mparams.use_mmap = req.use_mmap.value;
  if (req.use_mlock.not_null())
    mparams.use_mlock = req.use_mlock.value;
  if (req.use_webgpu.value) {
    app.device = ggml_backend_dev_by_name("WebGPU");
  } else {
    app.device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  }
  if (!app.device) {
    throw app_exception(
      req.use_webgpu.value
        ? "WebGPU backend not available"
        : "CPU backend not available"
    );
  }
  ggml_backend_dev_t devices[] = { app.device, nullptr };
  mparams.devices = devices;

  if (req.n_gpu_layers.not_null())
    mparams.n_gpu_layers = req.n_gpu_layers.value;

  auto cparams = llama_context_default_params();
  app.seed = req.seed.value;
  cparams.n_ctx = req.n_ctx.value;
  cparams.n_threads = req.n_threads.value;
  cparams.n_threads_batch = cparams.n_threads;
  cparams.no_perf = req.no_perf.value;
  if (req.embeddings.not_null())
    cparams.embeddings = req.embeddings.value;
  if (req.offload_kqv.not_null())
    cparams.offload_kqv = req.offload_kqv.value;
  if (req.n_batch.not_null())
    cparams.n_batch = req.n_batch.value;
  if (req.n_seq_max.not_null())
    cparams.n_seq_max = req.n_seq_max.value;
  if (req.pooling_type.not_null())
    cparams.pooling_type = pooling_type_from_str(req.pooling_type.value);
  // context extending: https://github.com/ggerganov/llama.cpp/pull/2054
  if (req.rope_scaling_type.not_null())
    cparams.rope_scaling_type = rope_scaling_type_from_str(req.rope_scaling_type.value);
  if (req.rope_freq_base.not_null())
    cparams.rope_freq_base = req.rope_freq_base.value;
  if (req.rope_freq_scale.not_null())
    cparams.rope_freq_scale = req.rope_freq_scale.value;
  if (req.yarn_ext_factor.not_null())
    cparams.yarn_ext_factor = req.yarn_ext_factor.value;
  if (req.yarn_attn_factor.not_null())
    cparams.yarn_attn_factor = req.yarn_attn_factor.value;
  if (req.yarn_beta_fast.not_null())
    cparams.yarn_beta_fast = req.yarn_beta_fast.value;
  if (req.yarn_beta_slow.not_null())
    cparams.yarn_beta_slow = req.yarn_beta_slow.value;
  if (req.yarn_orig_ctx.not_null())
    cparams.yarn_orig_ctx = req.yarn_orig_ctx.value;
  // optimizations
  if (req.cache_type_k.not_null())
    cparams.type_k = kv_cache_type_from_str(req.cache_type_k.value);
  if (req.cache_type_v.not_null())
    cparams.type_v = kv_cache_type_from_str(req.cache_type_v.value);
  if (req.swa_full.not_null())
    cparams.swa_full = req.swa_full.value;
  if (req.flash_attn.not_null())
    cparams.flash_attn_type = req.flash_attn.value ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;

  // init threadpool
  ggml_threadpool_params_default(cparams.n_threads);

  // prepare model paths
  std::vector<const char *> model_paths_ptrs;
  for (auto &path : model_paths)
  {
    model_paths_ptrs.push_back(path.c_str());
  }

  // load model
  app.model = llama_model_load_from_splits(
      model_paths_ptrs.data(), model_paths_ptrs.size(), mparams);
  if (app.model == nullptr)
  {
    free_all(app);
    throw app_exception("Error while loading model");
  }
  app.vocab = llama_model_get_vocab(app.model);
  for (; cparams.n_ctx > 0; cparams.n_ctx -= 1024)
  {
    app.ctx = llama_init_from_model(app.model, cparams);
    if (app.ctx != nullptr)
    {
      break; // OK
    }
    if (!n_ctx_auto)
    {
      free_all(app);
      throw app_exception("Error while creating llama_context model");
    }
    else
    {
      std::cerr << "llama_context == nullptr, Retrying with n_ctx = " << cparams.n_ctx;
      continue;
    }
  }
  if (cparams.n_ctx < 0)
  {
    free_all(app);
    throw app_exception("Out of memory, cannot create llama_context model");
  }
  llama_batch_free(app.batch);
  app.batch = llama_batch_init(cparams.n_batch, 0, 1);
  auto decoder_start_token = llama_model_decoder_start_token(app.model);
  if (decoder_start_token < 0)
  {
    decoder_start_token = llama_vocab_bos(app.vocab);
  }
  int n_vocab = llama_vocab_n_tokens(app.vocab);
  llama_tokens list_tokens_eog;
  for (int i = 0; i < n_vocab; i++)
  {
    if (llama_vocab_is_eog(app.vocab, i))
    {
      list_tokens_eog.push_back(i);
    }
  }
  kv_dump metadata = dump_metadata(app);

  glue_msg_load_res res;
  res.success.value = true;
  res.n_ctx.value = cparams.n_ctx;
  res.n_batch.value = llama_n_batch(app.ctx);
  res.n_ubatch.value = llama_n_ubatch(app.ctx);
  res.n_vocab.value = n_vocab;
  res.n_ctx_train.value = llama_model_n_ctx_train(app.model);
  res.n_embd.value = llama_model_n_embd(app.model);
  res.n_layer.value = llama_model_n_layer(app.model);
  res.metadata_key.arr = metadata.keys;
  res.metadata_val.arr = metadata.vals;
  res.token_bos.value = llama_vocab_bos(app.vocab);
  res.token_eos.value = llama_vocab_eos(app.vocab);
  res.token_eot.value = llama_vocab_eot(app.vocab);
  res.list_tokens_eog.arr = std::move(list_tokens_eog);
  res.add_bos_token.value = llama_vocab_get_add_bos(app.vocab) == 1;
  res.add_eos_token.value = llama_vocab_get_add_eos(app.vocab) == 1;
  res.has_encoder.value = llama_model_has_encoder(app.model);
  res.token_decoder_start.value = llama_model_decoder_start_token(app.model);
  return res;
}

// set various options at runtime (after loading model)
glue_msg_set_options_res action_set_options(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_set_options_req);
  if (req.embeddings.value)
  {
    llama_set_embeddings(app.ctx, true);
    llama_set_causal_attn(app.ctx, false);
  }
  else
  {
    llama_set_embeddings(app.ctx, false);
    llama_set_causal_attn(app.ctx, true);
  }
  glue_msg_set_options_res res;
  res.success.value = true;
  return res;
}

// init (or re-init) sampling context
glue_msg_sampling_init_res action_sampling_init(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_sampling_init_req);
  // sampling
  wcommon_params_sampling sparams;
  sparams.seed = app.seed;
  if (sparams.seed == LLAMA_DEFAULT_SEED)
    sparams.seed = time(NULL);

  if (req.mirostat.not_null())
    sparams.mirostat = req.mirostat.value;
  if (req.mirostat_tau.not_null())
    sparams.mirostat_tau = req.mirostat_tau.value;
  if (req.mirostat_eta.not_null())
    sparams.mirostat_eta = req.mirostat_eta.value;
  if (req.temp.not_null())
    sparams.temp = req.temp.value;
  if (req.top_p.not_null())
    sparams.top_p = req.top_p.value;
  if (req.top_k.not_null())
    sparams.top_k = req.top_k.value;
  if (req.penalty_last_n.not_null())
    sparams.penalty_last_n = req.penalty_last_n.value;
  if (req.penalty_repeat.not_null())
    sparams.penalty_repeat = req.penalty_repeat.value;
  if (req.penalty_freq.not_null())
    sparams.penalty_freq = req.penalty_freq.value;
  if (req.penalty_present.not_null())
    sparams.penalty_present = req.penalty_present.value;
  if (req.dynatemp_range.not_null())
    sparams.dynatemp_range = req.dynatemp_range.value;
  if (req.dynatemp_exponent.not_null())
    sparams.dynatemp_exponent = req.dynatemp_exponent.value;
  // if (req.samplers_sequence.not_null())
  //   sparams.samplers_sequence = req.samplers_sequence.value;
  if (req.grammar.not_null())
    sparams.grammar = req.grammar.value;
  if (req.n_prev.not_null())
    sparams.n_prev = req.n_prev.value;
  if (req.n_probs.not_null())
    sparams.n_probs = req.n_probs.value;
  if (req.min_p.not_null())
    sparams.min_p = req.min_p.value;
  if (req.typical_p.not_null())
    sparams.typ_p = req.typical_p.value; // for compat
  if (req.typ_p.not_null())
    sparams.typ_p = req.typ_p.value;
  // logit bias
  if (req.logit_bias_vals.not_null() && req.logit_bias_toks.not_null())
  {
    std::vector<llama_token> tokens = std::move(req.logit_bias_toks.arr);
    std::vector<float> &bias = req.logit_bias_vals.arr;
    for (size_t i = 0; i < tokens.size(); i++)
    {
      sparams.logit_bias.push_back({tokens[i], bias[i]});
    }
  }
  // maybe free before creating a new one
  if (app.ctx_sampling != nullptr)
  {
    wcommon_sampler_free(app.ctx_sampling);
  }
  app.ctx_sampling = wcommon_sampler_init(app.model, sparams);
  if (req.tokens.not_null())
  {
    for (auto id : req.tokens.arr)
    {
      wcommon_sampler_accept(app.ctx_sampling, id, false);
    }
  }

  glue_msg_sampling_init_res res;
  res.success.value = true;
  return res;
}

// get map token ID to vocab (be careful, it is slow!)
glue_msg_get_vocab_res action_get_vocab(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_get_vocab_req);
  int32_t max_tokens = llama_vocab_n_tokens(app.vocab);
  std::vector<std::vector<char>> vocab;
  vocab.resize(max_tokens);
  for (int32_t id = 0; id < max_tokens; id++)
  {
    std::string token_as_str = wcommon_token_to_piece(app.ctx, id);
    vocab.emplace_back(convert_string_to_buf(token_as_str));
  }

  glue_msg_get_vocab_res res;
  res.success.value = true;
  res.vocab.arr = vocab;
  return res;
}

// lookup single token (also be able to check if it exists or not)
glue_msg_lookup_token_res action_lookup_token(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_lookup_token_req);
  std::string &piece = req.piece.value;
  int32_t max_tokens = llama_vocab_n_tokens(app.vocab);
  glue_msg_lookup_token_res res;
  for (int32_t id = 0; id < max_tokens; id++)
  {
    std::string token_as_str = wcommon_token_to_piece(app.ctx, id);
    if (token_as_str == piece)
    {
      res.success.value = true;
      res.token.value = id;
      return res;
    }
  }
  // not found
  res.success.value = false;
  return res;
}

// tokenize an input string
glue_msg_tokenize_res action_tokenize(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_tokenize_req);
  std::string &text = req.text.value;
  bool special = req.special.value;
  llama_tokens tokens_list = wcommon_tokenize(app.vocab, text, false, special);

  glue_msg_tokenize_res res;
  res.success.value = true;
  res.tokens.arr = std::move(tokens_list);
  return res;
}

// detokenize a list of tokens
glue_msg_detokenize_res action_detokenize(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_detokenize_req);
  llama_tokens tokens = std::move(req.tokens.arr);
  std::stringstream output;
  for (auto id : tokens)
  {
    output << wcommon_token_to_piece(app.ctx, id);
  }
  std::string parsed_str = output.str();

  glue_msg_detokenize_res res;
  res.success.value = true;
  res.buffer.buf = convert_string_to_buf(parsed_str);
  return res;
}

// decode an array of tokens
glue_msg_decode_res action_decode(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_decode_req);
  llama_tokens tokens_list = std::move(req.tokens.arr);
  bool skip_logits = req.skip_logits.value;
  size_t i = 0;
  wcommon_batch_clear(app.batch);
  for (auto id : tokens_list)
  {
    bool grp_attn_enabled = false; // TODO: maybe remove grp_attn
    int32_t n_past = app.tokens.size();
    wcommon_batch_add(app.batch, id, n_past, {0}, false);
    app.tokens.push_back(id);
    i++;
  }
  // llama_decode will output logits only for the last token of the prompt
  if (!skip_logits)
  {
    app.batch.logits[app.batch.n_tokens - 1] = true;
  }
  glue_msg_decode_res res;
  if (llama_decode(app.ctx, app.batch) != 0)
  {
    app.last_logits_index = -1;
    res.success.value = false;
    res.message.value = "llama_decode failed, maybe n_batch is too small?";
    res.n_past.value = app.tokens.size();
  }
  else
  {
    app.last_logits_index = !skip_logits && app.batch.n_tokens > 0 ? app.batch.n_tokens - 1 : -1;
    res.success.value = true;
    res.n_past.value = app.tokens.size();
  }
  return res;
}

// encode an array of tokens
glue_msg_encode_res action_encode(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_encode_req);
  llama_tokens tokens_list = std::move(req.tokens.arr);
  if (!llama_model_has_encoder(app.model))
  {
    glue_msg_encode_res res;
    res.success.value = false;
    res.message.value = "this model does not have an encoder";
    return res;
  }
  size_t n_past = 0;
  wcommon_batch_clear(app.batch);
  for (auto id : tokens_list)
  {
    wcommon_batch_add(app.batch, id, n_past, {0}, false);
    n_past++;
  }
  glue_msg_encode_res res;
  if (llama_encode(app.ctx, app.batch) != 0)
  {
    res.success.value = false;
    res.message.value = "llama_encode failed, maybe n_batch is too small?";
    res.n_past.value = n_past;
  }
  else
  {
    res.success.value = true;
    res.n_past.value = n_past;
  }
  return res;
}

// decode the current logits and sample the new token
glue_msg_sampling_sample_res action_sampling_sample(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_sampling_sample_req);
  int32_t idx = app.last_logits_index;
  const llama_token new_token_id = wcommon_sampler_sample(app.ctx_sampling, app.ctx, idx, false);
  std::string piece = wcommon_token_to_piece(app.ctx, new_token_id);

  glue_msg_sampling_sample_res res;
  res.success.value = true;
  res.piece.buf = convert_string_to_buf(piece);
  res.token.value = new_token_id;
  return res;
}

// accept this token
glue_msg_sampling_accept_res action_sampling_accept(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_sampling_accept_req);
  llama_tokens tokens_list = std::move(req.tokens.arr);
  for (auto id : tokens_list)
  {
    wcommon_sampler_accept(app.ctx_sampling, id, false);
  }

  glue_msg_sampling_accept_res res;
  res.success.value = true;
  return res;
}

// get softmax-ed probability of logits, can be used for custom sampling. The output is always sorted
glue_msg_get_logits_res action_get_logits(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_get_logits_req);
  int top_k = req.top_k.value; // if is -1, we take all logits (will be slow!)
  int32_t idx = app.last_logits_index;
  float *logits = llama_get_logits_ith(app.ctx, idx);
  int32_t n_vocab = llama_vocab_n_tokens(app.vocab);
  auto sort_fn = [](llama_token_data &a, llama_token_data &b) -> bool
  {
    return b.logit < a.logit;
  };
  // get all candidates and sort
  std::vector<llama_token_data> candidates;
  candidates.reserve(n_vocab);
  float sum = 0.0f; // for softmax
  for (llama_token token_id = 0; token_id < n_vocab; token_id++)
  {
    float exp_val = exp(logits[token_id]);
    candidates.emplace_back(llama_token_data{token_id, logits[token_id], exp_val});
    sum += exp_val;
  }
  for (auto &c : candidates)
  {
    c.p /= sum; // calculate softmax
  }
  std::sort(candidates.begin(), candidates.end(), sort_fn);
  if (top_k >= 0)
  {
    candidates.erase(candidates.begin() + top_k, candidates.end());
  }
  // convert response to json
  std::vector<int32_t> output_tokens;
  std::vector<float> output_probs;
  output_tokens.reserve(candidates.size());
  output_probs.reserve(candidates.size());
  for (auto &c : candidates)
  {
    output_tokens.push_back(c.id);
    output_probs.push_back(c.p);
  }

  glue_msg_get_logits_res res;
  res.success.value = true;
  res.tokens.arr = std::move(output_tokens);
  res.probs.arr = std::move(output_probs);
  return res;
}

// get embeddings, this will call action_decode internally
glue_msg_get_embeddings_res action_embeddings(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_get_embeddings_req);
  auto &tokens_list = req.tokens.arr;
  // allocate output
  const int n_embd = llama_model_n_embd(app.model);
  std::vector<float> embeddings(n_embd, 0); // single seq
  float *out = embeddings.data();
  // decode
  glue_msg_get_embeddings_res res;
  glue_msg_decode_req decode_req;
  decode_req.tokens.arr = std::move(tokens_list);
  decode_req.skip_logits.value = false;
  glue_outbuf decode_req_buf;
  decode_req.handler.serialize(decode_req_buf);
  auto decode_res = action_decode(app, decode_req_buf.data.data());
  if (decode_res.success.value == false)
  {
    res.success.value = false;
    res.message.value = std::move(decode_res.message.value);
    return res;
  }
  int32_t idx = app.batch.n_tokens - 1;
  const float *embd = llama_get_embeddings_seq(app.ctx, 0);
  if (embd == NULL)
  {
    embd = llama_get_embeddings_ith(app.ctx, idx);
    if (embd == NULL)
    {
      // fprintf(stderr, "%s: failed to get embeddings for token %d\n", __func__, idx);
      res.success.value = false;
      res.message.value = "failed to get embeddings";
      return res;
    }
  }
  wcommon_embd_normalize(embd, out, n_embd, 2);

  res.success.value = true;
  res.embeddings.arr = std::move(embeddings);
  return res;
}

// remove tokens in kv, for context-shifting
glue_msg_get_kv_remove_res action_kv_remove(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_get_kv_remove_req);
  const int n_keep = req.n_keep.value;
  const int n_discard = req.n_discard.value;
  auto * mem = llama_get_memory(app.ctx);

  glue_msg_get_kv_remove_res res;
  bool & success = res.success.value;
  success = false;
  res.n_past.value = app.tokens.size();

  llama_pos pos_min = llama_memory_seq_pos_min(mem, 0);
  if (pos_min > 0) {
    // TODO: rm tokens from SWA is currently unsupported
    success = false;
    return res;
  }

  if (n_discard > 0)
  {
    // TODO: this code branch is kinda broken, to be fixed later
    const int n_past = app.tokens.size();
    success = llama_memory_seq_rm(mem, 0, n_keep, n_keep + n_discard);
    if (!success)
    {
      return res;
    }
    llama_memory_seq_add(mem, 0, n_keep + n_discard, n_past, -n_discard);
    app.tokens.erase(
        app.tokens.begin() + n_keep,
        app.tokens.begin() + n_keep + n_discard);
  }
  else if (n_discard < 0)
  {
    if (n_keep == 0)
    {
      llama_memory_clear(mem, true);
    }
    else
    {
      success = llama_memory_seq_rm(mem, 0, n_keep, -1);
      if (!success)
      {
        return res;
      }
      app.tokens.erase(
          app.tokens.begin() + n_keep,
          app.tokens.end());
    }
  }

  return res;
}

// clear all tokens in kv
glue_msg_get_kv_clear_res action_kv_clear(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_get_kv_clear_req);
  auto * mem = llama_get_memory(app.ctx);
  llama_memory_clear(mem, true);
  app.tokens.clear();

  glue_msg_get_kv_clear_res res;
  res.success.value = true;
  res.n_past.value = app.tokens.size();
  return res;
}

/*
// save current session
json action_session_save(app_t &app, json &body)
{
  std::string session_path = body["session_path"];
  llama_tokens dummy;
  if (!llama_state_seq_save_file(
          app.ctx,
          session_path.c_str(),
          0,            // seq_id
          dummy.data(), // tokens
          dummy.size()  // n_token_count
          ))
  {
    return json{{"error", "action_session_save failed"}};
  }
  return json{
      {"success", true},
      {"tokens", app.tokens},
  };
}

// load a session from disk
json action_session_load(app_t &app, json &body)
{
  std::string session_path = body["session_path"];
  llama_tokens saved_tokens = body["tokens"];
  auto n_ctx = llama_n_ctx(app.ctx);
  size_t n_token_count_out = 0;
  llama_tokens dummy;
  if (!llama_state_seq_load_file(
          app.ctx,
          session_path.c_str(),
          0,                 // dest_seq_id
          dummy.data(),      // tokens_out
          dummy.capacity(),  // n_token_capacity
          &n_token_count_out // n_token_count_out
          ))
  {
    return json{{"error", "llama_load_session_file failed"}};
  }
  // load tokens
  app.tokens.clear();
  app.tokens.reserve(saved_tokens.size());
  for (auto id : saved_tokens)
  {
    app.tokens.push_back(id);
  }
  return json{{"success", true}};
}
*/

// get the current status
glue_msg_status_res action_current_status(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_status_req);
  glue_msg_status_res res;
  res.success.value = true;
  res.tokens.arr = app.tokens; // copy
  return res;
}

glue_msg_perf_context_res action_perf_context(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_perf_context_req);
  glue_msg_perf_context_res res;
  if (app.ctx == nullptr)
  {
    res.success.value = false;
    return res;
  }
  const llama_perf_context_data data = llama_perf_context(app.ctx);
  res.success.value = true;
  res.t_start_ms.value = data.t_start_ms;
  res.t_load_ms.value = data.t_load_ms;
  res.t_p_eval_ms.value = data.t_p_eval_ms;
  res.t_eval_ms.value = data.t_eval_ms;
  res.n_p_eval.value = data.n_p_eval;
  res.n_eval.value = data.n_eval;
  res.n_reused.value = data.n_reused;
  return res;
}

glue_msg_perf_reset_res action_perf_reset(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_perf_reset_req);
  glue_msg_perf_reset_res res;
  if (app.ctx == nullptr)
  {
    res.success.value = false;
    return res;
  }
  llama_perf_context_reset(app.ctx);
  res.success.value = true;
  return res;
}

//
// benchmark & perplexity
//

glue_msg_test_benchmark_res action_test_benchmark(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_test_benchmark_req);
  std::string type = req.type.value;   // "pp" (prompt proc) or "tg" (tok gen)
  int n_samples = req.n_samples.value; // n_batch in pp and n_predict in pg

  llama_memory_clear(llama_get_memory(app.ctx), true);
  int n_vocab = llama_vocab_n_tokens(app.vocab);
  int64_t t_start = ggml_time_ms();

  if (type == "pp")
  {
    llama_batch batch = llama_batch_init(n_samples, 0, 1);
    for (int i = 0; i < n_samples; i++)
    {
      wcommon_batch_add(batch, i % n_vocab, i, {0}, i == n_samples - 1);
    }
    int ret = llama_decode(app.ctx, batch);
    llama_batch_free(batch);
    if (ret != 0)
    {
      glue_msg_test_benchmark_res res;
      res.success.value = false;
      res.message.value = "llama_decode failed with status = " + std::to_string(ret);
      return res;
    }
  }
  else if (type == "tg")
  {
    llama_batch batch = llama_batch_init(1, 0, 1);
    for (int i = 0; i < n_samples; i++)
    {
      wcommon_batch_clear(batch);
      wcommon_batch_add(batch, i % n_vocab, i, {0}, true);
      int ret = llama_decode(app.ctx, batch);
      if (ret != 0)
      {
        glue_msg_test_benchmark_res res;
        res.success.value = false;
        res.message.value = "llama_decode failed with status = " + std::to_string(ret);
        return res;
      }
    }
    llama_batch_free(batch);
  }
  else
  {
    glue_msg_test_benchmark_res res;
    res.success.value = false;
    res.message.value = "unknown type: " + type;
    return res;
  }

  int64_t t_end = ggml_time_ms();
  glue_msg_test_benchmark_res res;
  res.success.value = true;
  res.t_ms.value = t_end - t_start;
  return res;
}

glue_msg_test_perplexity_res action_test_perplexity(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_test_perplexity_req);
  llama_tokens input = std::move(req.tokens.arr);
  const size_t n = input.size();

  int64_t t_start = ggml_time_ms();

  if (n < 2)
  {
    glue_msg_test_perplexity_res res;
    res.success.value = false;
    res.message.value = "Input must contain at least two tokens";
    return res;
  }

  // Clear existing context to start fresh
  llama_memory_clear(llama_get_memory(app.ctx), true);
  app.tokens.clear();

  const int32_t n_vocab = llama_vocab_n_tokens(app.vocab);
  double nll = 0.0;

  static auto log_softmax = [](int n_vocab, const float *logits, int tok) -> double
  {
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i)
    {
      max_logit = std::max(max_logit, logits[i]);
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i)
    {
      sum_exp += expf(logits[i] - max_logit);
    }
    return logits[tok] - max_logit - log(sum_exp);
  };

  for (size_t i = 0; i < n - 1; ++i)
  {
    // Prepare batch with current token (input[i])
    wcommon_batch_clear(app.batch);
    wcommon_batch_add(app.batch, input[i], i, {0}, true); // Enable logits for this token

    if (llama_decode(app.ctx, app.batch) != 0)
    {
      glue_msg_test_perplexity_res res;
      res.success.value = false;
      res.message.value = "llama_decode failed at position " + std::to_string(i);
      return res;
    }

    float *logits = llama_get_logits_ith(app.ctx, 0);

    // Get true next token (input[i+1])
    const int32_t true_token = input[i + 1];

    nll += -log_softmax(n_vocab, logits, true_token);
  }

  // Calculate final metrics
  const double cross_entropy = nll / (n - 1);
  const double ppl = std::exp(cross_entropy);

  int64_t t_end = ggml_time_ms();

  glue_msg_test_perplexity_res res;
  res.success.value = true;
  res.ppl.value = ppl;
  res.nll.value = nll;
  res.cross_entropy.value = cross_entropy;
  res.n_tokens.value = n - 1;
  res.t_ms.value = t_end - t_start;
  return res;
}

glue_msg_chat_format_res action_chat_format(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_chat_format_req);
  std::string tmpl = req.tmpl.not_null() ? req.tmpl.value : "";
  bool add_ass = req.add_ass.not_null() ? req.add_ass.value : false;
  std::vector<std::string> &roles = req.roles.arr;
  std::vector<std::string> &contents = req.contents.arr;
  std::vector<wcommon_chat_msg> chat;
  for (size_t i = 0; i < roles.size(); i++)
  {
    chat.push_back({roles[i], contents[i]});
  }
  try
  {
    std::string formatted_chat = wcommon_chat_apply_template(app.model, tmpl, chat, add_ass);
    glue_msg_chat_format_res res;
    res.success.value = true;
    res.formatted_chat.value = formatted_chat;
    return res;
  }
  catch (const std::exception &e)
  {
    glue_msg_chat_format_res res;
    res.success.value = true;
    res.message.value = std::string(e.what());
    return res;
  }
}

inline static void mtmd_clear_loaded_media(app_t &app)
{
  app.mtmd_bitmap_ptrs.clear();
  app.mtmd_bitmaps.clear();
  app.media_bytes.clear();
}

inline static void mtmd_resize_token_ledger(app_t &app, llama_pos new_n_past)
{
  if (new_n_past < 0)
  {
    return;
  }
  app.tokens.resize(static_cast<size_t>(new_n_past), LLAMA_TOKEN_NULL);
}

inline static int32_t mtmd_to_i32(size_t value)
{
  const size_t max_i32 = static_cast<size_t>(std::numeric_limits<int32_t>::max());
  return value > max_i32 ? std::numeric_limits<int32_t>::max() : static_cast<int32_t>(value);
}

inline static int32_t mtmd_to_i32(uint32_t value)
{
  return value > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())
             ? std::numeric_limits<int32_t>::max()
             : static_cast<int32_t>(value);
}

inline static int32_t mtmd_tensor_dim_to_i32(int64_t value)
{
  if (value < 0)
    return -1;
  return value > static_cast<int64_t>(std::numeric_limits<int32_t>::max())
             ? std::numeric_limits<int32_t>::max()
             : static_cast<int32_t>(value);
}

inline static void mtmd_native_breadcrumb(const char *message)
{
  wllama_native_breadcrumb(message);
}

inline static std::string mtmd_bounded_exception_message(const std::string &message)
{
  constexpr size_t max_raw_exception_message_bytes = 384;
  if (message.size() <= max_raw_exception_message_bytes)
  {
    return message;
  }
  std::ostringstream ss;
  ss << message.substr(0, max_raw_exception_message_bytes)
     << "...<truncated raw_exception_message_bytes=" << message.size()
     << " max_raw_exception_message_bytes=" << max_raw_exception_message_bytes << ">";
  return ss.str();
}

inline static std::string mtmd_u64_to_string(uint64_t value)
{
  return std::to_string(value);
}

inline static int32_t mtmd_u64_to_i32(uint64_t value)
{
  return value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())
             ? std::numeric_limits<int32_t>::max()
             : static_cast<int32_t>(value);
}

inline static void mtmd_fill_media_decode_diagnostics(
    glue_msg_load_media_bytes_res &res,
    const mtmd_helper_media_decode_diagnostics &diagnostics)
{
  res.media_byte_length.value = mtmd_u64_to_i32(diagnostics.input_byte_length);
  res.detected_media_format.value = mtmd_helper_media_format_name(diagnostics.detected_media_format);
  res.audio_detected.value = diagnostics.detected_audio;
  res.audio_target_sample_rate.value = diagnostics.target_sample_rate;
  res.audio_output_channels.value = diagnostics.output_channels;
  res.audio_decoder_init_result.value = diagnostics.decoder_init_result;
  res.audio_decoder_init_result_description.value =
      mtmd_helper_audio_result_description(diagnostics.decoder_init_result);
  res.audio_decoder_length_result.value = diagnostics.decoder_length_result;
  res.audio_decoder_length_result_description.value =
      mtmd_helper_audio_result_description(diagnostics.decoder_length_result);
  res.audio_decoder_length_available.value = diagnostics.decoder_length_available;
  res.audio_decoder_frame_count.value = mtmd_u64_to_string(diagnostics.decoder_frame_count);
  res.audio_decoder_length_unreliable.value = diagnostics.decoder_length_unreliable;
  res.audio_planned_pcm_samples.value = mtmd_u64_to_string(diagnostics.planned_pcm_samples);
  res.audio_planned_pcm_allocation_bytes.value =
      mtmd_u64_to_string(diagnostics.planned_pcm_allocation_bytes);
  res.audio_actual_pcm_allocation_bytes.value =
      mtmd_u64_to_string(diagnostics.actual_pcm_allocation_bytes);
  res.audio_max_pcm_allocation_bytes.value =
      mtmd_u64_to_string(diagnostics.max_pcm_allocation_bytes);
  res.audio_allocation_exceeds_limit.value = diagnostics.allocation_exceeds_limit;
  res.audio_chunked_decode_used.value = diagnostics.chunked_decode_used;
  res.audio_read_reached_end.value = diagnostics.read_reached_end;
  res.audio_decoder_read_result.value = diagnostics.decoder_read_result;
  res.audio_decoder_read_result_description.value =
      mtmd_helper_audio_result_description(diagnostics.decoder_read_result);
  res.audio_frames_read.value = mtmd_u64_to_string(diagnostics.frames_read);
  res.audio_pcm_resize_succeeded.value = diagnostics.pcm_resize_succeeded;
  res.audio_bitmap_init_succeeded.value = diagnostics.bitmap_init_succeeded;
  res.audio_decode_failure_stage.value =
      mtmd_helper_media_decode_stage_name(diagnostics.failure_stage);
}

inline static std::string mtmd_audio_decode_failure_message(
    const mtmd_helper_media_decode_diagnostics &diagnostics)
{
  std::ostringstream ss;
  ss << "audio decode failed at " << mtmd_helper_media_decode_stage_name(diagnostics.failure_stage)
     << " (format=" << mtmd_helper_media_format_name(diagnostics.detected_media_format)
     << ", input_byte_length=" << diagnostics.input_byte_length
     << ", target_sample_rate=" << diagnostics.target_sample_rate
     << ", decoder_init_result=" << diagnostics.decoder_init_result
     << " " << mtmd_helper_audio_result_description(diagnostics.decoder_init_result)
     << ", decoder_length_result=" << diagnostics.decoder_length_result
     << " " << mtmd_helper_audio_result_description(diagnostics.decoder_length_result)
     << ", decoder_frame_count=" << diagnostics.decoder_frame_count
     << ", planned_pcm_allocation_bytes=" << diagnostics.planned_pcm_allocation_bytes
     << ", max_pcm_allocation_bytes=" << diagnostics.max_pcm_allocation_bytes
     << ", decoder_read_result=" << diagnostics.decoder_read_result
     << " " << mtmd_helper_audio_result_description(diagnostics.decoder_read_result)
     << ", frames_read=" << diagnostics.frames_read
     << ", actual_pcm_allocation_bytes=" << diagnostics.actual_pcm_allocation_bytes
     << ", allocation_exceeds_limit=" << (diagnostics.allocation_exceeds_limit ? "true" : "false")
     << ", chunked_decode_used=" << (diagnostics.chunked_decode_used ? "true" : "false")
     << ", read_reached_end=" << (diagnostics.read_reached_end ? "true" : "false") << ")";
  return ss.str();
}

struct mtmd_projector_split_inventory
{
  int32_t tensor_count = -1;
  bool position_embeddings_present = false;
  std::string position_embedding_tensor_name;
  int32_t position_embedding_cols = -1;
  int32_t position_embedding_rows = -1;
  int32_t position_embedding_planes = -1;
  std::string error;
};

inline static mtmd_projector_split_inventory mtmd_read_projector_split_inventory(const std::string &path)
{
  mtmd_projector_split_inventory inventory;
  struct ggml_context *meta = nullptr;
  struct gguf_init_params params = {
      /*.no_alloc = */ true,
      /*.ctx      = */ &meta,
  };
  struct gguf_context *ctx_gguf = gguf_init_from_file(path.c_str(), params);
  if (ctx_gguf == nullptr)
  {
    inventory.error = "gguf_init_from_file_failed";
    if (meta != nullptr)
      ggml_free(meta);
    return inventory;
  }

  const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
  inventory.tensor_count = mtmd_tensor_dim_to_i32(n_tensors);
  for (int64_t i = 0; i < n_tensors; ++i)
  {
    const char *name = gguf_get_tensor_name(ctx_gguf, i);
    if (name == nullptr || std::string(name) != "v.position_embd.weight")
      continue;
    inventory.position_embeddings_present = true;
    inventory.position_embedding_tensor_name = name;
    ggml_tensor *tensor = meta == nullptr ? nullptr : ggml_get_tensor(meta, name);
    if (tensor != nullptr)
    {
      inventory.position_embedding_cols = mtmd_tensor_dim_to_i32(tensor->ne[0]);
      inventory.position_embedding_rows = mtmd_tensor_dim_to_i32(tensor->ne[1]);
      inventory.position_embedding_planes = mtmd_tensor_dim_to_i32(tensor->ne[2]);
    }
    break;
  }

  gguf_free(ctx_gguf);
  if (meta != nullptr)
    ggml_free(meta);
  return inventory;
}

inline static void mtmd_fill_projector_split_inventory(
    glue_msg_load_mmproj_or_mtmd_context_res &res,
    const std::vector<std::string> &paths)
{
  res.projector_split_paths.arr.clear();
  res.projector_split_tensor_counts.arr.clear();
  res.projector_split_position_embeddings_present.arr.clear();
  res.projector_split_position_embedding_tensor_names.arr.clear();
  res.projector_split_position_embedding_cols.arr.clear();
  res.projector_split_position_embedding_rows.arr.clear();
  res.projector_split_position_embedding_planes.arr.clear();
  res.projector_split_inventory_errors.arr.clear();

  for (const auto &path : paths)
  {
    const auto inventory = mtmd_read_projector_split_inventory(path);
    res.projector_split_paths.arr.push_back(path);
    res.projector_split_tensor_counts.arr.push_back(inventory.tensor_count);
    res.projector_split_position_embeddings_present.arr.push_back(inventory.position_embeddings_present);
    res.projector_split_position_embedding_tensor_names.arr.push_back(inventory.position_embedding_tensor_name);
    res.projector_split_position_embedding_cols.arr.push_back(inventory.position_embedding_cols);
    res.projector_split_position_embedding_rows.arr.push_back(inventory.position_embedding_rows);
    res.projector_split_position_embedding_planes.arr.push_back(inventory.position_embedding_planes);
    res.projector_split_inventory_errors.arr.push_back(inventory.error);
  }
}

inline static int32_t mtmd_chunk_type_id(enum mtmd_input_chunk_type type)
{
  switch (type)
  {
  case MTMD_INPUT_CHUNK_TYPE_TEXT:
    return 0;
  case MTMD_INPUT_CHUNK_TYPE_IMAGE:
    return 1;
  case MTMD_INPUT_CHUNK_TYPE_AUDIO:
    return 2;
  default:
    return -1;
  }
}

inline static int32_t mtmd_vision_position_blocker_priority(const char *kind)
{
  const std::string value = kind == nullptr ? "none" : kind;
  if (value == "invalid_patch_geometry")
    return 3;
  if (value == "missing_position_embeddings")
    return 2;
  if (value == "row_index_out_of_bounds")
    return 1;
  return 0;
}

inline static const char *mtmd_merge_vision_position_blocker_kind(const char *current, const char *next)
{
  return mtmd_vision_position_blocker_priority(next) > mtmd_vision_position_blocker_priority(current)
             ? next
             : current;
}

template <typename Res>
inline static void mtmd_reset_chunk_diagnostics(app_t &app, Res &res)
{
  res.vocab_size.value = app.vocab == nullptr ? -1 : llama_vocab_n_tokens(app.vocab);
  res.text_token_bounds_passed.value = true;
  res.text_token_out_of_range_total.value = 0;
  res.vision_projector_type.value = -1;
  res.vision_projector_type_name.value = "unknown";
  res.vision_position_blocker_kind.value = "none";
  res.vision_position_embeddings_available.value = false;
  res.vision_position_embedding_cols.value = -1;
  res.vision_position_embedding_rows.value = -1;
  res.vision_position_embedding_planes.value = -1;
  res.image_position_bounds_passed.value = true;
  res.image_position_out_of_range_total.value = 0;
  res.chunk_types.arr.clear();
  res.chunk_n_tokens.arr.clear();
  res.chunk_n_pos.arr.clear();
  res.chunk_text_token_min.arr.clear();
  res.chunk_text_token_max.arr.clear();
  res.chunk_text_first_negative_token.arr.clear();
  res.chunk_text_first_over_vocab_token.arr.clear();
  res.chunk_text_out_of_range_count.arr.clear();
  res.chunk_media_n_tokens.arr.clear();
  res.chunk_media_n_pos.arr.clear();
  res.chunk_image_nx.arr.clear();
  res.chunk_image_ny.arr.clear();
  res.chunk_media_decoder_pos_t_min.arr.clear();
  res.chunk_media_decoder_pos_t_max.arr.clear();
  res.chunk_media_decoder_pos_x_min.arr.clear();
  res.chunk_media_decoder_pos_x_max.arr.clear();
  res.chunk_media_decoder_pos_y_min.arr.clear();
  res.chunk_media_decoder_pos_y_max.arr.clear();
  res.chunk_media_decoder_pos_z_min.arr.clear();
  res.chunk_media_decoder_pos_z_max.arr.clear();
  res.chunk_vision_position_blocker_kind.arr.clear();
  res.chunk_vision_position_bounds_applicable.arr.clear();
  res.chunk_vision_position_bounds_passed.arr.clear();
  res.chunk_vision_position_out_of_range_count.arr.clear();
  res.chunk_vision_position_embedding_row_limit.arr.clear();
  res.chunk_vision_position_pos_x_max.arr.clear();
  res.chunk_vision_position_pos_y_max.arr.clear();
  res.chunk_vision_position_input_nx_max.arr.clear();
  res.chunk_vision_position_input_ny_max.arr.clear();
  res.chunk_vision_position_patch_size.arr.clear();
}

template <typename Res>
inline static int32_t mtmd_populate_chunk_diagnostics(app_t &app, Res &res)
{
  mtmd_reset_chunk_diagnostics(app, res);
  if (!app.mtmd_input_chunks)
  {
    return 0;
  }

  const int32_t vocab_size = res.vocab_size.value;
  const size_t n_chunks = mtmd_input_chunks_size(app.mtmd_input_chunks.get());
  int32_t total_out_of_range = 0;
  int32_t total_position_out_of_range = 0;
  const char *position_blocker_kind = "none";

  for (size_t i = 0; i < n_chunks; i++)
  {
    const mtmd_input_chunk *chunk = mtmd_input_chunks_get(app.mtmd_input_chunks.get(), i);
    const auto chunk_type = mtmd_input_chunk_get_type(chunk);
    const int32_t n_tokens = mtmd_to_i32(mtmd_input_chunk_get_n_tokens(chunk));
    const int32_t n_pos = static_cast<int32_t>(mtmd_input_chunk_get_n_pos(chunk));

    res.chunk_types.arr.push_back(mtmd_chunk_type_id(chunk_type));
    res.chunk_n_tokens.arr.push_back(n_tokens);
    res.chunk_n_pos.arr.push_back(n_pos);

    int32_t text_min = -1;
    int32_t text_max = -1;
    int32_t first_negative = -1;
    int32_t first_over_vocab = -1;
    int32_t out_of_range_count = 0;
    int32_t media_n_tokens = -1;
    int32_t media_n_pos = -1;
    int32_t image_nx = -1;
    int32_t image_ny = -1;
    int32_t decoder_t_min = -1;
    int32_t decoder_t_max = -1;
    int32_t decoder_x_min = -1;
    int32_t decoder_x_max = -1;
    int32_t decoder_y_min = -1;
    int32_t decoder_y_max = -1;
    int32_t decoder_z_min = -1;
    int32_t decoder_z_max = -1;
    mtmd_vision_position_bounds position_bounds = mtmd_vision_position_bounds_default();

    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT)
    {
      size_t n_text_tokens = 0;
      const llama_token *tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_text_tokens);
      for (size_t token_index = 0; tokens != nullptr && token_index < n_text_tokens; token_index++)
      {
        const int32_t token = static_cast<int32_t>(tokens[token_index]);
        if (token_index == 0 || token < text_min)
        {
          text_min = token;
        }
        if (token_index == 0 || token > text_max)
        {
          text_max = token;
        }
        const bool token_is_negative = token < 0;
        const bool token_is_over_vocab = vocab_size >= 0 && token >= vocab_size;
        if (token_is_negative && first_negative == -1)
        {
          first_negative = token;
        }
        if (token_is_over_vocab && first_over_vocab == -1)
        {
          first_over_vocab = token;
        }
        if (token_is_negative || token_is_over_vocab)
        {
          out_of_range_count++;
        }
      }
      total_out_of_range += out_of_range_count;
    }
    else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE || chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO)
    {
      media_n_tokens = n_tokens;
      media_n_pos = n_pos;
      if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE)
      {
        const mtmd_image_tokens *image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
        if (image_tokens != nullptr)
        {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
          image_nx = mtmd_to_i32(mtmd_image_tokens_get_nx(image_tokens));
          image_ny = mtmd_to_i32(mtmd_image_tokens_get_ny(image_tokens));
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
          position_bounds = mtmd_get_vision_position_bounds(app.mtmd_context.get(), image_tokens);
          res.vision_projector_type.value = position_bounds.projector_type;
          res.vision_projector_type_name.value = position_bounds.projector_type_name == nullptr
                                                   ? "unknown"
                                                   : position_bounds.projector_type_name;
          position_blocker_kind = mtmd_merge_vision_position_blocker_kind(
              position_blocker_kind,
              position_bounds.blocker_kind);
          res.vision_position_blocker_kind.value = position_blocker_kind;
          res.vision_position_embeddings_available.value = position_bounds.has_position_embeddings;
          res.vision_position_embedding_cols.value = position_bounds.position_embedding_cols;
          res.vision_position_embedding_rows.value = position_bounds.position_embedding_rows;
          res.vision_position_embedding_planes.value = position_bounds.position_embedding_planes;
          total_position_out_of_range += position_bounds.out_of_range_count;
          const size_t image_token_count = mtmd_image_tokens_get_n_tokens(image_tokens);
          for (size_t token_index = 0; token_index < image_token_count; token_index++)
          {
            const mtmd_decoder_pos pos = mtmd_image_tokens_get_decoder_pos(image_tokens, 0, token_index);
            const int32_t t = mtmd_to_i32(pos.t);
            const int32_t x = mtmd_to_i32(pos.x);
            const int32_t y = mtmd_to_i32(pos.y);
            const int32_t z = mtmd_to_i32(pos.z);
            if (token_index == 0 || t < decoder_t_min)
              decoder_t_min = t;
            if (token_index == 0 || t > decoder_t_max)
              decoder_t_max = t;
            if (token_index == 0 || x < decoder_x_min)
              decoder_x_min = x;
            if (token_index == 0 || x > decoder_x_max)
              decoder_x_max = x;
            if (token_index == 0 || y < decoder_y_min)
              decoder_y_min = y;
            if (token_index == 0 || y > decoder_y_max)
              decoder_y_max = y;
            if (token_index == 0 || z < decoder_z_min)
              decoder_z_min = z;
            if (token_index == 0 || z > decoder_z_max)
              decoder_z_max = z;
          }
        }
      }
    }

    res.chunk_text_token_min.arr.push_back(text_min);
    res.chunk_text_token_max.arr.push_back(text_max);
    res.chunk_text_first_negative_token.arr.push_back(first_negative);
    res.chunk_text_first_over_vocab_token.arr.push_back(first_over_vocab);
    res.chunk_text_out_of_range_count.arr.push_back(out_of_range_count);
    res.chunk_media_n_tokens.arr.push_back(media_n_tokens);
    res.chunk_media_n_pos.arr.push_back(media_n_pos);
    res.chunk_image_nx.arr.push_back(image_nx);
    res.chunk_image_ny.arr.push_back(image_ny);
    res.chunk_media_decoder_pos_t_min.arr.push_back(decoder_t_min);
    res.chunk_media_decoder_pos_t_max.arr.push_back(decoder_t_max);
    res.chunk_media_decoder_pos_x_min.arr.push_back(decoder_x_min);
    res.chunk_media_decoder_pos_x_max.arr.push_back(decoder_x_max);
    res.chunk_media_decoder_pos_y_min.arr.push_back(decoder_y_min);
    res.chunk_media_decoder_pos_y_max.arr.push_back(decoder_y_max);
    res.chunk_media_decoder_pos_z_min.arr.push_back(decoder_z_min);
    res.chunk_media_decoder_pos_z_max.arr.push_back(decoder_z_max);
    res.chunk_vision_position_blocker_kind.arr.push_back(
        position_bounds.blocker_kind == nullptr ? "none" : position_bounds.blocker_kind);
    res.chunk_vision_position_bounds_applicable.arr.push_back(position_bounds.bounds_applicable);
    res.chunk_vision_position_bounds_passed.arr.push_back(position_bounds.bounds_passed);
    res.chunk_vision_position_out_of_range_count.arr.push_back(position_bounds.out_of_range_count);
    res.chunk_vision_position_embedding_row_limit.arr.push_back(position_bounds.position_embedding_rows);
    res.chunk_vision_position_pos_x_max.arr.push_back(position_bounds.pos_x_max);
    res.chunk_vision_position_pos_y_max.arr.push_back(position_bounds.pos_y_max);
    res.chunk_vision_position_input_nx_max.arr.push_back(position_bounds.input_nx_max);
    res.chunk_vision_position_input_ny_max.arr.push_back(position_bounds.input_ny_max);
    res.chunk_vision_position_patch_size.arr.push_back(position_bounds.patch_size);
  }

  res.text_token_out_of_range_total.value = total_out_of_range;
  res.text_token_bounds_passed.value = total_out_of_range == 0;
  res.image_position_out_of_range_total.value = total_position_out_of_range;
  res.vision_position_blocker_kind.value = position_blocker_kind;
  res.image_position_bounds_passed.value =
      total_position_out_of_range == 0 && mtmd_vision_position_blocker_priority(position_blocker_kind) == 0;
  return total_out_of_range;
}

glue_msg_load_mmproj_or_mtmd_context_res action_load_mmproj_or_mtmd_context(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_load_mmproj_or_mtmd_context_req);
  glue_msg_load_mmproj_or_mtmd_context_res res;
  res.supports_vision.value = false;
  res.supports_image.value = false;
  res.supports_audio.value = false;
  res.audio_sample_rate.value = -1;
  res.media_marker.value = mtmd_default_marker();
  res.projector_split_loading_strategy.value = "not_started";
  res.projector_split_loading_status.value = "not_started";

  if (app.model == nullptr)
  {
    res.success.value = false;
    res.message.value = "load a text model before loading the MTMD context";
    return res;
  }
  if (req.mmproj_paths.arr.empty())
  {
    res.success.value = false;
    res.message.value = "mmproj_paths must contain at least one projector path";
    return res;
  }

  app.mtmd_input_chunks.reset();
  mtmd_clear_loaded_media(app);
  app.mtmd_context.reset();
  app.mmproj_paths = req.mmproj_paths.arr;
  mtmd_fill_projector_split_inventory(res, app.mmproj_paths);

  mtmd_context_params params = mtmd_context_params_default();
  params.use_gpu = req.use_gpu.value;
  params.print_timings = req.print_timings.value;
  params.warmup = req.warmup.value;
  params.load_audio = req.load_audio.value;
  if (req.n_threads.value > 0)
  {
    params.n_threads = req.n_threads.value;
  }
  params.media_marker = res.media_marker.value.c_str();

  const std::string &mmproj_path = app.mmproj_paths.front();
  const bool has_split_projector_paths = app.mmproj_paths.size() > 1;
  res.projector_split_loading_strategy.value =
      has_split_projector_paths ? "explicit_mmproj_path_list" : "single_projector_file";
  if (has_split_projector_paths)
  {
    std::vector<const char *> mmproj_path_ptrs;
    mmproj_path_ptrs.reserve(app.mmproj_paths.size());
    for (const auto &path : app.mmproj_paths)
    {
      mmproj_path_ptrs.push_back(path.c_str());
    }
    app.mtmd_context.reset(
        mtmd_init_from_files(mmproj_path_ptrs.data(), mmproj_path_ptrs.size(), app.model, params));
  }
  else
  {
    app.mtmd_context.reset(mtmd_init_from_file(mmproj_path.c_str(), app.model, params));
  }
  if (!app.mtmd_context)
  {
    res.success.value = false;
    res.projector_split_loading_status.value = "failed";
    res.message.value =
        (has_split_projector_paths
             ? "mtmd_init_from_files failed for explicit projector split path list starting at "
             : "mtmd_init_from_file failed for projector entry path ") +
        mmproj_path +
        (has_split_projector_paths
             ? "; all requested projector split paths were passed to mtmd_init_from_files"
             : "");
    return res;
  }

  res.supports_vision.value = mtmd_support_vision(app.mtmd_context.get());
  res.supports_image.value = res.supports_vision.value;
  res.supports_audio.value = mtmd_support_audio(app.mtmd_context.get());
  res.audio_sample_rate.value = mtmd_get_audio_sample_rate(app.mtmd_context.get());
  res.success.value = true;
  res.projector_split_loading_status.value =
      has_split_projector_paths ? "loaded_all_explicit_projector_paths" : "loaded_single_projector_path";
  res.message.value =
      "loaded MTMD context from " +
      (has_split_projector_paths ? std::to_string(app.mmproj_paths.size()) + " explicit projector split paths starting at " : std::string("projector entry path ")) +
      mmproj_path +
      (has_split_projector_paths
           ? "; all requested projector split paths were passed to mtmd_init_from_files"
           : "") +
      (req.load_audio.value ? "" : "; audio context disabled by load_audio=false");
  return res;
}

glue_msg_load_media_bytes_res action_load_media_bytes(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_load_media_bytes_req);
  glue_msg_load_media_bytes_res res;
  res.media_index.value = -1;
  res.n_media.value = app.media_bytes.size();
  mtmd_helper_media_decode_diagnostics decode_diagnostics;
  mtmd_helper_media_decode_diagnostics_reset(&decode_diagnostics);
  auto finish = [&]() {
    mtmd_fill_media_decode_diagnostics(res, decode_diagnostics);
    return res;
  };

  if (!app.mtmd_context)
  {
    res.success.value = false;
    res.message.value = "load an MTMD context before loading media bytes";
    return finish();
  }
  const size_t media_byte_length = req.media_bytes.buf.size();
  decode_diagnostics.input_byte_length = media_byte_length;
  if (req.media_bytes.buf.empty())
  {
    res.success.value = false;
    res.message.value = "media_bytes must not be empty";
    return finish();
  }

  const size_t existing_media_count = app.media_bytes.size();
  auto exception_message = [&](const char *type, const std::string &message) {
    std::ostringstream ss;
    ss << "mtmd_helper_bitmap_init_from_buf threw " << type;
    if (!message.empty())
    {
      ss << ": " << message;
    }
    ss << " (media_byte_length=" << media_byte_length
       << ", mtmd_context_loaded=true"
       << ", existing_media_count=" << existing_media_count << ")";
    return ss.str();
  };

  mtmd_bitmap *bitmap = nullptr;
  try
  {
    bitmap = mtmd_helper_bitmap_init_from_buf_with_diagnostics(
        app.mtmd_context.get(),
        reinterpret_cast<const unsigned char *>(req.media_bytes.buf.data()),
        media_byte_length,
        &decode_diagnostics);
  }
  catch (const std::exception &e)
  {
    res.success.value = false;
    res.message.value = exception_message("std::exception", e.what());
    return finish();
  }
  catch (const std::string &e)
  {
    res.success.value = false;
    res.message.value = exception_message("std::string", e);
    return finish();
  }
  catch (const char *e)
  {
    res.success.value = false;
    res.message.value = exception_message("const char *", e == nullptr ? "<null>" : std::string(e));
    return finish();
  }
  catch (...)
  {
    res.success.value = false;
    res.message.value = exception_message("unknown exception", "");
    return finish();
  }
  if (bitmap == nullptr)
  {
    res.success.value = false;
    res.message.value = decode_diagnostics.detected_audio
                            ? mtmd_audio_decode_failure_message(decode_diagnostics)
                            : "mtmd_helper_bitmap_init_from_buf failed to decode media bytes";
    return finish();
  }
  if (!req.media_id.value.empty())
  {
    mtmd_bitmap_set_id(bitmap, req.media_id.value.c_str());
  }

  app.media_bytes.push_back(std::move(req.media_bytes.buf));
  app.mtmd_bitmaps.emplace_back(bitmap);
  res.media_index.value = static_cast<int32_t>(app.mtmd_bitmaps.size() - 1);
  res.n_media.value = static_cast<int32_t>(app.mtmd_bitmaps.size());
  res.success.value = true;
  res.message.value = "loaded media bytes into MTMD bitmap";
  return finish();
}

glue_msg_mtmd_tokenize_res action_mtmd_tokenize(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_mtmd_tokenize_req);
  glue_msg_mtmd_tokenize_res res;
  res.n_chunks.value = 0;
  res.n_tokens.value = 0;
  res.n_pos.value = 0;
  mtmd_reset_chunk_diagnostics(app, res);

  if (!app.mtmd_context)
  {
    res.success.value = false;
    res.message.value = "load an MTMD context before tokenizing media";
    return res;
  }

  app.mtmd_input_chunks.reset(mtmd_input_chunks_init());
  if (!app.mtmd_input_chunks)
  {
    res.success.value = false;
    res.message.value = "mtmd_input_chunks_init failed";
    return res;
  }

  app.mtmd_bitmap_ptrs.clear();
  app.mtmd_bitmap_ptrs.reserve(app.mtmd_bitmaps.size());
  for (const auto &bitmap : app.mtmd_bitmaps)
  {
    app.mtmd_bitmap_ptrs.push_back(bitmap.ptr.get());
  }

  mtmd_input_text text;
  text.text = req.text.value.c_str();
  text.add_special = req.add_special.value;
  text.parse_special = req.parse_special.value;
  int32_t tokenized = -1;
  try
  {
    tokenized = mtmd_tokenize(
        app.mtmd_context.get(),
        app.mtmd_input_chunks.get(),
        &text,
        app.mtmd_bitmap_ptrs.data(),
        app.mtmd_bitmap_ptrs.size());
  }
  catch (const std::exception &e)
  {
    app.mtmd_bitmap_ptrs.clear();
    mtmd_clear_loaded_media(app);
    app.mtmd_input_chunks.reset();
    res.success.value = false;
    res.message.value = "mtmd_tokenize threw std::exception: " + mtmd_bounded_exception_message(e.what());
    return res;
  }
  catch (const std::string &e)
  {
    app.mtmd_bitmap_ptrs.clear();
    mtmd_clear_loaded_media(app);
    app.mtmd_input_chunks.reset();
    res.success.value = false;
    res.message.value = "mtmd_tokenize threw std::string: " + mtmd_bounded_exception_message(e);
    return res;
  }
  catch (const char *e)
  {
    app.mtmd_bitmap_ptrs.clear();
    mtmd_clear_loaded_media(app);
    app.mtmd_input_chunks.reset();
    res.success.value = false;
    res.message.value = "mtmd_tokenize threw const char *: " +
                        mtmd_bounded_exception_message(e == nullptr ? "<null>" : std::string(e));
    return res;
  }
  catch (...)
  {
    app.mtmd_bitmap_ptrs.clear();
    mtmd_clear_loaded_media(app);
    app.mtmd_input_chunks.reset();
    res.success.value = false;
    res.message.value = "mtmd_tokenize threw unknown exception";
    return res;
  }

  app.mtmd_bitmap_ptrs.clear();
  mtmd_clear_loaded_media(app);

  if (tokenized != 0)
  {
    app.mtmd_input_chunks.reset();
    res.success.value = false;
    res.message.value = "mtmd_tokenize failed with code " + std::to_string(tokenized);
    return res;
  }

  res.n_chunks.value = static_cast<int32_t>(mtmd_input_chunks_size(app.mtmd_input_chunks.get()));
  res.n_tokens.value = static_cast<int32_t>(mtmd_helper_get_n_tokens(app.mtmd_input_chunks.get()));
  res.n_pos.value = static_cast<int32_t>(mtmd_helper_get_n_pos(app.mtmd_input_chunks.get()));
  mtmd_populate_chunk_diagnostics(app, res);
  res.success.value = true;
  res.message.value = "tokenized MTMD input chunks";
  return res;
}

glue_msg_mtmd_eval_chunks_res action_mtmd_eval_chunks(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_mtmd_eval_chunks_req);
  glue_msg_mtmd_eval_chunks_res res;
  res.n_past.value = app.tokens.size();
  mtmd_reset_chunk_diagnostics(app, res);

  if (!app.mtmd_context)
  {
    res.success.value = false;
    res.message.value = "load an MTMD context before evaluating chunks";
    return res;
  }
  if (app.ctx == nullptr)
  {
    res.success.value = false;
    res.message.value = "load a llama context before evaluating chunks";
    return res;
  }
  if (!app.mtmd_input_chunks)
  {
    res.success.value = false;
    res.message.value = "tokenize MTMD input chunks before evaluating them";
    return res;
  }

  const llama_pos n_past = static_cast<llama_pos>(app.tokens.size());
  const int32_t n_batch = req.n_batch.value > 0 ? req.n_batch.value : llama_n_batch(app.ctx);
  const size_t n_chunks = mtmd_input_chunks_size(app.mtmd_input_chunks.get());
  const size_t n_tokens = mtmd_helper_get_n_tokens(app.mtmd_input_chunks.get());
  const llama_pos n_pos = mtmd_helper_get_n_pos(app.mtmd_input_chunks.get());
  mtmd_native_breadcrumb("action_mtmd_eval_chunks before mtmd_populate_chunk_diagnostics");
  const int32_t text_token_out_of_range_count = mtmd_populate_chunk_diagnostics(app, res);
  mtmd_native_breadcrumb("action_mtmd_eval_chunks after mtmd_populate_chunk_diagnostics");
  if (text_token_out_of_range_count > 0)
  {
    res.success.value = false;
    res.message.value = "mtmd_eval_chunks text token bounds preflight failed";
    res.n_past.value = static_cast<int32_t>(n_past);
    return res;
  }
  if (res.vision_position_blocker_kind.value == "missing_position_embeddings")
  {
    res.success.value = false;
    res.message.value = "mtmd_eval_chunks vision position embeddings missing";
    res.n_past.value = static_cast<int32_t>(n_past);
    return res;
  }
  if (res.vision_position_blocker_kind.value == "invalid_patch_geometry")
  {
    res.success.value = false;
    res.message.value = "mtmd_eval_chunks invalid vision patch geometry";
    res.n_past.value = static_cast<int32_t>(n_past);
    return res;
  }
  if (res.image_position_out_of_range_total.value > 0 ||
      res.vision_position_blocker_kind.value == "row_index_out_of_bounds")
  {
    res.success.value = false;
    res.message.value = "mtmd_eval_chunks image position bounds preflight failed";
    res.n_past.value = static_cast<int32_t>(n_past);
    return res;
  }

  llama_pos new_n_past = n_past;
  auto eval_exception_message = [&](const char *type, const std::string &message) {
    std::ostringstream ss;
    ss << "mtmd_helper_eval_chunks threw " << type;
    if (!message.empty())
    {
      ss << ": " << mtmd_bounded_exception_message(message);
    }
    ss << " (n_past=" << n_past
       << ", n_batch=" << n_batch
       << ", logits_last=" << (req.logits_last.value ? "true" : "false")
       << ", n_chunks=" << n_chunks
       << ", n_tokens=" << n_tokens
       << ", n_pos=" << n_pos << ")";
    return ss.str();
  };
  int32_t evaluated = 0;
  try
  {
    mtmd_native_breadcrumb("action_mtmd_eval_chunks before mtmd_helper_eval_chunks");
    evaluated = mtmd_helper_eval_chunks(
        app.mtmd_context.get(),
        app.ctx,
        app.mtmd_input_chunks.get(),
        n_past,
        0,
        n_batch,
        req.logits_last.value,
        &new_n_past);
    mtmd_native_breadcrumb("action_mtmd_eval_chunks after mtmd_helper_eval_chunks");
  }
  catch (const std::exception &e)
  {
    res.success.value = false;
    res.message.value = eval_exception_message("std::exception", e.what());
    res.n_past.value = static_cast<int32_t>(n_past);
    return res;
  }
  catch (const std::string &e)
  {
    res.success.value = false;
    res.message.value = eval_exception_message("std::string", e);
    res.n_past.value = static_cast<int32_t>(n_past);
    return res;
  }
  catch (const char *e)
  {
    res.success.value = false;
    res.message.value = eval_exception_message("const char *", e == nullptr ? "<null>" : std::string(e));
    res.n_past.value = static_cast<int32_t>(n_past);
    return res;
  }
  catch (...)
  {
    res.success.value = false;
    res.message.value = eval_exception_message("unknown exception", "");
    res.n_past.value = static_cast<int32_t>(n_past);
    return res;
  }
  if (evaluated != 0)
  {
    res.success.value = false;
    res.message.value = "mtmd_helper_eval_chunks failed with code " + std::to_string(evaluated);
    res.n_past.value = static_cast<int32_t>(n_past);
    return res;
  }

  mtmd_resize_token_ledger(app, new_n_past);
  app.last_logits_index = -1;
  app.mtmd_input_chunks.reset();

  res.success.value = true;
  res.message.value = "evaluated MTMD input chunks";
  res.n_past.value = static_cast<int32_t>(new_n_past);
  return res;
}
