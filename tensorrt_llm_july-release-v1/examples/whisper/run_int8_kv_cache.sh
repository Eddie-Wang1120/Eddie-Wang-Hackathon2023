python3 build.py --use_gpt_attention_plugin --use_gemm_plugin --use_layernorm_plugin --int8_kv_cache --output_dir int8_kv_cache
python3 run.py --engine_dir int8_kv_cache