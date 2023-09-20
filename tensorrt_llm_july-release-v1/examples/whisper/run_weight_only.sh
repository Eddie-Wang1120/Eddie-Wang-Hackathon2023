python3 build.py --use_gpt_attention_plugin --use_gemm_plugin --use_layernorm_plugin --use_weight_only --output_dir weight_only
python3 run.py --engine_dir weight_only