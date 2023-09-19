#export PYTHONPATH=$PYTHONPATH:/mnt/samsung-t7/yuekai/asr/Eddie-Wang-Hackthon2023/tensorrt_llm_july-release-v1
export PYTHONPATH=$PYTHONPATH:/mnt/samsung-t7/yuekai/asr/Eddie-Wang-Hackthon2023-latest/tensorrt_llm_july-release-v1
export CUDA_VISIBLE_DEVICES="1"
python build.py --use_gpt_attention_plugin  --use_gemm_plugin  --use_layernorm_plugin
#python build.py --use_gpt_attention_plugin float16 --use_gemm_plugin float16 --use_layernorm_plugin float16 --int8_kv_cache --use_weight_only
#python run.py --engine_dir whisper_outputs
#python run.py --engine_dir whisper_outputs_plugin
