Whisper Model For Tensorrt-LLM
==============
## 模型实现思路
使用的模型为whisper的large模型，支持多语言，保证功能完全。将whisper分解为encoder和decoder两部分，在tensorrt-llm的model文件夹中分别用WhisperEncoder和WhisperDecoder。同时在完整的编码和解码流程在run.py中分别封装为WhisperEncoding和WhisperDecoding两部分，具体实现包含在encoding.py和decoding.py中，最大程度的保证代码的整洁和可读性。

## 对Tensorrt-LLM的改动
原始的Tensorrt-LLM不支持Conv1D和CrossAttention两个算子，对语音识别模型并不友好。在原本Tensorrt-LLM的基础上，我基于原版的Conv2D算子实现了Conv1D算子，基于原版的Attention算子（只支持MultiHeadAttention）实现了CrossAttention算子（仍在Attention内部实现，使Attention抽象正确），最大程度保留了原版代码的同时增加了新的功能，保证之前的模型仍能正常使用，完成了Tensorrt-LLM的正确迭代。

## 可能的BUG
* linear算子权重初始化问题  
  在weight.py对权重进行初始化时，使用np.ascontiguousarray(weight.numpy())将权重内存空间作为连续存储。该操作应该对按连续内存存储的权重无改变，对不按连续存储的权重有正面作用，即无负面作用。但对tensorrt_llm_whisper.blocks[i].attn.dense.weight.value和tensorrt_llm_whisper.blocks[i].attn.dense.weight.value进行该操作时，生成的结果错误，推测linear内部关于内存处理方面操作存在问题。  
  操作前结果：
  ![image](./imgs/before.png)  
  操作后结果：  
  ![image](./imgs/bug1_after.png)  
  如何复现bug：  
  在examples/whisper/weight.py中将67行代码改为下图所示，之后重新build encoder即可
  ![image](./imgs/bug1_how.png)  
  
* linear算子transB问题  
  在weight.py对权重进行初始化时，由于原版RowLinear算子固定transb为True，所以将原本的weight进行transpose后置入模型，但结果错误。更加离谱的是，就算不进行transpose操作，单纯将原本的weight置入模型，依然会得到相同的错误结果，即此时是否transpose都会得到相同的错误结果，推测linear内部关于内存处理方面操作存在问题。  
  操作前结果：  
  ![image](./imgs/before.png)  
  操作后结果：  
  ![image](./imgs/bug2_after.png)  
  如何复现bug：
  在examples/whisper/weight.py中将67行代码改为下图所示（有无transpose(0,1)对结果无影响）
  ![image](./imgs/bug2_how_1.png)  
  在tensorrt-llm/layers/linear.py中将当前RowLinear类代码注释，并将142行被注释的原版代码解除注释，之后重新build encoder即可
  ![image](./imgs/bug2_how_2.png)  


## 性能对比
* 纯英文音频，全长18.15秒  
pytorch FP16版本encoder+decoder 总计时长约2.18秒
![image](./imgs/torch_en.png)
trt-llm FP16版本encoder+decoder 总计时长约1.15秒 （未使用任何插件）
![image](./imgs/trt_en.png)
提速约47%，质量未见滑落
* 中英文混杂视频，全长11.90秒
pytorch FP16版本encoder+decoder 总计时长约2.66秒
![image](./imgs/torch_cn.png)
trt-llm FP16版本encoder+decoder 总计时长约1.53秒 （未使用任何插件）
![image](./imgs/trt_cn.png)
提速约42%，质量未见滑落

## 测试方法
进入examples/whisper目录  
To prepare model:  
将whisper的large-v2.pt放置于该文件夹下  
To build:  
```
python3 build.py
```

To run:
```
python3 run.py --input_file test.m4a
```
提供两个作者录制的示例音频，分别为test.m4a和test_chinese.m4a

To int8:
```
python3 torch_whisper_convert.py -i large-v2.py -o quantize
python3 build.py --use_gpt_attention_plugin --use_gemm_plugin --use_layernorm_plugin --int8_kv_cache
python3 run.py
```

To summarize:
```
python3 summarize.py
```
