# CosyVoice2 单一说话人模型训练指南

本文档提供了使用大量同一说话人MP3音频数据训练CosyVoice2模型的完整指南。

## 准备工作

1. **安装依赖**

确保已安装CosyVoice2的所有依赖：
```bash
pip install -r requirements.txt
```

此外，还需要安装以下额外依赖以支持MP3格式和音频处理：
```bash
pip install soundfile torchaudio pydub librosa numpy tqdm
```

2. **准备预训练模型**

确保您已下载CosyVoice2-0.5B预训练模型，并放置在以下目录：
```
modelscope download --model iic/CosyVoice2-0.5B --local_dir ./pretrained_models/CosyVoice2-0.5B/
```

预训练模型应包含以下文件：
- campplus.onnx (说话人嵌入模型)
- speech_tokenizer_v2.onnx (语音标记提取模型)
- llm.pt, flow.pt, hifigan.pt (预训练的CosyVoice2模型)
- CosyVoice-BlankEN (Qwen预训练模型)

3. **准备数据**

准备您的MP3音频文件和对应的文本转录。文本文件应与音频文件同名，但扩展名为.txt。例如：
- audio1.mp3 → audio1.txt
- audio2.mp3 → audio2.txt

对于较长的音频文件（超过15秒），建议先进行拆分处理，以获得更好的训练效果。

## 训练流程

### 1. 拆分长音频文件（可选但推荐）

如果您的MP3文件较长（超过15秒），请使用`split_long_audio.py`脚本将其拆分为短片段：

```bash
python split_long_audio.py \
  --src_dir ./asset/mp3s \
  --output_dir ./asset/split_mp3s \
  --min_length 5.0 \
  --max_length 15.0 \
  --generate_transcripts
```

参数说明：
- `--src_dir`: 源MP3文件目录
- `--output_dir`: 输出目录
- `--min_length`: 最小片段长度(秒)，默认5秒
- `--max_length`: 最大片段长度(秒)，默认15秒
- `--silence_thresh`: 静音阈值(dB)，默认-40dB，越小越严格
- `--min_silence_len`: 最小静音长度(毫秒)，默认500ms
- `--keep_silence`: 保留的静音长度(毫秒)，默认300ms
- `--generate_transcripts`: 生成文本转录占位符

脚本会在输出目录中创建两个子目录：
- `wavs`: 包含拆分后的音频片段
- `transcripts`: 包含对应的文本转录占位符

**重要**: 请编辑生成的文本文件，填写正确的转录内容，这对训练效果至关重要。

### 2. 数据准备

使用`prepare_custom_data.py`脚本处理MP3文件：

```bash
python prepare_custom_data.py \
  --src_dir ./asset/split_mp3s/wavs \
  --des_dir /mnt/c/Users/lms/Desktop/CosyVoice/custom_speaker_model/data/custom_speaker \
  --speaker_id my_tts \
  --transcript_dir ./asset/split_mp3s/transcripts
```

如果您没有拆分音频，则直接使用原始MP3目录：

```bash
python prepare_custom_data.py \
  --src_dir ./asset/mp3s \
  --des_dir /mnt/c/Users/lms/Desktop/CosyVoice/custom_speaker_model/data/custom_speaker \
  --speaker_id my_tts
```

参数说明：
- `--src_dir`: MP3文件所在目录
- `--des_dir`: 处理后的数据输出目录
- `--speaker_id`: 说话人ID，用于标识模型
- `--transcript_dir`: (可选) 如果文本文件与音频不在同一目录，可以指定文本目录

### 3. 执行训练

使用`train_custom_speaker.sh`脚本执行完整的训练流程：

```bash
# 首先修改脚本中的custom_data_dir变量，指向您的MP3文件目录
# 然后添加执行权限
chmod +x train_custom_speaker.sh

# 执行训练
./train_custom_speaker.sh
```

训练脚本包含以下阶段：
- 阶段0: 数据准备 (如果您已经使用prepare_custom_data.py准备好数据，可以跳过)
- 阶段1: 提取说话人嵌入向量
- 阶段2: 提取离散语音标记
- 阶段3: 准备Parquet格式数据
- 阶段4: 创建训练和验证数据列表
- 阶段5: 准备配置文件
- 阶段6: 训练模型 (llm, flow, hifigan)
- 阶段7: 模型平均
- 阶段8: 导出模型

您可以通过修改脚本中的`stage`和`stop_stage`变量来控制执行哪些阶段。

### 4. 使用训练好的模型

训练完成后，您可以使用`use_custom_model.py`脚本生成语音：

```bash
python use_custom_model.py \
  --model_dir /mnt/c/Users/lms/Desktop/CosyVoice/custom_speaker_model/exp/export \
  --speaker_id custom_speaker \
  --text "这是使用我的自定义说话人模型生成的语音。" \
  --output output.wav
```

参数说明：
- `--model_dir`: 训练好的模型目录
- `--speaker_id`: 说话人ID，与训练时使用的ID一致
- `--text`: 要合成的文本
- `--output`: 输出音频文件路径
- `--use_jit`: 使用JIT模型加速 (默认开启)
- `--use_fp16`: 使用半精度浮点数(FP16)加速
- `--stream`: 使用流式生成模式
- `--instruction`: 可选的指令，如情感控制

## 训练参数调整

如果需要调整训练参数，可以修改以下文件：
- `custom_speaker_model/conf/cosyvoice2.yaml`: 模型和训练配置

主要可调整的参数包括：
- 学习率: `lr: 1e-5`
- 训练轮数: `max_epoch: 200`
- 梯度累积: `accum_grad: 2`
- 批次大小: 通过`max_frames_in_batch`间接控制

## 故障排除

1. **内存不足**
   - 减小`max_frames_in_batch`值
   - 减少`num_processes`参数

2. **GPU内存不足**
   - 使用`--use_fp16`参数启用半精度训练
   - 增加梯度累积步数`accum_grad`

3. **训练不稳定**
   - 减小学习率
   - 检查音频质量，确保没有噪音或静音片段

## 数据质量建议

为获得最佳效果，请确保：
1. 音频质量清晰，无明显背景噪音
2. 每个音频片段长度适中(5-15秒) - 使用`split_long_audio.py`可以帮助您实现这一点
3. 文本转录准确 - 这对模型学习正确的发音至关重要
4. 数据量充足(建议30分钟以上的有效语音)
5. 内容多样，覆盖不同语调和表达方式
6. 音频片段开头和结尾没有长时间的静音
7. 避免背景音乐或其他说话人的声音干扰
