# 自定义说话人数据工作区说明

本文档说明 `custom_speaker_model/data/custom_speaker/` 在各阶段会生成的文件与用途。流程由 `train_custom_speaker.sh` 驱动。

## 概览
- 数据根目录：`custom_speaker_model/data/custom_speaker/`
- 阶段总览（环境变量控制：`stage`、`stop_stage`）：
  - stage 0：准备 16kHz WAV 与 Kaldi 清单
  - stage 1：提取说话人/语音嵌入（campplus.onnx）
  - stage 2：提取语音离散 token（speech_tokenizer_v2.onnx）
  - stage 3：生成 parquet 与 `parquet/data.list`
  - stage 4：生成训练/验证列表 `train.data.list`、`dev.data.list`

## 按阶段的产物与说明

- wavs/
  - 16kHz 单声道 WAV。由你的 mp3/wav 统一重采样而来。
  - 示例：`wavs/1025059903_031.wav`

- wav.scp
  - Kaldi 音频清单：每行“语音ID 绝对路径”。
  - 示例：`my_tts_1025059903_031 /abs/path/.../wavs/1025059903_031.wav`

- text
  - Kaldi 文本：每行“语音ID 文本内容”。用于有监督微调。
  - 示例：`my_tts_1025059903_031 你好，今天天气怎么样？`

- utt2spk
  - 语音ID → 说话人ID。示例：`my_tts_1025059903_031 my_tts`

- spk2utt
  - 说话人ID → 其下全部语音ID 列表。

- spk2embedding.pt（stage 1）
  - 说话人级嵌入，字典：`{spk_id: embedding_tensor}`。

- utt2embedding.pt（stage 1）
  - 语音级嵌入，字典：`{utt_id: embedding_tensor}`。

- utt2speech_token.pt（stage 2）
  - 语音离散 token，字典：`{utt_id: token_tensor}`。

- parquet/（stage 3）
  - 若干 `.parquet` 文件：聚合音频路径、文本、嵌入、token 等字段。
  - `parquet/data.list`：列出全部 parquet 路径，供训练装载。

- 上级目录 `custom_speaker_model/data/`（stage 4）
  - `train.data.list`、`dev.data.list`：从 `parquet/data.list` 切分得到。

## 快速开始（常用命令）

- 仅数据准备（Kaldi 文件与 16k WAV）
```bash
stage=0 stop_stage=0 bash train_custom_speaker.sh
```

- 说话人嵌入（campplus.onnx）
```bash
stage=1 stop_stage=1 bash train_custom_speaker.sh
```

- 语音 token（speech_tokenizer_v2.onnx）
```bash
stage=2 stop_stage=2 bash train_custom_speaker.sh
```

- 生成 parquet 与 data.list
```bash
stage=3 stop_stage=3 bash train_custom_speaker.sh
```

- 生成 train/dev 列表
```bash
stage=4 stop_stage=4 bash train_custom_speaker.sh
```

## 必要条件与排错

- 文本 `.txt` 需与音频同名且一一对应，否则该条会被跳过，不写入 `wav.scp/text`。
- 若 `onnxruntime` 与 NumPy 版本冲突（如 `_ARRAY_API not found`）：
```bash
pip install "numpy<2" --upgrade --no-cache-dir
pip install onnxruntime==1.17.3 --no-cache-dir
# 如需 GPU 版：
# pip install onnxruntime-gpu==1.17.3 --no-cache-dir
```
- 若未使用 OpenCV，可卸载冲突的 `opencv-python-headless`；若需要，降级到与 NumPy<2 兼容的版本（如 4.10.0.84）。

## 参考
- 脚本：`train_custom_speaker.sh`
- 数据准备：`prepare_custom_data.py`（支持 mp3/wav 输入，统一重采样至 16kHz 单声道）
