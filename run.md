To give you an exact end-to-end flow with correct command syntax, I'll quickly open `split_long_audio.py` to confirm its CLI arguments and outputs, so the steps for MP3 processing and ASR are accurate. Then I'll provide a numbered list of independent commands from MP3 → splitting → ASR → data prep → embeddings → tokens → parquet → lists → config → train → average → export.

I'll open [batch_asr.py](cci:7://file:///mnt/c/Users/lms/Desktop/CosyVoice/batch_asr.py:0:0-0:0) to confirm its CLI arguments so I can give you exact, runnable commands for ASR.

下面给出“从 MP3 开始的完整流程”，每一步都是可独立执行的一条命令。请先在 [train_custom_speaker.sh](cci:7://file:///mnt/c/Users/lms/Desktop/CosyVoice/train_custom_speaker.sh:0:0-0:0) 中确认或设置：
- use_split_data=true
- split_data_dir="asset/split_mp3s/wavs"
- transcript_dir="asset/split_mp3s/transcripts"
- pretrained_model_dir、output_dir、MODELS

若你未改脚本变量，请按上面路径来创建数据并运行。

# 0. 将 MP3 切分为短音频片段（并生成转写占位符）
```bash
python3 split_long_audio.py \
  --src_dir asset/mp3 \
  --output_dir asset/split_mp3s \
  --min_length 5 \
  --max_length 15 \
  --silence_thresh -40 \
  --min_silence_len 500 \
  --keep_silence 300 \
  --generate_transcripts
```
产物：
- 切分音频：`asset/split_mp3s/wavs/*.wav`
- 占位文本：`asset/split_mp3s/transcripts/*.txt`

# 1. 批量 ASR 生成转写（覆盖占位符）
```bash
python3 batch_asr.py \
  --wav_dir asset/split_mp3s/wavs \
  --out_dir asset/split_mp3s/transcripts \
  --num_workers 2
  --overwrite

python3 batch_asr.py \
  --wav_dir asset/split_mp3s/wavs \
  --out_dir asset/split_mp3s/transcripts \
  --num_workers 2 \
  --limit 53 \
  --overwrite
```
说明：如需重复识别并覆盖已有 .txt，加 `--overwrite`。

# 2. 数据准备（Kaldi 文件与 16k wav）
```bash
stage=0 stop_stage=0 bash train_custom_speaker.sh
```

# 3. 提取说话人嵌入（campplus.onnx）
```bash
stage=1 stop_stage=1 bash train_custom_speaker.sh
```

# 4. 提取语音离散 token（speech_tokenizer_v2.onnx）
```bash
stage=2 stop_stage=2 bash train_custom_speaker.sh
```

# 5. 生成 parquet 与 data.list
```bash
stage=3 stop_stage=3 bash train_custom_speaker.sh
```

# 6. 生成 train/dev 列表
```bash
stage=4 stop_stage=4 bash train_custom_speaker.sh
```

# 7. 准备训练配置（cosyvoice2.yaml）
```bash
stage=5 stop_stage=5 bash train_custom_speaker.sh
```

# 8. 训练（默认 MODELS=\"llm\"；如需全训改为 \"llm flow hifigan\"）
```bash
CUDA_VISIBLE_DEVICES="0" stage=6 stop_stage=6 bash train_custom_speaker.sh
```

# 9. 模型平均
```bash
stage=7 stop_stage=7 bash train_custom_speaker.sh
```

# 10. 导出（JIT/ONNX）
```bash
stage=8 stop_stage=8 bash train_custom_speaker.sh
```

可选与检查
- 切换 GPU：
```bash
CUDA_VISIBLE_DEVICES="0,1" stage=6 stop_stage=6 bash train_custom_speaker.sh
```
- 快速确认切分与转写数量是否匹配：
```bash
bash -lc 'ls asset/split_mp3s/wavs/*.wav | wc -l; ls asset/split_mp3s/transcripts/*.txt | wc -l'
```

总结
- 上述命令覆盖 MP3 → 切分 → ASR → 数据准备 → 嵌入/Token → Parquet → 列表 → 配置 → 训练 → 平均 → 导出，全链路每步一条独立命令。
- 脚本自带幂等与失败即停；重复执行会自动跳过已完成步骤。若你希望，我也可以把“ASR 作为前置 stage=-1”集成进 [train_custom_speaker.sh](cci:7://file:///mnt/c/Users/lms/Desktop/CosyVoice/train_custom_speaker.sh:0:0-0:0)，再提供对应一步一命令。