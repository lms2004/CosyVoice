# CosyVoice/CosyVoice2 LibriTTS 训练流程

本说明覆盖 `examples/libritts/` 下两套示例：
- cosyvoice（对应 `examples/libritts/cosyvoice/`）
- cosyvoice2（对应 `examples/libritts/cosyvoice2/`）

两者的数据准备基本一致，差异主要在预训练模型与语音分词器版本、以及 CosyVoice2 的 `qwen_pretrain_path`。

---

## 目录结构

- `cosyvoice/`
  - `run.sh`：完整流程脚本（下载→准备→embedding→speech token→parquet→训练→平均→导出）
  - `conf/`：训练配置 `cosyvoice.yaml`，以及可选的 DeepSpeed 配置 `ds_stage2.json`
  - `local/`：`download_and_untar.sh`、`prepare_data.py` 等
  - `path.sh`：设置 `PYTHONPATH`
- `cosyvoice2/`
  - `run.sh`：CosyVoice2 对应的流程脚本
  - `run_dpo.sh`：DPO 相关脚本（如需）
  - `conf/`：`cosyvoice2.yaml` 与 `ds_stage2.json`
  - `path.sh`

---

## 前置依赖

1. Python 依赖
   - 项目根目录安装：
     ```bash
     pip install -r requirements.txt
     ```
2. 环境变量
   - 所有命令需在子目录执行前先 `source ./path.sh`，确保 `PYTHONPATH` 指向项目根与 `third_party/Matcha-TTS`。
3. 预训练模型目录（默认脚本里的路径）
   - cosyvoice: `../../../pretrained_models/CosyVoice-300M/`
     - 需要：`llm.pt`、`flow.pt`、`hifigan.pt`、`campplus.onnx`、`speech_tokenizer_v1.onnx`
   - cosyvoice2: `../../../pretrained_models/CosyVoice2-0.5B/`
     - 需要：`llm.pt`、`flow.pt`、`hifigan.pt`、`campplus.onnx`、`speech_tokenizer_v2.onnx`
     - 还需：`CosyVoice-BlankEN`（用于 `--qwen_pretrain_path`）
4. GPU 与分布式
   - 默认 `CUDA_VISIBLE_DEVICES="0,1,2,3"`，使用 `torchrun` + DDP
   - 也可切换为 `deepspeed`，对应配置见 `conf/ds_stage2.json`

---

## 数据集

- LibriTTS（OpenSLR 60）：`dev-clean`、`dev-other`、`test-clean`、`test-other`、`train-clean-100`、`train-clean-360`、`train-other-500`
- 默认下载根目录变量：`data_dir=/path/to/libritts`（脚本中给了示例路径，需按本机修改）

---

## 统一流程说明（以 cosyvoice 为例，cosyvoice2 类似）

进入对应目录：
```bash
cd examples/libritts/cosyvoice
source ./path.sh
```

脚本通过 `stage` 与 `stop_stage` 控制分阶段执行，默认如下：
- `stage=-1` 到 `stop_stage=3`：完成数据下载、准备、特征提取与 parquet 列表生成。
- 训练与导出步骤需将 `stop_stage` 调整为更大数值（见下方）。

你可以直接运行：
```bash
bash run.sh
```
或仅执行某阶段区间，例如只做数据准备到 parquet：
```bash
stage=-1 stop_stage=3 bash run.sh
```

### 阶段 -1：下载数据（可选）
- 脚本：`local/download_and_untar.sh`
- 控制：当 `stage<=-1<=stop_stage` 时执行
- 从 `www.openslr.org/resources/60` 下载并解压到 `${data_dir}/LibriTTS/` 下各子集

### 阶段 0：数据准备
- 脚本：`local/prepare_data.py`
- 输出到 `data/<subset>/`，包含：`wav.scp`、`text`、`utt2spk`、`spk2utt`
- 命令（由 `run.sh` 调用）：
  ```bash
  python local/prepare_data.py --src_dir $data_dir/LibriTTS/$x --des_dir data/$x
  ```

### 阶段 1：说话人 embedding 提取
- 脚本：`tools/extract_embedding.py`
- 依赖模型：`campplus.onnx`
- 输出：`spk2embedding.pt`、`utt2embedding.pt`（位于 `data/<subset>/`）
- 命令示例：
  ```bash
  tools/extract_embedding.py --dir data/$x --onnx_path $pretrained_model_dir/campplus.onnx
  ```

### 阶段 2：离散语音 token 提取
- 脚本：`tools/extract_speech_token.py`
- 依赖模型：
  - cosyvoice：`speech_tokenizer_v1.onnx`
  - cosyvoice2：`speech_tokenizer_v2.onnx`
- 输出：`utt2speech_token.pt`（位于 `data/<subset>/`）
- 命令示例：
  ```bash
  tools/extract_speech_token.py --dir data/$x --onnx_path $pretrained_model_dir/speech_tokenizer_v1.onnx
  # 或在 cosyvoice2：speech_tokenizer_v2.onnx
  ```

### 阶段 3：生成 parquet 列表
- 脚本：`tools/make_parquet_list.py`
- 输出：`data/<subset>/parquet/data.list` 与 parquet 目录
- 典型参数：`--num_utts_per_parquet 1000`、`--num_processes 10`
- 命令示例：
  ```bash
  tools/make_parquet_list.py \
    --num_utts_per_parquet 1000 \
    --num_processes 10 \
    --src_dir data/$x \
    --des_dir data/$x/parquet
  ```

### 阶段 5：训练（当前示例仅支持 LLM 训练，已保留 flow/hifigan 循环以兼容流程）
- 预先合并训练/验证列表：
  ```bash
  cat data/{train-clean-100,train-clean-360,train-other-500}/parquet/data.list > data/train.data.list
  cat data/{dev-clean,dev-other}/parquet/data.list > data/dev.data.list
  ```
- 关键参数：
  - `--config`：`conf/cosyvoice.yaml` 或 `conf/cosyvoice2.yaml`
  - `--model`：`llm`（示例会循环到 `flow`、`hifigan`，但备注中提示“仅支持 llm 训练”）
  - `--checkpoint`：加载对应的预训练 `*.pt`
  - `--train_engine`：`torch_ddp`（默认）或 `deepspeed`
  - cosyvoice2 额外：`--qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN`
- 训练命令（由脚本 `torchrun` 调用，示例）：
  ```bash
  torchrun --nnodes=1 --nproc_per_node=$num_gpus \
    --rdzv_id=1986 --rdzv_backend=c10d --rdzv_endpoint=localhost:1234 \
    cosyvoice/bin/train.py \
    --train_engine torch_ddp \
    --config conf/cosyvoice.yaml \
    --train_data data/train.data.list \
    --cv_data data/dev.data.list \
    --model llm \
    --checkpoint $pretrained_model_dir/llm.pt \
    --model_dir `pwd`/exp/cosyvoice/llm/torch_ddp \
    --tensorboard_dir `pwd`/tensorboard/cosyvoice/llm/torch_ddp \
    --ddp.dist_backend nccl \
    --num_workers 2 \
    --prefetch 100 \
    --pin_memory \
    --use_amp \
    --deepspeed_config ./conf/ds_stage2.json \
    --deepspeed.save_states model+optimizer
  ```

### 阶段 6：模型平均
- 脚本：`cosyvoice/bin/average_model.py`
- 将 `exp/.../$model/$train_engine/` 下最佳若干 checkpoint 平均，输出为 `${model}.pt`
- 典型命令：
  ```bash
  python cosyvoice/bin/average_model.py \
    --dst_model `pwd`/exp/cosyvoice/llm/torch_ddp/llm.pt \
    --src_path `pwd`/exp/cosyvoice/llm/torch_ddp \
    --num 5 \
    --val_best
  ```

### 阶段 7：导出 JIT/ONNX（推理加速）
- 脚本：`cosyvoice/bin/export_jit.py`、`cosyvoice/bin/export_onnx.py`
- 命令：
  ```bash
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
  ```
  提示：将你训练得到/平均后的 `llm.pt` 或 `flow.pt` 覆盖/拷贝到 `--model_dir` 再导出。

---

## CosyVoice2 与 CosyVoice 的差异点小结

- 预训练目录不同：`CosyVoice-300M` vs `CosyVoice2-0.5B`
- 语音分词器：v1 → v2（`speech_tokenizer_v2.onnx`）
- 训练参数增加：`--qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN`
- 其余流程（下载/准备/embedding/提取 token/parquet/训练/平均/导出）一致

---

## 常见问题（FAQ）

- 路径不存在或权限错误
  - 请先检查 `data_dir`、`pretrained_model_dir`，并确保 `path.sh` 正确 source。
- 显卡数量与显存
  - 调整 `CUDA_VISIBLE_DEVICES` 与 batch/accum 参数（如从配置中修改）以适配本机资源。
- 仅做数据准备
  - 使用 `stage=-1 stop_stage=3 bash run.sh`。
- 自有语音数据集替换 LibriTTS
  - 保持 `data/<subset>/` 下 Kaldi 风格文件（`wav.scp`、`text`、`utt2spk`、`spk2utt`）的一致性，然后执行阶段 1/2/3。

---

## 一键脚本入口

- CosyVoice：`examples/libritts/cosyvoice/run.sh`
- CosyVoice2：`examples/libritts/cosyvoice2/run.sh`

根据需要设置 `stage/stop_stage` 与各自的 `pretrained_model_dir`/`data_dir` 后运行即可。
