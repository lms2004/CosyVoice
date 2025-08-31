#!/bin/bash
# 严格模式：任一命令失败立即退出；未定义变量报错；管道失败冒泡
set -Eeuo pipefail

# 失败时打印出错位置
trap 'echo "[ERROR] Failed at line $LINENO. Exit." >&2' ERR
# 单一说话人适配脚本 - 符合CosyVoice2官方流程

#=====================================================================
# 配置区域 - 所有可自定义参数集中在此处
#=====================================================================

# 1. 基本路径配置
#---------------------------------------------------------------------
# 预训练模型目录
pretrained_model_dir=/mnt/c/Users/lms/Desktop/CosyVoice/pretrained_models/CosyVoice2-0.5B
# 输出目录
output_dir=/mnt/c/Users/lms/Desktop/CosyVoice/custom_speaker_model

# 2. 数据目录配置
#---------------------------------------------------------------------
# 原始MP3文件目录（根据你的数据目录调整）
custom_data_dir=./asset/mp3
# 拆分后的音频文件目录
split_data_dir=./asset/split_mp3s/wavs
# 拆分后的文本转录目录
transcript_dir=./asset/split_mp3s/transcripts
# 说话人ID
speaker_id=my_tts

# 3. 训练控制参数
#---------------------------------------------------------------------
# 阶段控制
stage=0
stop_stage=7
# 是否使用拆分后的数据
use_split_data=true  # 设置为false则使用原始MP3文件

# 4. GPU和分布式训练配置
#---------------------------------------------------------------------
# GPU设置
export CUDA_VISIBLE_DEVICES="0"  # 根据您的GPU情况调整
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
# 分布式训练参数
job_id=1234
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp

# 5. 模型平均参数
#---------------------------------------------------------------------
average_num=5  # 平均最后几个检查点

# 6. 训练组件选择（默认先只训练 LLM；如需全部，改为："llm flow hifigan"）
MODELS="llm"

#=====================================================================
# 内部变量计算 - 不需要手动修改
#=====================================================================

# 根据配置决定使用哪个数据目录
if [ "$use_split_data" = true ] && [ -d "$split_data_dir" ]; then
  audio_data_dir=$split_data_dir
  audio_transcript_dir=$transcript_dir
  echo "使用拆分后的音频数据: $audio_data_dir"
else
  audio_data_dir=$custom_data_dir
  audio_transcript_dir=""
  echo "使用原始MP3文件: $audio_data_dir"
fi

# 创建必要的目录
mkdir -p $output_dir/data
mkdir -p $output_dir/exp/cosyvoice2
mkdir -p $output_dir/tensorboard/cosyvoice2
mkdir -p $output_dir/conf

# 1. 数据准备
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "数据准备，生成 wav.scp/text/utt2spk/spk2utt"
  
  # 准备数据目录
  mkdir -p $output_dir/data/custom_speaker
  # 已完成检测（四个Kaldi文件）
  if [ -s "$output_dir/data/custom_speaker/wav.scp" ] \
    && [ -s "$output_dir/data/custom_speaker/text" ] \
    && [ -s "$output_dir/data/custom_speaker/utt2spk" ] \
    && [ -s "$output_dir/data/custom_speaker/spk2utt" ]; then
    echo "[SKIP] 已检测到现有 Kaldi 文件，跳过数据准备阶段。"
  else
    
  # 根据是否有单独的文本目录决定命令参数
  if [ -n "$audio_transcript_dir" ] && [ -d "$audio_transcript_dir" ]; then
    # 使用单独的文本目录
    python /mnt/c/Users/lms/Desktop/CosyVoice/prepare_custom_data.py \
      --src_dir $audio_data_dir \
      --des_dir $output_dir/data/custom_speaker \
      --speaker_id $speaker_id \
      --transcript_dir $audio_transcript_dir
  else
    # 使用默认目录结构
    python /mnt/c/Users/lms/Desktop/CosyVoice/prepare_custom_data.py \
      --src_dir $audio_data_dir \
      --des_dir $output_dir/data/custom_speaker \
      --speaker_id $speaker_id
  fi
    
    # 成功校验
    for f in wav.scp text utt2spk spk2utt; do
      test -s "$output_dir/data/custom_speaker/$f" || { echo "[ERROR] 缺少 $f"; exit 1; }
    done
    echo "[OK] 数据准备完成。"
  fi
fi

# 2. 提取说话人嵌入向量
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "提取说话人嵌入向量"
  # 已完成检测
  if [ -s "$output_dir/data/custom_speaker/spk2embedding.pt" ] \
    && [ -s "$output_dir/data/custom_speaker/utt2embedding.pt" ]; then
    echo "[SKIP] 已检测到 embedding 结果，跳过。"
  else
    python /mnt/c/Users/lms/Desktop/CosyVoice/tools/extract_embedding.py \
      --dir $output_dir/data/custom_speaker \
      --onnx_path $pretrained_model_dir/campplus.onnx
    test -s "$output_dir/data/custom_speaker/spk2embedding.pt" || { echo "[ERROR] 缺少 spk2embedding.pt"; exit 1; }
    test -s "$output_dir/data/custom_speaker/utt2embedding.pt" || { echo "[ERROR] 缺少 utt2embedding.pt"; exit 1; }
    echo "[OK] 说话人嵌入完成。"
  fi
fi
# 注：以下行可能是误加的外部调用，会中断流程，故注释掉。
# python3 client.py --port 50000 --mode sft

# 3. 提取离散语音标记
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "提取离散语音标记"
  if [ -s "$output_dir/data/custom_speaker/utt2speech_token.pt" ]; then
    echo "[SKIP] 已检测到 utt2speech_token.pt，跳过。"
  else
    python /mnt/c/Users/lms/Desktop/CosyVoice/tools/extract_speech_token.py \
      --dir $output_dir/data/custom_speaker \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx
    test -s "$output_dir/data/custom_speaker/utt2speech_token.pt" || { echo "[ERROR] 缺少 utt2speech_token.pt"; exit 1; }
    echo "[OK] 语音 token 提取完成。"
  fi
fi

# 4. 准备Parquet格式数据
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "准备Parquet格式数据"
  mkdir -p $output_dir/data/custom_speaker/parquet
  if [ -s "$output_dir/data/custom_speaker/parquet/data.list" ]; then
    echo "[SKIP] 已检测到 parquet/data.list，跳过。"
  else
    python /mnt/c/Users/lms/Desktop/CosyVoice/tools/make_parquet_list.py \
      --num_utts_per_parquet 1000 \
      --num_processes 4 \
      --src_dir $output_dir/data/custom_speaker \
      --des_dir $output_dir/data/custom_speaker/parquet
    test -s "$output_dir/data/custom_speaker/parquet/data.list" || { echo "[ERROR] 缺少 parquet/data.list"; exit 1; }
    echo "[OK] Parquet 列表生成完成。"
  fi
fi

# 5. 创建训练和验证数据列表
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "创建训练和验证数据列表"
  # 将数据分为训练集和验证集
  if [ -s "$output_dir/data/train.data.list" ] && [ -s "$output_dir/data/dev.data.list" ]; then
    echo "[SKIP] 已检测到 train/dev 列表，跳过。"
  else
    total_files=$(wc -l < $output_dir/data/custom_speaker/parquet/data.list)
    train_count=$((total_files * 9 / 10))
    if [ $train_count -lt 1 ]; then
      train_count=1
    fi
    head -n $train_count $output_dir/data/custom_speaker/parquet/data.list > $output_dir/data/train.data.list
    if [ $train_count -lt $total_files ]; then
      tail -n +$((train_count + 1)) $output_dir/data/custom_speaker/parquet/data.list > $output_dir/data/dev.data.list
    else
      cp $output_dir/data/train.data.list $output_dir/data/dev.data.list
    fi
    test -s "$output_dir/data/train.data.list" || { echo "[ERROR] 缺少 train.data.list"; exit 1; }
    test -s "$output_dir/data/dev.data.list" || { echo "[ERROR] 缺少 dev.data.list"; exit 1; }
    echo "[OK] 训练/验证列表创建完成。"
  fi
fi

# 6. 复制并修改配置文件
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "准备配置文件"
  if [ -s "$output_dir/conf/cosyvoice2.yaml" ]; then
    echo "[SKIP] 检测到已存在的配置文件，跳过复制与修改。"
  else
    cp /mnt/c/Users/lms/Desktop/CosyVoice/examples/libritts/cosyvoice2/conf/cosyvoice2.yaml $output_dir/conf/
    # 修改配置文件中的微调参数
    sed -i 's/use_spk_embedding: False/use_spk_embedding: True/' $output_dir/conf/cosyvoice2.yaml || true
    sed -i 's/lr: 1e-5 # change to 1e-5 during sft/lr: 1e-5/' $output_dir/conf/cosyvoice2.yaml || true
    sed -i 's/scheduler: constantlr # change to constantlr during sft/scheduler: constantlr/' $output_dir/conf/cosyvoice2.yaml || true
    test -s "$output_dir/conf/cosyvoice2.yaml" || { echo "[ERROR] 配置文件缺失"; exit 1; }
  fi
fi

# 7. 训练模型
# GPU和分布式训练参数已在配置区域设置

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "开始训练模型"
  
  # 训练所选模型组件
  for model in $MODELS; do
    # 如果最终产物已存在则跳过该组件训练
    if [ -s "$output_dir/exp/cosyvoice2/$model/$train_engine/${model}.pt" ]; then
      echo "[SKIP] 检测到 $model 已训练完成，跳过。"
      continue
    fi
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
      --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      /mnt/c/Users/lms/Desktop/CosyVoice/cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config $output_dir/conf/cosyvoice2.yaml \
      --train_data $output_dir/data/train.data.list \
      --cv_data $output_dir/data/dev.data.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir $output_dir/exp/cosyvoice2/$model/$train_engine \
      --tensorboard_dir $output_dir/tensorboard/cosyvoice2/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp
    test -s "$output_dir/exp/cosyvoice2/$model/$train_engine/${model}.pt" || { echo "[ERROR] $model 训练未生成 ${model}.pt"; exit 1; }
  done
fi

# 8. 模型平均
# 模型平均参数已在配置区域设置
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  for model in $MODELS; do
    model_dir=$output_dir/exp/cosyvoice2/$model/$train_engine
    decode_checkpoint=$model_dir/${model}.pt
    
    # 检查是否有足够的检查点进行平均
    checkpoint_count=$(ls -1 $model_dir/checkpoint_*.pt 2>/dev/null | wc -l)
    if [ $checkpoint_count -lt $average_num ]; then
      average_num=$checkpoint_count
    fi
    
    if [ $average_num -gt 0 ]; then
      echo "执行模型平均，最终检查点为 $decode_checkpoint"
      python /mnt/c/Users/lms/Desktop/CosyVoice/cosyvoice/bin/average_model.py \
        --dst_model $decode_checkpoint \
        --src_path $model_dir \
        --num ${average_num} \
        --val_best
    else
      echo "警告: 没有足够的检查点进行平均，跳过模型 $model 的平均步骤"
    fi
    test -s "$decode_checkpoint" || { echo "[ERROR] $model 平均后未生成 ${model}.pt"; exit 1; }
  done
fi

# 9. 导出模型
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "导出模型以加速推理"
  # 将训练好的模型复制到一个目录
  export_dir=$output_dir/exp/export
  mkdir -p $export_dir
  
  for model in $MODELS; do
    
    if [ -f $output_dir/exp/cosyvoice2/$model/$train_engine/${model}.pt ]; then
      cp $output_dir/exp/cosyvoice2/$model/$train_engine/${model}.pt $export_dir/
    else
      echo "警告: 模型文件 $model.pt 不存在，无法导出"
    fi
  done
  
  # 检查是否所有必要的模型文件都存在
  all_present=true
  for model in $MODELS; do
    if [ ! -f $export_dir/${model}.pt ]; then all_present=false; fi
  done
  if $all_present; then
    # 导出为JIT和ONNX格式
    python /mnt/c/Users/lms/Desktop/CosyVoice/cosyvoice/bin/export_jit.py --model_dir $export_dir
    python /mnt/c/Users/lms/Desktop/CosyVoice/cosyvoice/bin/export_onnx.py --model_dir $export_dir
    echo "模型导出完成，可以在 $export_dir 目录找到导出的模型"
  else
    echo "错误: 缺少必要的模型文件，无法完成导出"
    exit 1
  fi
fi

echo "训练流程完成！"
