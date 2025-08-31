#!/usr/bin/env bash
# 手动启动入口脚本：便于一条命令控制训练阶段、模型组件、是否使用切分数据和 GPU
# 用法示例：
#   bash run_custom_speaker_manual.sh --stage 0 --stop-stage 3 --models "llm" --gpu 0 --use-split
#   bash run_custom_speaker_manual.sh --prep-only --use-split
#   bash run_custom_speaker_manual.sh --full --models "llm flow hifigan" --gpu 0,1

set -Eeuo pipefail
trap 'echo "[ERROR] Failed at line $LINENO. Exit." >&2' ERR

# 默认参数
STAGE=0
STOP_STAGE=8
MODELS="llm"
GPU="0"
USE_SPLIT=false
PREP_ONLY=false
FULL_RUN=false

print_help() {
  cat <<'EOF'
用法: bash run_custom_speaker_manual.sh [选项]

选项：
  --stage N           起始阶段（默认 0）
  --stop-stage N      结束阶段（默认 8）
  --models "..."      训练组件列表（默认 "llm"；全训示例："llm flow hifigan"）
  --gpu "0,1"        指定 CUDA_VISIBLE_DEVICES（默认 "0"）
  --use-split         使用切分数据（asset/split_mp3s/{wavs,transcripts}）
  --no-split          使用原始 mp3（asset/mp3），需同名 .txt
  --prep-only         仅跑 0-3 阶段（数据准备到 parquet）
  --full              等价于 --stage 0 --stop-stage 8
  -h, --help          显示帮助

说明：脚本将把参数传递给 train_custom_speaker.sh，并设置 MODELS/环境变量。
EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="${2}"; shift 2;;
    --stop-stage) STOP_STAGE="${2}"; shift 2;;
    --models) MODELS="${2}"; shift 2;;
    --gpu) GPU="${2}"; shift 2;;
    --use-split) USE_SPLIT=true; shift;;
    --no-split) USE_SPLIT=false; shift;;
    --prep-only) PREP_ONLY=true; shift;;
    --full) FULL_RUN=true; shift;;
    -h|--help) print_help; exit 0;;
    *) echo "未知参数: $1"; print_help; exit 1;;
  esac
done

# 派生快捷选项
if $PREP_ONLY; then
  STAGE=0
  STOP_STAGE=3
fi
if $FULL_RUN; then
  STAGE=0
  STOP_STAGE=8
fi

# 环境设置
export CUDA_VISIBLE_DEVICES="$GPU"

# 透传给训练脚本的关键环境变量
export MODELS

# 依据是否使用切分数据，提示当前数据入口
if $USE_SPLIT; then
  echo "[INFO] 使用切分数据: asset/split_mp3s/wavs 与 transcripts"
  # 用户需在 train_custom_speaker.sh 中配置 use_split_data=true
  USE_SPLIT_FLAG=true
else
  echo "[INFO] 使用原始 mp3: asset/mp3（需有同名 .txt）"
  USE_SPLIT_FLAG=false
fi

# 将 use_split_data 覆盖传入子脚本（通过 env 覆写同名变量）
USE_SPLIT_ENV="use_split_data=${USE_SPLIT_FLAG}"

# 调用主脚本
echo "[INFO] 执行阶段: ${STAGE} -> ${STOP_STAGE}; MODELS=\"${MODELS}\"; GPU=\"${GPU}\""
${USE_SPLIT_ENV} stage=${STAGE} stop_stage=${STOP_STAGE} bash ./train_custom_speaker.sh
