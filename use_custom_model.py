#!/usr/bin/env python3
import sys
import os
import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path

# 添加Matcha-TTS路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'third_party/Matcha-TTS'))

# 导入CosyVoice2
from cosyvoice.cli.cosyvoice import CosyVoice2

def main():
    parser = argparse.ArgumentParser(description='使用自定义训练的CosyVoice2模型生成语音')
    parser.add_argument('--model_dir', type=str, default='custom_speaker_model/exp/export',
                        help='训练好的模型目录路径')
    parser.add_argument('--speaker_id', type=str, default='custom_speaker',
                        help='说话人ID，与训练时使用的ID一致')
    parser.add_argument('--text', type=str, required=True,
                        help='要合成的文本')
    parser.add_argument('--output', type=str, default='output.wav',
                        help='输出音频文件路径')
    parser.add_argument('--use_jit', action='store_true', default=True,
                        help='使用JIT模型加速')
    parser.add_argument('--use_fp16', action='store_true', default=False,
                        help='使用半精度浮点数(FP16)加速')
    parser.add_argument('--stream', action='store_true', default=False,
                        help='使用流式生成模式')
    parser.add_argument('--instruction', type=str, default=None,
                        help='可选的指令，如情感控制')
    args = parser.parse_args()

    # 确保模型目录存在
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"错误: 模型目录 {model_dir} 不存在")
        return

    # 检查必要的模型文件
    required_files = ['llm.pt', 'flow.pt', 'hifigan.pt']
    if args.use_jit:
        required_files = ['llm.jit', 'flow.jit', 'hifigan.jit']
    
    missing_files = [f for f in required_files if not (model_dir / f).exists()]
    if missing_files:
        print(f"错误: 缺少必要的模型文件: {', '.join(missing_files)}")
        print(f"如果指定了--use_jit但模型未导出为JIT格式，请移除--use_jit参数")
        return

    try:
        print(f"正在加载模型，路径: {model_dir}")
        cosyvoice = CosyVoice2(
            str(model_dir),
            load_jit=args.use_jit,
            load_trt=False,
            fp16=args.use_fp16
        )
        
        print(f"模型加载成功，开始生成语音...")
        print(f"文本: {args.text}")
        print(f"说话人: {args.speaker_id}")
        
        # 生成语音
        if args.instruction:
            # 使用指令模式
            print(f"使用指令: {args.instruction}")
            results = list(cosyvoice.inference_instruct(
                args.text, 
                args.instruction,
                args.speaker_id, 
                stream=args.stream
            ))
        else:
            # 使用标准模式
            results = list(cosyvoice.inference_sft(
                args.text, 
                args.speaker_id, 
                stream=args.stream
            ))
        
        if not results:
            print("警告: 未生成任何语音")
            return
            
        # 保存结果
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if len(results) == 1:
            # 单个结果
            torchaudio.save(args.output, results[0]['tts_speech'], cosyvoice.sample_rate)
            print(f"语音已保存至: {args.output}")
        else:
            # 多个结果（流式模式）
            base_name = os.path.splitext(args.output)[0]
            ext = os.path.splitext(args.output)[1]
            for i, result in enumerate(results):
                output_path = f"{base_name}_{i}{ext}"
                torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)
            print(f"语音已分段保存至: {base_name}_*{ext}")
            
            # 合并所有片段（可选）
            all_speech = torch.cat([r['tts_speech'] for r in results], dim=1)
            combined_path = f"{base_name}_combined{ext}"
            torchaudio.save(combined_path, all_speech, cosyvoice.sample_rate)
            print(f"合并后的语音已保存至: {combined_path}")
            
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
