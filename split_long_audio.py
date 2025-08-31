#!/usr/bin/env python3
import argparse
import os
import glob
import json
import numpy as np
import torch
import torchaudio
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
import shutil

def convert_mp3_to_wav(mp3_path, output_dir=None, target_sr=16000):
    """将MP3文件转换为WAV格式"""
    try:
        # 加载MP3文件
        waveform, sample_rate = torchaudio.load(mp3_path)
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 重采样到目标采样率
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # 如果指定了输出目录，则保存为WAV文件
        if output_dir:
            basename = os.path.basename(mp3_path)
            wav_name = os.path.splitext(basename)[0] + ".wav"
            wav_path = os.path.join(output_dir, wav_name)
            sf.write(wav_path, waveform.squeeze().numpy(), target_sr)
            return wav_path
        else:
            return waveform, target_sr
            
    except Exception as e:
        print(f"转换文件 {mp3_path} 时出错: {e}")
        return None

def split_audio_by_silence(audio_path, output_dir, min_segment_length=5, max_segment_length=15, 
                          silence_thresh=-40, min_silence_len=500, keep_silence=300):
    """
    根据静音将音频文件拆分为多个片段
    
    参数:
        audio_path: 音频文件路径
        output_dir: 输出目录
        min_segment_length: 最小片段长度(秒)
        max_segment_length: 最大片段长度(秒)
        silence_thresh: 静音阈值(dB)，越小越严格
        min_silence_len: 最小静音长度(毫秒)
        keep_silence: 保留的静音长度(毫秒)
    """
    try:
        # 加载音频文件
        audio = AudioSegment.from_file(audio_path)
        
        # 获取文件名(不含扩展名)
        basename = os.path.basename(audio_path)
        filename = os.path.splitext(basename)[0]
        
        # 检测非静音片段
        print(f"检测 {basename} 中的非静音片段...")
        nonsilent_chunks = detect_nonsilent(audio, 
                                           min_silence_len=min_silence_len, 
                                           silence_thresh=silence_thresh)
        
        # 如果没有检测到非静音片段，则返回
        if not nonsilent_chunks:
            print(f"警告: 在 {basename} 中未检测到非静音片段")
            return []
        
        # 将检测到的片段转换为AudioSegment对象
        chunks = []
        for i, (start_ms, end_ms) in enumerate(nonsilent_chunks):
            # 添加前后静音
            start_with_silence = max(0, start_ms - keep_silence)
            end_with_silence = min(len(audio), end_ms + keep_silence)
            
            # 提取片段
            chunk = audio[start_with_silence:end_with_silence]
            
            # 检查片段长度
            chunk_length_sec = len(chunk) / 1000
            
            # 如果片段太短，跳过
            if chunk_length_sec < min_segment_length:
                continue
                
            # 如果片段太长，进一步拆分
            if chunk_length_sec > max_segment_length:
                # 计算需要拆分的子片段数
                num_subchunks = int(np.ceil(chunk_length_sec / max_segment_length))
                subchunk_length_ms = len(chunk) // num_subchunks
                
                # 拆分为多个子片段
                for j in range(num_subchunks):
                    start_ms_sub = j * subchunk_length_ms
                    end_ms_sub = min((j + 1) * subchunk_length_ms, len(chunk))
                    subchunk = chunk[start_ms_sub:end_ms_sub]
                    
                    # 只保留长度足够的子片段
                    if len(subchunk) / 1000 >= min_segment_length:
                        chunks.append((subchunk, f"{filename}_{i:03d}_{j:03d}"))
            else:
                chunks.append((chunk, f"{filename}_{i:03d}"))
        
        # 保存片段
        output_paths = []
        for chunk, chunk_name in chunks:
            output_path = os.path.join(output_dir, f"{chunk_name}.wav")
            chunk.export(output_path, format="wav")
            output_paths.append(output_path)
            
        print(f"已将 {basename} 拆分为 {len(chunks)} 个片段")
        return output_paths
        
    except Exception as e:
        print(f"拆分文件 {audio_path} 时出错: {e}")
        return []

def generate_transcripts_placeholder(output_paths, transcript_dir):
    """为拆分后的音频生成占位符文本文件"""
    os.makedirs(transcript_dir, exist_ok=True)
    
    for audio_path in output_paths:
        basename = os.path.basename(audio_path)
        filename = os.path.splitext(basename)[0]
        transcript_path = os.path.join(transcript_dir, f"{filename}.txt")
        
        # 创建空文本文件
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write("请在此处填写文本转录")
            
    print(f"已在 {transcript_dir} 中生成 {len(output_paths)} 个文本文件占位符")

def main():
    parser = argparse.ArgumentParser(description='拆分长音频文件为短片段')
    parser.add_argument('--src_dir', type=str, required=True, help='源MP3文件目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--min_length', type=float, default=5.0, help='最小片段长度(秒)')
    parser.add_argument('--max_length', type=float, default=15.0, help='最大片段长度(秒)')
    parser.add_argument('--silence_thresh', type=float, default=-40, help='静音阈值(dB)，越小越严格')
    parser.add_argument('--min_silence_len', type=int, default=500, help='最小静音长度(毫秒)')
    parser.add_argument('--keep_silence', type=int, default=300, help='保留的静音长度(毫秒)')
    parser.add_argument('--generate_transcripts', action='store_true', help='生成文本转录占位符')
    args = parser.parse_args()
    
    # 创建输出目录
    wavs_dir = os.path.join(args.output_dir, 'wavs')
    transcripts_dir = os.path.join(args.output_dir, 'transcripts')
    os.makedirs(wavs_dir, exist_ok=True)
    
    # 获取所有MP3文件
    mp3_files = glob.glob(os.path.join(args.src_dir, '*.mp3'))
    if not mp3_files:
        print(f"错误: 在 {args.src_dir} 中未找到MP3文件")
        return
        
    print(f"找到 {len(mp3_files)} 个MP3文件")
    
    # 处理每个MP3文件
    all_output_paths = []
    for mp3_path in tqdm(mp3_files, desc="处理MP3文件"):
        # 首先转换为WAV格式
        temp_wav_path = convert_mp3_to_wav(mp3_path, output_dir=os.path.join(args.output_dir, 'temp'))
        
        if temp_wav_path:
            # 然后拆分为短片段
            output_paths = split_audio_by_silence(
                temp_wav_path, 
                wavs_dir,
                min_segment_length=args.min_length,
                max_segment_length=args.max_length,
                silence_thresh=args.silence_thresh,
                min_silence_len=args.min_silence_len,
                keep_silence=args.keep_silence
            )
            all_output_paths.extend(output_paths)
    
    # 清理临时文件
    temp_dir = os.path.join(args.output_dir, 'temp')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # 生成文本转录占位符
    if args.generate_transcripts and all_output_paths:
        generate_transcripts_placeholder(all_output_paths, transcripts_dir)
    
    print(f"处理完成! 共生成 {len(all_output_paths)} 个音频片段")
    print(f"音频片段保存在: {wavs_dir}")
    if args.generate_transcripts:
        print(f"文本转录占位符保存在: {transcripts_dir}")
        print("请编辑这些文本文件，填写正确的转录内容")

if __name__ == "__main__":
    main()
