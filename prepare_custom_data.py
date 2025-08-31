#!/usr/bin/env python3
import argparse
import os
import glob
from tqdm import tqdm
import torchaudio
import torch
import soundfile as sf
import tempfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, help='源数据目录，包含mp3音频文件')
    parser.add_argument('--des_dir', type=str, help='目标数据目录')
    parser.add_argument('--speaker_id', type=str, default='custom_speaker', help='说话人ID')
    parser.add_argument('--transcript_dir', type=str, help='文本转录目录，如果与音频不在同一目录')
    args = parser.parse_args()
    
    # 创建目标目录
    os.makedirs(args.des_dir, exist_ok=True)
    
    # 获取所有mp3文件
    mp3_files = glob.glob(os.path.join(args.src_dir, '*.mp3'))
    
    # 如果没有指定文本目录，默认使用源目录
    transcript_dir = args.transcript_dir if args.transcript_dir else args.src_dir
    
    utt2wav, utt2text, utt2spk = {}, {}, {}
    spk2utt = {args.speaker_id: []}
    
    # 创建临时目录存储转换后的WAV文件
    wav_dir = os.path.join(args.des_dir, 'wavs')
    os.makedirs(wav_dir, exist_ok=True)
    
    print(f"处理 {len(mp3_files)} 个MP3文件...")
    for mp3_path in tqdm(mp3_files):
        # 获取文件名（不含扩展名）
        basename = os.path.basename(mp3_path)
        utt = os.path.splitext(basename)[0]
        utt_id = f"{args.speaker_id}_{utt}"  # 添加说话人前缀，确保唯一ID
        
        # 查找对应的文本文件
        txt_path = os.path.join(transcript_dir, f'{utt}.txt')
        if not os.path.exists(txt_path):
            print(f'警告: {txt_path} 不存在，跳过此音频')
            continue
        
        # 读取文本内容
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 转换MP3到WAV (16kHz, 单声道)
        wav_path = os.path.join(wav_dir, f"{utt}.wav")
        try:
            # 加载MP3文件
            waveform, sample_rate = torchaudio.load(mp3_path)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 重采样到16kHz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # 保存为WAV
            sf.write(wav_path, waveform.squeeze().numpy(), 16000)
            
            # 添加到字典
            utt2wav[utt_id] = os.path.abspath(wav_path)
            utt2text[utt_id] = content
            utt2spk[utt_id] = args.speaker_id
            spk2utt[args.speaker_id].append(utt_id)
            
        except Exception as e:
            print(f"处理文件 {mp3_path} 时出错: {e}")
            continue
    
    # 写入文件
    with open(f'{args.des_dir}/wav.scp', 'w', encoding='utf-8') as f:
        for k, v in utt2wav.items():
            f.write(f'{k} {v}\n')
    
    with open(f'{args.des_dir}/text', 'w', encoding='utf-8') as f:
        for k, v in utt2text.items():
            f.write(f'{k} {v}\n')
    
    with open(f'{args.des_dir}/utt2spk', 'w', encoding='utf-8') as f:
        for k, v in utt2spk.items():
            f.write(f'{k} {v}\n')
    
    with open(f'{args.des_dir}/spk2utt', 'w', encoding='utf-8') as f:
        for k, v in spk2utt.items():
            f.write(f'{k} {" ".join(v)}\n')
    
    print(f'处理完成，共处理 {len(utt2wav)} 个音频文件')
    print(f'转换后的WAV文件保存在: {wav_dir}')
    print(f'Kaldi格式数据文件保存在: {args.des_dir}')

if __name__ == "__main__":
    main()
