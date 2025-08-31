#!/usr/bin/env python3
import argparse
import os
import glob
from tqdm import tqdm
import librosa
import soundfile as sf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, help='源数据目录，包含mp3音频文件')
    parser.add_argument('--des_dir', type=str, help='目标数据目录')
    parser.add_argument('--speaker_id', type=str, default='custom_speaker', help='说话人ID')
    parser.add_argument('--transcript_dir', type=str, help='文本转录目录，如果与音频不在同一目录')
    args = parser.parse_args()
    
    # 创建目标目录
    os.makedirs(args.des_dir, exist_ok=True)
    
    # 获取音频文件（同时支持 mp3 与 wav）
    mp3_files = glob.glob(os.path.join(args.src_dir, '*.mp3'))
    wav_files = glob.glob(os.path.join(args.src_dir, '*.wav'))
    audio_files = sorted(mp3_files + wav_files)
    
    # 如果没有指定文本目录，默认使用源目录
    transcript_dir = args.transcript_dir if args.transcript_dir else args.src_dir
    
    utt2wav, utt2text, utt2spk = {}, {}, {}
    spk2utt = {args.speaker_id: []}
    
    # 创建临时目录存储转换后的WAV文件
    wav_dir = os.path.join(args.des_dir, 'wavs')
    os.makedirs(wav_dir, exist_ok=True)
    
    print(f"处理 {len(audio_files)} 个音频文件...")
    for src_path in tqdm(audio_files):
        # 获取文件名（不含扩展名）
        basename = os.path.basename(src_path)
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
        
        # 统一转换/重采样到 WAV (16kHz, 单声道)
        wav_path = os.path.join(wav_dir, f"{utt}.wav")
        try:
            # 使用 librosa 读入，强制单声道、16kHz
            audio, sr = librosa.load(src_path, sr=16000, mono=True)
            # 保存为WAV
            sf.write(wav_path, audio, 16000)
            
            # 添加到字典
            utt2wav[utt_id] = os.path.abspath(wav_path)
            utt2text[utt_id] = content
            utt2spk[utt_id] = args.speaker_id
            spk2utt[args.speaker_id].append(utt_id)
            
        except Exception as e:
            print(f"处理文件 {src_path} 时出错: {e}")
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
