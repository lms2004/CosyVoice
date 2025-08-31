#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import concurrent.futures as futures
from functools import partial

# 复用现有 asr.py 的配置与调用
import asr


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def extract_text_from_response(resp_json):
    """从 asr API 返回中提取文本，优先使用 result.text，其次拼接 utterances。"""
    try:
        result = resp_json.get('result') or {}
        text = (result.get('text') or '').strip()
        if text:
            return text
        utts = result.get('utterances') or []
        if utts:
            return ''.join((u.get('text') or '') for u in utts).strip()
        return ''
    except Exception:
        return ''


def asr_one(file_path, out_txt_path, max_retries=3, backoff=1.5, overwrite=False):
    """对单文件进行 ASR，保存到 out_txt_path。返回 (wav, ok, msg)。"""
    if (not overwrite) and os.path.exists(out_txt_path):
        return file_path, True, 'skip_exists'

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = asr.recognize_task(file_path=file_path)
            # 解析 header 状态
            code = resp.headers.get('X-Api-Status-Code', '')
            if code and code != '20000000':
                msg = resp.headers.get('X-Api-Message', 'unknown')
                raise RuntimeError(f'api_status_code={code}, msg={msg}')
            data = resp.json()
            text = extract_text_from_response(data)
            # 允许空字符串，但写入文件，后续可人工修订
            ensure_dir(os.path.dirname(out_txt_path))
            with open(out_txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            return file_path, True, 'ok'
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                sleep_s = backoff ** (attempt - 1)
                time.sleep(sleep_s)
            else:
                break
    return file_path, False, f'failed: {last_exc}'


def main():
    parser = argparse.ArgumentParser(description='批量调用 ASR 并生成逐文件转写 .txt')
    parser.add_argument('--wav_dir', type=str, required=True, help='切分后 WAV 目录，如 data/splits/wavs')
    parser.add_argument('--out_dir', type=str, required=True, help='输出转写目录，如 data/splits/transcripts')
    parser.add_argument('--num_workers', type=int, default=2, help='并发工作线程数（建议小一些避免限流）')
    parser.add_argument('--overwrite', action='store_true', help='若已存在 .txt 是否覆盖')
    parser.add_argument('--max_retries', type=int, default=3, help='失败重试次数')
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # 收集待处理 wav 列表
    wavs = [
        os.path.join(args.wav_dir, n) for n in sorted(os.listdir(args.wav_dir))
        if n.lower().endswith('.wav')
    ]
    if not wavs:
        print(f'未在 {args.wav_dir} 找到 WAV 文件', file=sys.stderr)
        sys.exit(1)

    print(f'共 {len(wavs)} 个切片，开始批量 ASR，num_workers={args.num_workers}')

    worker = partial(
        asr_one,
        max_retries=args.max_retries,
        backoff=1.6,
        overwrite=args.overwrite,
    )

    ok_cnt, fail_cnt, skip_cnt = 0, 0, 0

    if args.num_workers <= 1:
        for wav in wavs:
            base = os.path.splitext(os.path.basename(wav))[0]
            out_txt = os.path.join(args.out_dir, base + '.txt')
            _, ok, msg = worker(wav, out_txt)
            if ok and msg == 'ok':
                ok_cnt += 1
            elif ok and msg == 'skip_exists':
                skip_cnt += 1
            else:
                fail_cnt += 1
                print(f'[FAIL] {wav} -> {msg}', file=sys.stderr)
    else:
        with futures.ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futs = []
            for wav in wavs:
                base = os.path.splitext(os.path.basename(wav))[0]
                out_txt = os.path.join(args.out_dir, base + '.txt')
                futs.append(ex.submit(worker, wav, out_txt))
            for fu in futures.as_completed(futs):
                wav, ok, msg = fu.result()
                if ok and msg == 'ok':
                    ok_cnt += 1
                elif ok and msg == 'skip_exists':
                    skip_cnt += 1
                else:
                    fail_cnt += 1
                    print(f'[FAIL] {wav} -> {msg}', file=sys.stderr)

    print(f'完成：成功 {ok_cnt}，跳过 {skip_cnt}，失败 {fail_cnt}')
    if fail_cnt > 0:
        sys.exit(2)


if __name__ == '__main__':
    main()
