import json
import time
import uuid
import requests
import base64
import os
from dotenv import load_dotenv

# 读取 .env 环境变量
load_dotenv()

def _getenv_bool(name, default):
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")

# ================= 配置区（从 .env 读取） =================
CONFIG = {
    "appid": os.getenv("ASR_APPID", ""),
    "access_token": os.getenv("ASR_ACCESS_TOKEN", ""),
    "resource_id": os.getenv("ASR_RESOURCE_ID", "volc.bigasr.auc_turbo"),
    "model_name": os.getenv("ASR_MODEL_NAME", "bigmodel"),
    "enable_itn": _getenv_bool("ASR_ENABLE_ITN", True),
    "enable_punc": _getenv_bool("ASR_ENABLE_PUNC", True),
    "enable_ddc": _getenv_bool("ASR_ENABLE_DDC", True),
    "enable_speaker_info": _getenv_bool("ASR_ENABLE_SPK_INFO", False),
}
# 校验必要变量
if not CONFIG["appid"] or not CONFIG["access_token"]:
    raise RuntimeError("Missing ASR_APPID or ASR_ACCESS_TOKEN in .env")
# =======================================================

# 辅助函数：下载文件
def download_file(file_url):
    response = requests.get(file_url)
    if response.status_code == 200:
        return response.content  # 返回文件内容（二进制）
    else:
        raise Exception(f"下载失败，HTTP状态码: {response.status_code}")

# 辅助函数：将本地文件转换为Base64
def file_to_base64(file_path):
    with open(file_path, 'rb') as file:
        file_data = file.read()  # 读取文件内容
        base64_data = base64.b64encode(file_data).decode('utf-8')  # Base64 编码
    return base64_data

# recognize_task 函数
def recognize_task(file_url=None, file_path=None):
    recognize_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"

    headers = {
        "X-Api-App-Key": CONFIG["appid"],
        "X-Api-Access-Key": CONFIG["access_token"],
        "X-Api-Resource-Id": CONFIG["resource_id"], 
        "X-Api-Request-Id": str(uuid.uuid4()),
        "X-Api-Sequence": "-1", 
    }

    # 检查是使用文件URL还是直接上传数据
    audio_data = None
    if file_url:
        audio_data = {"url": file_url}
    elif file_path:
        base64_data = file_to_base64(file_path)
        audio_data = {"data": base64_data}

    if not audio_data:
        raise ValueError("必须提供 file_url 或 file_path 其中之一")

    request = {
        "user": {"uid": CONFIG["appid"]},
        "audio": audio_data,
        "request": {
            "model_name": CONFIG["model_name"],
            "enable_itn": CONFIG["enable_itn"],
            "enable_punc": CONFIG["enable_punc"],
            "enable_ddc": CONFIG["enable_ddc"],
            "enable_speaker_info": CONFIG["enable_speaker_info"],
        },
    }

    response = requests.post(recognize_url, json=request, headers=headers)
    
    if 'X-Api-Status-Code' in response.headers:
        print(f'recognize task response header X-Api-Status-Code: {response.headers["X-Api-Status-Code"]}')
        print(f'recognize task response header X-Api-Message: {response.headers["X-Api-Message"]}')
        print(time.asctime() + " recognize task response header X-Tt-Logid: {}".format(response.headers["X-Tt-Logid"]))
        print(f'recognize task response content is: {response.json()}\n')
    else:
        print(f'recognize task failed and the response headers are: {response.headers}\n')
        exit(1)

    return response

# recognizeMode 不变
def recognizeMode(file_url=None, file_path=None):
    start_time = time.time()
    print(time.asctime() + " START!")
    recognize_response = recognize_task(file_url=file_url, file_path=file_path)
    code = recognize_response.headers['X-Api-Status-Code']
    logid = recognize_response.headers['X-Tt-Logid']
    
    if code == '20000000':  # task finished
        with open("result.json", "w", encoding='utf-8') as f:
            f.write(json.dumps(recognize_response.json(), indent=4, ensure_ascii=False))
        print(time.asctime() + " SUCCESS! \n")
        print(f"程序运行耗时: {time.time() - start_time:.6f} 秒")
    elif code != '20000001' and code != '20000002':  # task failed
        print(time.asctime() + " FAILED! code: {}, logid: {}".format(code, logid))
        print("headers:")
        # print(query_response.content)

def main(): 
    # 示例：通过 URL 或 文件路径选择传入参数
    file_url = "https://example.mp3"
    file_path = "/mnt/c/Users/lms/Desktop/CosyVoice/asset/mp3/test.mp3"
    recognizeMode(file_path=file_path)  # 或者 recognizeMode(file_path=file_path)
 
if __name__ == '__main__': 
    main()
