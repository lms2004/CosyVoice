pip install "numpy<2" --upgrade --no-cache-dir
pip install onnxruntime==1.17.3 --no-cache-dir
# 如果要用 GPU 版本（且机器/驱动/CUDA 匹配），可用：
# pip install onnxruntime-gpu==1.17.3 --no-cache-dir