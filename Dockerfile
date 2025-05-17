FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 必要パッケージのインストール
RUN apt update && apt install -y \
    git cmake build-essential python3 python3-pip curl wget unzip

# llama.cpp クローンとCUDAビルド
RUN git clone https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp
WORKDIR /opt/llama.cpp
RUN mkdir build && cd build && cmake .. -DGGML_CUDA=on && make -j

# llama-cpp-python のCUDAビルドインストール
ENV CMAKE_ARGS="-DGGML_CUDA=on -DLLAMA_CMAKE_DIR=/opt/llama.cpp"
RUN pip3 install --upgrade pip
RUN pip3 install llama-cpp-python flask

# アプリケーション配置
WORKDIR /app
COPY . /app

# モデルディレクトリを明示
RUN mkdir -p /app/models

EXPOSE 8888
CMD ["python3", "app.py"]
