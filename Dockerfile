# ベースイメージの指定
FROM python:3.8

# OSパッケージの更新と必要なライブラリのインストール
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /app

# 依存関係ファイルのコピー
COPY requirements.txt /app/

# 依存関係のインストール
# 先にPyTorchをインストールする行を追加
RUN pip install torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0 -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install -r requirements.txt \
    && pip install git+https://github.com/facebookresearch/detectron2.git

# アプリケーションコードのコピー
COPY . /app

# コンテナのポート公開
EXPOSE 8080

# アプリケーションの起動コマンド
CMD ["gunicorn", "-b", ":8080", "main:app"]
