# ベースイメージの指定
FROM python:3.8

# OSパッケージの更新と必要なライブラリのインストール
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# gdownのインストール
RUN pip install gdown

# 作業ディレクトリの設定
WORKDIR /app

# Google Driveからモデルファイルをダウンロード
RUN gdown --id '1TxLPwCDtDkvYCpGXk34uKvb7TJalzJZN' -O model/tooth_and_plaque.pth && \
    # ファイルの存在とサイズを確認
    if [ -f model/tooth_and_plaque.pth ]; then \
        echo "tooth_and_plaque.pth has been successfully downloaded."; \
        ls -lh model/tooth_and_plaque.pth; \
        # ファイルの最初の数行を表示（バイナリファイルの場合はこのステップをスキップ）
        head -n 10 model/tooth_and_plaque.pth; \
    else \
        echo "Error: Failed to download tooth_and_plaque.pth."; \
        exit 1; \
    fi

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

https://drive.google.com/file/d/1TxLPwCDtDkvYCpGXk34uKvb7TJalzJZN/view?usp=sharing