FROM tensorflow/tensorflow:2.12.0-gpu

WORKDIR /app

RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install -U jaxlib==0.4.7+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install git+https://github.com/sanchit-gandhi/whisper-jax.git

COPY main.py .
ENTRYPOINT ["python", "main.py"]