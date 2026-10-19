# CUDA 12 + torch 2.7 (VoxCPM needs torch >= 2.5, Python 3.10-3.12).
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV HF_HOME=/models \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the weights so a cold worker never downloads them. snapshot_download, not
# from_pretrained: the build runner has no GPU.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('openbmb/VoxCPM2')"

ENV HF_HUB_OFFLINE=1

COPY handler.py .
CMD ["python", "-u", "handler.py"]
