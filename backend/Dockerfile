# InferEdge backend — CPU image (mock/nim modes).
# For GPU-accelerated routing, build on an NVIDIA base image instead and
# add cupy-cuda12x (see README "Running with a GPU"):
#   FROM nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM python:3.11-slim

WORKDIR /srv/inferedge

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV INFERENCE_MODE=mock \
    INFEREDGE_DB_PATH=/srv/inferedge/data/inferedge.db

RUN mkdir -p /srv/inferedge/data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
