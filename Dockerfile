# 中文临床指南问答服务 —— 生产镜像
#
# 说明两点取舍：
# 1. 基础镜像用 slim 而不是 alpine：faiss-cpu / torch 的 wheel 在 glibc 下最省事。
# 2. 索引文件（data/index，几百 MB）不 COPY 进镜像，改为运行时挂载。
#    这样换语料不用重新构建镜像——这是 RAG 服务最常见的运维动作。

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    SERVICE_CACHE_MAX=256

WORKDIR /app

# 依赖单独一层，改代码时不重装 torch
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# 非 root 运行
RUN useradd -m -u 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
