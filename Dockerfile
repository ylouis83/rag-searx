# RAG-SearX Dockerfile
# 基于Python 3.11官方镜像构建

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 更换为中国大陆源
RUN echo "deb https://mirrors.aliyun.com/debian/ bookworm main non-free non-free-firmware contrib" > /etc/apt/sources.list && \
    echo "deb-src https://mirrors.aliyun.com/debian/ bookworm main non-free non-free-firmware contrib" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian-security/ bookworm-security main" >> /etc/apt/sources.list && \
    echo "deb-src https://mirrors.aliyun.com/debian-security/ bookworm-security main" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian/ bookworm-updates main non-free non-free-firmware contrib" >> /etc/apt/sources.list && \
    echo "deb-src https://mirrors.aliyun.com/debian/ bookworm-updates main non-free non-free-firmware contrib" >> /etc/apt/sources.list

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 创建必要的目录
RUN mkdir -p /app/data /app/logs /app/config

# 复制依赖文件
COPY requirements.txt .

# 配置pip使用中国源并安装Python依赖
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 复制环境配置文件
COPY env.example .env

# 设置权限
RUN chmod +x /app && \
    chmod -R 755 /app/data && \
    chmod -R 755 /app/logs

# 创建非root用户
RUN groupadd -r raguser && useradd -r -g raguser raguser
RUN chown -R raguser:raguser /app
USER raguser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 