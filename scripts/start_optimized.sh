#!/bin/bash

# RAG-Searx 内存优化启动脚本
# 提供多种内存配置选项，从4GB降低到512MB以下

echo "RAG-Searx 内存优化启动脚本"
echo "=========================================="
echo ""

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo "错误: Docker未运行，请先启动Docker"
    exit 1
fi

echo "请选择内存配置模式:"
echo ""
echo "1. 超轻量级模式 (约512MB)"
echo "   - 仅API服务 + 内存向量存储"
echo "   - 适合: 开发测试、小规模数据"
echo ""
echo "2. 轻量级模式 (约1.5GB)"  
echo "   - ChromaDB向量数据库"
echo "   - 适合: 中小规模生产环境"
echo ""
echo "3. 内存优化模式 (约2.5GB)"
echo "   - Milvus + 优化配置"
echo "   - 适合: 生产环境，需要高性能"
echo ""
echo "4. 原始完整模式 (约4GB)"
echo "   - 完整Milvus集群"
echo "   - 适合: 大规模生产环境"
echo ""

read -p "请选择模式 (1-4): " choice

case $choice in
    1)
        echo "启动超轻量级模式..."
        MODE="lite"
        COMPOSE_FILE="docker-compose.lite.yml"
        ;;
    2)
        echo "启动轻量级模式 (ChromaDB)..."
        MODE="chromadb"
        COMPOSE_FILE="docker-compose.lite.yml"
        ;;
    3)
        echo "启动内存优化模式..."
        MODE="minimal"
        COMPOSE_FILE="docker-compose.minimal.yml"
        ;;
    4)
        echo "启动原始完整模式..."
        MODE="full"
        COMPOSE_FILE="docker-compose.yml"
        ;;
    *)
        echo "无效选择，使用超轻量级模式"
        MODE="lite"
        COMPOSE_FILE="docker-compose.lite.yml"
        ;;
esac

echo ""
echo "配置模式: $MODE"
echo "使用配置文件: $COMPOSE_FILE"

# 停止现有容器
echo "停止现有容器..."
docker-compose down > /dev/null 2>&1

# 创建必要的目录
mkdir -p data logs volumes/{milvus,etcd,minio,chromadb}

case $MODE in
    "lite")
        echo "启动超轻量级模式 (仅内存存储)..."
        if [ -f "$COMPOSE_FILE" ]; then
            docker-compose -f $COMPOSE_FILE up -d rag-searx-api
        else
            echo "配置文件 $COMPOSE_FILE 不存在，使用原始配置"
            docker-compose up -d rag-searx-api
        fi
        ;;
    "chromadb")
        echo "启动ChromaDB模式..."
        if [ -f "$COMPOSE_FILE" ]; then
            docker-compose -f $COMPOSE_FILE --profile chromadb up -d
        else
            echo "配置文件 $COMPOSE_FILE 不存在，使用原始配置"
            docker-compose up -d
        fi
        ;;
    "minimal")
        echo "启动Milvus内存优化模式..."
        if [ -f "$COMPOSE_FILE" ]; then
            docker-compose -f $COMPOSE_FILE up -d
        else
            echo "配置文件 $COMPOSE_FILE 不存在，使用原始配置"
            docker-compose up -d
        fi
        ;;
    "full")
        echo "启动完整模式..."
        docker-compose up -d
        ;;
esac

echo ""
echo "启动完成! 内存使用预期: "
case $MODE in
    "lite") echo "  ~512MB (超轻量级)" ;;
    "chromadb") echo "  ~1.5GB (轻量级)" ;;
    "minimal") echo "  ~2.5GB (内存优化)" ;;
    "full") echo "  ~4GB (完整模式)" ;;
esac

echo ""
echo "访问地址: http://localhost:8000"
echo "API文档: http://localhost:8000/docs"
echo ""
echo "监控内存使用: ./scripts/monitor_memory.sh"
