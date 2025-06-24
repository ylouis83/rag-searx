#!/bin/bash

# RAGFlow + BGE RAGFlow BGE
# RAGFlow + BGE-large-zh-v1.5集成演示

echo "启动 RAGFlow + BGE演示"
echo ""

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ ! -f "$PROJECT_ROOT/demos/ragflow_bge_demo.py" ]; then
    echo "错误: 未找到 demos/ragflow_bge_demo.py"
    echo "请确保在 rag-searx 项目根目录下运行此脚本"
    exit 1
fi

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 检查Python环境
echo "检查 Python环境..."
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "已激活本地虚拟环境"
elif [ -d "../.venv" ]; then
    source ../.venv/bin/activate
    echo "已激活父目录虚拟环境"
elif [ ! -z "$VIRTUAL_ENV" ]; then
    echo "已在虚拟环境中: $VIRTUAL_ENV"
else
    echo "警告: 未找到虚拟环境，使用系统Python"
fi

echo ""
echo "RAGFlow + BGE演示系统功能:"
echo "   1. 完整演示 (文档处理 + 向量化 + 检索 + 问答)"
echo "   2. 文档处理"
echo "   3. 问答系统"
echo "   4. 文档分块"
echo "   5. BGE嵌入"
echo "   6. 交互模式"
echo ""

# 处理命令行参数
if [ $# -eq 0 ]; then
    echo "请选择演示模式:"
    echo "   ./run_ragflow_bge.sh full      # 完整演示"
    echo "   ./run_ragflow_bge.sh process   # 文档处理"
    echo "   ./run_ragflow_bge.sh qa        # 问答系统"
    echo "   ./run_ragflow_bge.sh chunk     # 文档分块"
    echo "   ./run_ragflow_bge.sh embed     # BGE嵌入"
    echo "   ./run_ragflow_bge.sh interactive # 交互模式"
    echo ""
    
    read -p "请输入选择 (1-6): " choice
    
    case $choice in
        1) mode="full" ;;
        2) mode="process" ;;
        3) mode="qa" ;;
        4) mode="chunk" ;;
        5) mode="embed" ;;
        6) mode="interactive" ;;
        *) echo "使用默认模式"; mode="full" ;;
    esac
else
    mode="$1"
fi

echo ""
echo "演示模式: $mode"
echo "技术栈: RAGFlow + BGE-large-zh-v1.5 + Milvus + Qwen-Plus"
echo ""

# 运行RAGFlow + BGE演示
python demos/ragflow_bge_demo.py $mode

echo ""
echo "RAGFlow + BGE演示完成!" 