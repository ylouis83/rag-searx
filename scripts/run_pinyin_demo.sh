#!/bin/bash

# 拼音向量搜索演示启动脚本
# 用于演示解决语音识别中发音差异导致的搜索问题

set -e

# 获取脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🎯 拼音向量搜索演示"
echo "==============================================="
echo "项目目录: $PROJECT_ROOT"
echo "脚本目录: $SCRIPT_DIR"
echo

# 检查是否在项目根目录
cd "$PROJECT_ROOT"

# 检查虚拟环境
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ 检测到虚拟环境: $VIRTUAL_ENV"
else
    echo "⚠️  警告: 未检测到虚拟环境"
    echo "建议激活虚拟环境: source venv/bin/activate"
fi

# 检查依赖
echo "🔍 检查Python依赖..."

# 检查关键依赖
python -c "import pypinyin" 2>/dev/null || {
    echo "❌ 缺少 pypinyin 依赖"
    echo "正在安装依赖..."
    pip install pypinyin==0.51.0
}

python -c "import sentence_transformers" 2>/dev/null || {
    echo "❌ 缺少 sentence-transformers 依赖"
    echo "请安装依赖: pip install -r requirements.txt"
    exit 1
}

echo "✅ 依赖检查完成"
echo

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 运行演示
echo "🚀 启动拼音向量搜索演示..."
echo "==============================================="

python demos/pinyin_vector_search_demo.py

echo
echo "==============================================="
echo "📝 演示说明:"
echo "   - 该演示展示了如何解决语音识别中的发音差异问题"
echo "   - 例如：'周萍' vs '邹萍' 可以统一检索"
echo "   - 使用BGE嵌入模型 + 拼音向量化技术"
echo "   - 支持多种匹配模式：精确匹配、拼音向量匹配、模糊匹配"
echo "===============================================" 