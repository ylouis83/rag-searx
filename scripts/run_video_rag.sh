#!/bin/bash

# 视频RAG演示系统
# YouTube视频处理和RAG检索演示

# 定义颜色和格式
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# 定义分隔线
SEPARATOR="=================================================="
THIN_SEPARATOR="--------------------------------------------------"

# 清屏函数
clear_screen() {
    clear
}

# 打印标题
print_title() {
    echo -e "${BOLD}${CYAN}${SEPARATOR}${NC}"
    echo -e "${BOLD}${WHITE}           🎬 视频RAG演示系统 🚀           ${NC}"
    echo -e "${CYAN}        YouTube视频处理和RAG检索演示${NC}"
    echo -e "${BOLD}${CYAN}${SEPARATOR}${NC}"
    echo ""
}

# 打印状态信息
print_status() {
    local status="$1"
    local message="$2"
    case $status in
        "info")
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
        "success")
            echo -e "${GREEN}[✓]${NC} $message"
            ;;
        "warning")
            echo -e "${YELLOW}[!]${NC} $message"
            ;;
        "error")
            echo -e "${RED}[✗]${NC} $message"
            ;;
    esac
}

# 清屏并显示标题
clear_screen
print_title

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 验证文件存在
if [ ! -f "$PROJECT_ROOT/demos/video_rag_demo.py" ]; then
    print_status "error" "未找到 demos/video_rag_demo.py"
    print_status "error" "请确保在 rag-searx 项目根目录下运行此脚本"
    exit 1
fi

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 检查Python环境
echo -e "${BOLD}${PURPLE}🔍 环境检查${NC}"
echo -e "${THIN_SEPARATOR}"

if [ -d "venv" ]; then
    source venv/bin/activate
    print_status "success" "已激活本地虚拟环境"
elif [ -d "../.venv" ]; then
    source ../.venv/bin/activate
    print_status "success" "已激活父目录虚拟环境"
elif [ ! -z "$VIRTUAL_ENV" ]; then
    print_status "success" "已在虚拟环境中: $(basename $VIRTUAL_ENV)"
else
    print_status "warning" "未找到虚拟环境，使用系统Python"
fi

echo ""

# 显示功能菜单
show_menu() {
    echo -e "${BOLD}${GREEN}📋 可用功能${NC}"
    echo -e "${THIN_SEPARATOR}"
    echo -e "${WHITE} 1.${NC} ${BOLD}完整演示${NC}      - Schema构建 + 文档处理 + 向量化 + 检索 + 问答"
    echo -e "${WHITE} 2.${NC} ${BOLD}Schema构建${NC}    - 数据库结构初始化"
    echo -e "${WHITE} 3.${NC} ${BOLD}问答系统${NC}      - RAG问答演示"
    echo -e "${WHITE} 4.${NC} ${BOLD}提示词测试${NC}    - 提示词优化测试"
    echo -e "${WHITE} 5.${NC} ${BOLD}交互模式${NC}      - 实时问答体验"
    echo -e "${WHITE} 6.${NC} ${BOLD}多路召回演示${NC}  - 向量+关键词+语义检索"
    echo -e "${WHITE} 7.${NC} ${BOLD}多路对比分析${NC}  - 召回策略对比"
    echo -e "${WHITE} 8.${NC} ${BOLD}多路交互模式${NC}  - 多路召回实时体验"
    echo -e "${THIN_SEPARATOR}"
}

# 处理命令行参数
if [ $# -eq 0 ]; then
    show_menu
    echo ""
    echo -e "${CYAN}💡 使用提示:${NC}"
    echo -e "   ${WHITE}./run_video_rag.sh ${YELLOW}<mode>${NC}  (直接运行指定模式)"
    echo -e "   ${WHITE}./run_video_rag.sh ${YELLOW}full${NC}     (完整演示)"
    echo -e "   ${WHITE}./run_video_rag.sh ${YELLOW}interactive${NC} (交互模式)"
    echo ""
    
    read -p "$(echo -e ${CYAN}请输入选择 [1-8]:${NC} )" choice
    
    case $choice in
        1) mode="full" ;;
        2) mode="schema" ;;
        3) mode="qa" ;;
        4) mode="prompt" ;;
        5) mode="interactive" ;;
        6) mode="multi_recall" ;;
        7) mode="multi_comparison" ;;
        8) mode="multi_interactive" ;;
        *) 
            print_status "warning" "使用默认模式: 完整演示"
            mode="full" 
            ;;
    esac
else
    # 支持中文参数输入
    input_mode="$1"
    case $input_mode in
        "完整演示"|"full"|"Full"|"FULL")
            mode="full"
            ;;
        "Schema构建"|"schema"|"Schema"|"SCHEMA")
            mode="schema"
            ;;
        "问答系统"|"qa"|"QA"|"问答"|"question")
            mode="qa"
            ;;
        "提示词测试"|"prompt"|"Prompt"|"提示词"|"prompt_test")
            mode="prompt"
            ;;
        "交互模式"|"interactive"|"Interactive"|"交互"|"互动")
            mode="interactive"
            ;;
        "多路召回演示"|"multi_recall"|"多路召回"|"召回演示"|"multi-recall")
            mode="multi_recall"
            ;;
        "多路对比分析"|"multi_comparison"|"对比分析"|"comparison"|"多路对比")
            mode="multi_comparison"
            ;;
        "多路交互模式"|"multi_interactive"|"多路交互"|"multi-interactive")
            mode="multi_interactive"
            ;;
        *)
            print_status "warning" "未识别的参数: $input_mode，使用默认模式"
            mode="full"
            ;;
    esac
fi

# 显示运行信息
echo ""
echo -e "${BOLD}${PURPLE}🚀 启动参数${NC}"
echo -e "${THIN_SEPARATOR}"
echo -e "${WHITE}演示模式:${NC} ${BOLD}${GREEN}$mode${NC}"
echo -e "${WHITE}技术栈:${NC}   RAGFlow + BGE + Milvus + Qwen + 多路召回"
echo ""

# 显示启动提示
echo -e "${BOLD}${YELLOW}⏳ 正在启动演示...${NC}"
echo -e "${THIN_SEPARATOR}"

# 运行对应的演示
case $mode in
    "multi_recall")
        print_status "info" "启动多路召回演示..."
        echo ""
        python demos/multi_recall_demo.py full
        ;;
    "multi_comparison") 
        print_status "info" "启动多路对比分析..."
        echo ""
        python demos/multi_recall_demo.py comparison
        ;;
    "multi_interactive")
        print_status "info" "启动多路交互模式..."
        echo ""
        python demos/multi_recall_demo.py interactive
        ;;
    *)
        print_status "info" "启动视频RAG演示..."
        echo ""
        python demos/video_rag_demo.py $mode
        ;;
esac

# 结束提示
echo ""
echo -e "${BOLD}${CYAN}${SEPARATOR}${NC}"
echo -e "${BOLD}${GREEN}✨ 视频RAG演示完成! ✨${NC}" 
echo -e "${BOLD}${CYAN}${SEPARATOR}${NC}" 