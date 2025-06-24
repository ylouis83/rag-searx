#!/bin/bash

# Docker容器内存监控脚本
# 实时显示RAG-Searx相关容器的内存使用情况

echo "RAG-Searx 内存使用监控"
echo "======================================"
echo "按 Ctrl+C 退出监控"
echo ""

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo "错误: Docker未运行"
    exit 1
fi

# 获取相关容器
get_rag_containers() {
    docker ps --format "{{.Names}}" | grep -E "(rag-searx|milvus|etcd|minio|chroma|qdrant)" | sort
}

# 主监控循环
monitor_loop() {
    while true; do
        clear
        echo "RAG-Searx 内存使用监控 - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "======================================"
        
        # 获取运行中的容器
        containers=$(get_rag_containers)
        
        if [ -z "$containers" ]; then
            echo "未找到运行中的RAG-Searx相关容器"
            echo "请先启动服务: ./scripts/start_optimized.sh"
            sleep 5
            continue
        fi
        
        # 显示表头
        printf "%-20s %-15s %-15s %-10s\n" "容器名称" "内存使用" "内存限制" "CPU使用"
        echo "-------------------------------------------------------------"
        
        total_memory_mb=0
        
        # 显示每个容器的信息
        for container in $containers; do
            stats=$(docker stats --no-stream --format "{{.MemUsage}}\t{{.CPUPerc}}" $container 2>/dev/null)
            if [ $? -eq 0 ]; then
                mem_usage=$(echo "$stats" | cut -f1 | cut -d'/' -f1)
                mem_limit=$(echo "$stats" | cut -f1 | cut -d'/' -f2)
                cpu_usage=$(echo "$stats" | cut -f2)
                
                # 计算总内存（简化版本）
                if [[ $mem_usage =~ ([0-9.]+)MiB ]]; then
                    mb="${BASH_REMATCH[1]}"
                    total_memory_mb=$(echo "$total_memory_mb + $mb" | bc 2>/dev/null || echo "$total_memory_mb")
                elif [[ $mem_usage =~ ([0-9.]+)GiB ]]; then
                    gb="${BASH_REMATCH[1]}"
                    mb_from_gb=$(echo "$gb * 1024" | bc 2>/dev/null || echo "0")
                    total_memory_mb=$(echo "$total_memory_mb + $mb_from_gb" | bc 2>/dev/null || echo "$total_memory_mb")
                fi
                
                printf "%-20s %-15s %-15s %-10s\n" "$container" "$mem_usage" "$mem_limit" "$cpu_usage"
            else
                printf "%-20s %-15s %-15s %-10s\n" "$container" "N/A" "N/A" "N/A"
            fi
        done
        
        echo "-------------------------------------------------------------"
        
        # 显示总计
        if (( $(echo "$total_memory_mb > 1024" | bc -l 2>/dev/null || echo "0") )); then
            total_gb=$(echo "scale=2; $total_memory_mb / 1024" | bc -l 2>/dev/null || echo "0")
            echo "总内存使用: ${total_gb}GB"
        else
            echo "总内存使用: ${total_memory_mb}MB"
        fi
        
        container_count=$(echo "$containers" | wc -l)
        echo "运行容器数: $container_count"
        
        echo ""
        echo "内存使用建议:"
        if (( $(echo "$total_memory_mb > 3072" | bc -l 2>/dev/null || echo "0") )); then
            echo "高内存使用 - 建议切换到内存优化模式"
        elif (( $(echo "$total_memory_mb > 1536" | bc -l 2>/dev/null || echo "0") )); then
            echo "中等内存使用 - 运行状况良好"
        else
            echo "低内存使用 - 运行效率很高"
        fi
        
        echo ""
        echo "刷新间隔: 5秒 | 按 Ctrl+C 退出"
        
        sleep 5
    done
}

# 处理中断信号
trap 'echo -e "\n\n监控已停止"; exit 0' INT

# 开始监控
monitor_loop
