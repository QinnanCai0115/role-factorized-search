#!/bin/bash
# 训练日志查看工具

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

# 日志目录
LOG_DIR="$PROJECT_DIR/tmp_logs"
TENSORBOARD_DIR="$PROJECT_DIR/tensorboard_log"
TB_HOST="${TB_HOST:-0.0.0.0}"
TB_PORT="${TB_PORT:-6006}"
TB_LOCAL_PORT="${TB_LOCAL_PORT:-$TB_PORT}"

show_help() {
    cat << 'EOF'
训练日志查看工具

用法:
  ./view_logs.sh [选项]

选项:
  latest          查看最新的训练日志
  tail            实时查看日志末尾 (tail -f)
  errors          查看所有错误信息
  warnings        查看所有警告信息
  nccl            查看NCCL相关日志
  ray             查看Ray相关日志
  tensorboard     启动TensorBoard (默认支持远程访问)
  list            列出所有日志文件
  search <关键词> 搜索日志中的关键词
  help            显示此帮助信息

示例:
  ./view_logs.sh latest
  ./view_logs.sh tail
  ./view_logs.sh errors
  ./view_logs.sh search "NCCL timeout"
  ./view_logs.sh tensorboard

环境变量:
  TB_HOST         TensorBoard监听地址，默认 0.0.0.0
  TB_PORT         TensorBoard端口，默认 6006
  TB_LOCAL_PORT   本机映射端口，默认与 TB_PORT 相同
  TENSORBOARD_DIR TensorBoard日志目录，默认 $PROJECT_DIR/tensorboard_log
EOF
}

# 查找最新的日志文件
find_latest_log() {
    find "$LOG_DIR" -maxdepth 1 -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | \
        sort -rn | head -1 | cut -d' ' -f2-
}

# 查看最新日志
view_latest() {
    local latest=$(find_latest_log)
    if [ -z "$latest" ]; then
        echo "❌ 未找到日志文件"
        return 1
    fi
    echo "📄 查看最新日志: $latest"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    less "$latest"
}

# 实时查看日志
tail_latest() {
    local latest=$(find_latest_log)
    if [ -z "$latest" ]; then
        echo "❌ 未找到日志文件"
        return 1
    fi
    echo "📄 实时查看日志: $latest"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    tail -f "$latest"
}

# 查看错误
view_errors() {
    local latest=$(find_latest_log)
    if [ -z "$latest" ]; then
        echo "❌ 未找到日志文件"
        return 1
    fi
    echo "❌ 错误信息 (来自: $latest)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    grep -i "error\|failed\|exception\|traceback" "$latest" || echo "未找到错误信息"
}

# 查看警告
view_warnings() {
    local latest=$(find_latest_log)
    if [ -z "$latest" ]; then
        echo "❌ 未找到日志文件"
        return 1
    fi
    echo "⚠️  警告信息 (来自: $latest)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    grep -i "warning\|warn" "$latest" || echo "未找到警告信息"
}

# 查看NCCL日志
view_nccl() {
    echo "🔍 NCCL相关日志"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    local latest=$(find_latest_log)
    if [ -n "$latest" ]; then
        echo "📄 主日志中的NCCL信息:"
        grep -i "nccl" "$latest" | head -20 || echo "未找到NCCL信息"
    fi
    
    # 查找NCCL诊断日志
    local nccl_logs=$(find "$LOG_DIR" -name "*nccl*" -type f 2>/dev/null)
    if [ -n "$nccl_logs" ]; then
        echo ""
        echo "📄 NCCL诊断日志:"
        echo "$nccl_logs" | head -5
    fi
}

# 查看Ray日志
view_ray() {
    echo "🔍 Ray相关日志"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    local latest=$(find_latest_log)
    if [ -n "$latest" ]; then
        echo "📄 主日志中的Ray信息:"
        grep -i "ray\|actor\|worker" "$latest" | head -20 || echo "未找到Ray信息"
    fi
    
    # 查找Ray日志目录
    local ray_logs=$(find "$LOG_DIR/ray" -name "*.log" -type f 2>/dev/null | head -5)
    if [ -n "$ray_logs" ]; then
        echo ""
        echo "📄 Ray日志文件:"
        echo "$ray_logs"
    fi
}

# 启动TensorBoard
start_tensorboard() {
    if [ ! -d "$TENSORBOARD_DIR" ]; then
        echo "❌ TensorBoard目录不存在: $TENSORBOARD_DIR"
        return 1
    fi

    local primary_ip
    primary_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    local host_name
    host_name="$(hostname -f 2>/dev/null || hostname)"
    local ssh_user
    ssh_user="${USER:-$(whoami 2>/dev/null)}"
    
    echo "🚀 启动TensorBoard..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 本机访问: http://localhost:$TB_PORT"
    if [ -n "$primary_ip" ]; then
        echo "🌐 远程访问: http://$primary_ip:$TB_PORT"
    fi
    echo "📁 日志目录: $TENSORBOARD_DIR"
    echo "🧭 监听地址: $TB_HOST:$TB_PORT"
    echo ""
    echo "如果你在本机终端做 SSH 端口转发，可直接执行:"
    echo "ssh -N -L ${TB_LOCAL_PORT}:127.0.0.1:${TB_PORT} ${ssh_user}@${host_name}"
    echo ""
    echo "转发成功后，在本机浏览器打开:"
    echo "http://localhost:${TB_LOCAL_PORT}"
    echo ""
    echo "按 Ctrl+C 停止TensorBoard"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    tensorboard --logdir="$TENSORBOARD_DIR" --host="$TB_HOST" --port="$TB_PORT"
}

# 列出所有日志
list_logs() {
    echo "📋 所有日志文件"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    echo ""
    echo "主日志文件:"
    find "$LOG_DIR" -maxdepth 1 -name "*.log" -type f -exec ls -lh {} \; | \
        awk '{print $9, "(" $5 ")"}' | sort -r
    
    echo ""
    echo "诊断日志:"
    find "$LOG_DIR" -name "*diagnostic*" -o -name "*nccl*" -o -name "*fsdp*" 2>/dev/null | head -10
    
    echo ""
    echo "Ray日志:"
    find "$LOG_DIR/ray" -name "*.log" -type f 2>/dev/null | head -5
}

# 搜索日志
search_logs() {
    local keyword="$1"
    if [ -z "$keyword" ]; then
        echo "❌ 请提供搜索关键词"
        return 1
    fi
    
    local latest=$(find_latest_log)
    if [ -z "$latest" ]; then
        echo "❌ 未找到日志文件"
        return 1
    fi
    
    echo "🔍 搜索关键词: '$keyword'"
    echo "📄 日志文件: $latest"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    grep -i "$keyword" "$latest" || echo "未找到匹配的内容"
}

# 主程序
main() {
    if [ $# -eq 0 ]; then
        show_help
        return 0
    fi
    
    case "$1" in
        latest)
            view_latest
            ;;
        tail)
            tail_latest
            ;;
        errors)
            view_errors
            ;;
        warnings)
            view_warnings
            ;;
        nccl)
            view_nccl
            ;;
        ray)
            view_ray
            ;;
        tensorboard)
            start_tensorboard
            ;;
        list)
            list_logs
            ;;
        search)
            search_logs "$2"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "❌ 未知选项: $1"
            echo ""
            show_help
            return 1
            ;;
    esac
}

main "$@"
