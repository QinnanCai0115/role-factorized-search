#!/bin/bash
# 带诊断的GRPO训练启动脚本

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

# 诊断配置
ENABLE_NCCL_DIAGNOSTICS="${ENABLE_NCCL_DIAGNOSTICS:-1}"
NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
DIAG_DIR="${PROJECT_DIR}/tmp_logs/training_diagnostics"

mkdir -p "$DIAG_DIR"

# 设置诊断环境变量
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_NCCL_TRACE_BUFFER_SIZE=134217728
export NCCL_TIMEOUT=$NCCL_TIMEOUT
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CPP_LOG_LEVEL=INFO

# 创建诊断日志文件
DIAG_LOG="${DIAG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
NCCL_LOG="${DIAG_DIR}/nccl_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "GRPO训练 - 带NCCL诊断"
echo "=========================================="
echo "诊断目录: $DIAG_DIR"
echo "诊断日志: $DIAG_LOG"
echo "NCCL日志: $NCCL_LOG"
echo "NCCL超时: $NCCL_TIMEOUT 秒"
echo "=========================================="

# 导出NCCL日志路径
export NCCL_DEBUG_FILE="$NCCL_LOG"

# 获取原始脚本的所有参数
ORIGINAL_SCRIPT="${SCRIPT_DIR}/run_two_stage_subagent_grpo.sh"

# 运行原始脚本并捕获输出
if [ "$ENABLE_NCCL_DIAGNOSTICS" = "1" ]; then
    echo "启用NCCL诊断模式..."
    bash "$ORIGINAL_SCRIPT" "$@" 2>&1 | tee "$DIAG_LOG"
    EXIT_CODE=$?
else
    bash "$ORIGINAL_SCRIPT" "$@"
    EXIT_CODE=$?
fi

# 诊断总结
echo ""
echo "=========================================="
echo "训练完成 (退出码: $EXIT_CODE)"
echo "=========================================="

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "诊断文件位置:"
    echo "  - 训练日志: $DIAG_LOG"
    echo "  - NCCL日志: $NCCL_LOG"
    echo ""
    echo "分析NCCL超时:"
    if [ -f "$NCCL_LOG" ]; then
        echo "  最后100行NCCL日志:"
        tail -100 "$NCCL_LOG"
    fi
fi

exit $EXIT_CODE
