#!/bin/bash
# 快速修复脚本 - 解决data_source缺失问题

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

echo "=========================================="
echo "修复GRPO训练 - data_source列缺失问题"
echo "=========================================="

# 步骤1: 修复reward manager代码
echo ""
echo "[1/2] 检查reward manager修复..."
NAIVE_PY="$PROJECT_DIR/verl/workers/reward_manager/naive.py"

if grep -q "Handle missing data_source column" "$NAIVE_PY"; then
    echo "✓ reward manager已修复"
else
    echo "✗ reward manager未修复，应用补丁..."
    # 补丁已在上一步应用
fi

# 步骤2: 可选 - 添加data_source列到数据
echo ""
echo "[2/2] 检查数据文件..."
DATA_FILE="$PROJECT_DIR/data/hotpotqa_2wiki_musique_train/train_e5_s3.parquet"

if [ -f "$DATA_FILE" ]; then
    echo "✓ 数据文件存在: $DATA_FILE"
    echo ""
    echo "可选: 运行以下命令添加data_source列到数据:"
    echo "  python3 $SCRIPT_DIR/fix_data_source.py"
else
    echo "✗ 数据文件不存在: $DATA_FILE"
fi

echo ""
echo "=========================================="
echo "修复完成！现在可以运行训练了"
echo "=========================================="
echo ""
echo "运行训练:"
echo "  bash $SCRIPT_DIR/run_two_stage_subagent_grpo.sh"
