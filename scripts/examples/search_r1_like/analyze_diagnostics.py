#!/usr/bin/env python3
"""
NCCL诊断日志分析工具 - 识别真正的阻塞点
"""
import re
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def parse_nccl_log(log_file):
    """解析NCCL日志文件"""
    print(f"\n{'='*80}")
    print(f"分析NCCL日志: {log_file}")
    print(f"{'='*80}\n")
    
    if not Path(log_file).exists():
        print(f"错误: 日志文件不存在: {log_file}")
        return
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # 统计信息
    stats = {
        'total_lines': len(lines),
        'errors': [],
        'timeouts': [],
        'broadcasts': [],
        'barriers': [],
        'all_reduces': [],
        'rank_info': defaultdict(list),
        'timestamps': []
    }
    
    # 解析日志
    for i, line in enumerate(lines):
        # 检测超时
        if 'timeout' in line.lower() or 'timed out' in line.lower():
            stats['timeouts'].append({
                'line_num': i + 1,
                'content': line.strip(),
                'context': lines[max(0, i-2):min(len(lines), i+3)]
            })
        
        # 检测错误
        if any(x in line.upper() for x in ['ERROR', 'FAILED', 'EXCEPTION']):
            stats['errors'].append({
                'line_num': i + 1,
                'content': line.strip()
            })
        
        # 检测BROADCAST操作
        if 'BROADCAST' in line:
            stats['broadcasts'].append({
                'line_num': i + 1,
                'content': line.strip()
            })
        
        # 检测BARRIER操作
        if 'BARRIER' in line:
            stats['barriers'].append({
                'line_num': i + 1,
                'content': line.strip()
            })
        
        # 检测ALL_REDUCE操作
        if 'ALL_REDUCE' in line:
            stats['all_reduces'].append({
                'line_num': i + 1,
                'content': line.strip()
            })
        
        # 提取Rank信息
        rank_match = re.search(r'\[rank(\d+)\]', line)
        if rank_match:
            rank = rank_match.group(1)
            stats['rank_info'][rank].append({
                'line_num': i + 1,
                'content': line.strip()
            })
    
    # 打印分析结果
    print(f"总行数: {stats['total_lines']}")
    print(f"错误数: {len(stats['errors'])}")
    print(f"超时数: {len(stats['timeouts'])}")
    print(f"BROADCAST操作: {len(stats['broadcasts'])}")
    print(f"BARRIER操作: {len(stats['barriers'])}")
    print(f"ALL_REDUCE操作: {len(stats['all_reduces'])}")
    print(f"涉及的Rank: {sorted(stats['rank_info'].keys())}")
    
    # 详细输出超时信息
    if stats['timeouts']:
        print(f"\n{'='*80}")
        print("超时事件详情:")
        print(f"{'='*80}")
        for i, timeout in enumerate(stats['timeouts'][:5], 1):  # 显示前5个
            print(f"\n[超时 #{i}] 行 {timeout['line_num']}")
            print(f"内容: {timeout['content']}")
            print("上下文:")
            for ctx_line in timeout['context']:
                print(f"  {ctx_line.rstrip()}")
    
    # 详细输出错误信息
    if stats['errors']:
        print(f"\n{'='*80}")
        print("错误事件详情:")
        print(f"{'='*80}")
        for i, error in enumerate(stats['errors'][:10], 1):  # 显示前10个
            print(f"\n[错误 #{i}] 行 {error['line_num']}")
            print(f"内容: {error['content']}")
    
    # 分析Rank不平衡
    if stats['rank_info']:
        print(f"\n{'='*80}")
        print("Rank活动分析:")
        print(f"{'='*80}")
        for rank in sorted(stats['rank_info'].keys()):
            count = len(stats['rank_info'][rank])
            print(f"Rank {rank}: {count} 条日志")
            
            # 检查该Rank是否有超时
            rank_timeouts = [t for t in stats['timeouts'] if f'[rank{rank}]' in t['content']]
            if rank_timeouts:
                print(f"  ⚠️  该Rank有 {len(rank_timeouts)} 个超时事件")
    
    # 生成建议
    print(f"\n{'='*80}")
    print("诊断建议:")
    print(f"{'='*80}")
    
    if stats['timeouts']:
        print("❌ 检测到NCCL超时")
        print("   建议:")
        print("   1. 增加NCCL_TIMEOUT环境变量 (当前: 1800秒)")
        print("   2. 检查GPU间通信 (nvidia-smi topo -m)")
        print("   3. 检查网络连接和NCCL版本")
        print("   4. 减少batch size或模型大小")
    
    if len(stats['rank_info']) > 1:
        rank_counts = {r: len(stats['rank_info'][r]) for r in stats['rank_info']}
        max_count = max(rank_counts.values())
        min_count = min(rank_counts.values())
        if max_count > min_count * 1.5:
            print("⚠️  检测到Rank活动不平衡")
            print(f"   最活跃Rank: {max(rank_counts, key=rank_counts.get)} ({max_count}条)")
            print(f"   最不活跃Rank: {min(rank_counts, key=rank_counts.get)} ({min_count}条)")
            print("   建议: 检查是否有Rank卡住或通信不畅")
    
    print()

def parse_training_log(log_file):
    """解析训练日志"""
    print(f"\n{'='*80}")
    print(f"分析训练日志: {log_file}")
    print(f"{'='*80}\n")
    
    if not Path(log_file).exists():
        print(f"错误: 日志文件不存在: {log_file}")
        return
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # 查找关键事件
    print("关键事件时间线:")
    print("-" * 80)
    
    for i, line in enumerate(lines):
        # 查找初始化事件
        if any(x in line for x in ['init_workers', 'FSDP', 'FullySharded', 'wrap_policy']):
            print(f"[{i}] {line.rstrip()}")
        
        # 查找错误
        if any(x in line.upper() for x in ['ERROR', 'FAILED', 'TIMEOUT', 'EXCEPTION']):
            print(f"[{i}] ❌ {line.rstrip()}")
    
    print()

def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_diagnostics.py <诊断目录或日志文件>")
        print("\n示例:")
        print("  python analyze_diagnostics.py /ai/cqn/s3/tmp_logs/training_diagnostics")
        print("  python analyze_diagnostics.py /ai/cqn/s3/tmp_logs/training_diagnostics/nccl_20260330_145900.log")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if path.is_file():
        # 分析单个文件
        if 'nccl' in path.name.lower():
            parse_nccl_log(str(path))
        else:
            parse_training_log(str(path))
    elif path.is_dir():
        # 分析目录中的所有日志
        nccl_logs = sorted(path.glob('nccl_*.log'))
        training_logs = sorted(path.glob('training_*.log'))
        
        for log in nccl_logs[-1:]:  # 只分析最新的
            parse_nccl_log(str(log))
        
        for log in training_logs[-1:]:  # 只分析最新的
            parse_training_log(str(log))
    else:
        print(f"错误: 路径不存在: {path}")
        sys.exit(1)

if __name__ == '__main__':
    main()
