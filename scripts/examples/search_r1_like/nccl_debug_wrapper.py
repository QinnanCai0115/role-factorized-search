#!/usr/bin/env python3
"""
NCCL超时诊断包装器 - 捕获真正的阻塞点
"""
import os
import sys
import signal
import traceback
import logging
from datetime import datetime
from pathlib import Path

# 设置详细的NCCL调试
os.environ['NCCL_DEBUG'] = 'TRACE'
os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
os.environ['TORCH_NCCL_TRACE_BUFFER_SIZE'] = '134217728'
os.environ['NCCL_TIMEOUT'] = '1800'  # 30分钟
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
os.environ['TORCH_CPP_LOG_LEVEL'] = 'INFO'

# 创建诊断日志目录
DIAG_DIR = Path('/ai/cqn/s3/tmp_logs/nccl_diagnostics')
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# 设置日志
log_file = DIAG_DIR / f'nccl_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def save_stack_traces(signum, frame):
    """信号处理器 - 保存所有线程的堆栈跟踪"""
    import faulthandler
    import threading
    
    trace_file = DIAG_DIR / f'stack_trace_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.txt'
    logger.error(f"捕获到信号 {signum}，保存堆栈跟踪到 {trace_file}")
    
    with open(trace_file, 'w') as f:
        f.write(f"=== 堆栈跟踪 - {datetime.now()} ===\n")
        f.write(f"信号: {signum}\n")
        f.write(f"活跃线程数: {threading.active_count()}\n\n")
        
        # 保存所有线程的堆栈
        for thread_id, frame in sys._current_frames().items():
            f.write(f"\n--- 线程 {thread_id} ---\n")
            traceback.print_tb(frame, file=f)
        
        # 使用faulthandler获取更详细的信息
        f.write("\n\n=== Faulthandler 输出 ===\n")
        faulthandler.dump_traceback(f)
    
    logger.error(f"堆栈跟踪已保存到: {trace_file}")

def setup_signal_handlers():
    """设置信号处理器以捕获超时"""
    signal.signal(signal.SIGALRM, save_stack_traces)
    signal.signal(signal.SIGTERM, save_stack_traces)
    signal.signal(signal.SIGABRT, save_stack_traces)
    logger.info("信号处理器已设置")

def patch_nccl_monitoring():
    """补丁PyTorch分布式以添加更多监控"""
    import torch.distributed as dist
    
    original_barrier = dist.barrier
    
    def monitored_barrier(group=None, async_op=False):
        logger.debug(f"进入 barrier (group={group}, async_op={async_op})")
        try:
            result = original_barrier(group=group, async_op=async_op)
            logger.debug(f"barrier 完成")
            return result
        except Exception as e:
            logger.error(f"barrier 失败: {e}", exc_info=True)
            raise
    
    dist.barrier = monitored_barrier
    logger.info("NCCL监控补丁已应用")

def main():
    logger.info("=" * 80)
    logger.info("NCCL超时诊断包装器启动")
    logger.info(f"诊断目录: {DIAG_DIR}")
    logger.info("=" * 80)
    
    # 设置信号处理
    setup_signal_handlers()
    
    # 应用监控补丁
    patch_nccl_monitoring()
    
    # 导入并运行主训练脚本
    logger.info("启动主训练脚本...")
    
    # 移除此脚本的参数，保留其他参数
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    try:
        from verl.trainer.main_ppo import main as ppo_main
        ppo_main()
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        save_stack_traces(signal.SIGABRT, None)
        raise

if __name__ == '__main__':
    main()
