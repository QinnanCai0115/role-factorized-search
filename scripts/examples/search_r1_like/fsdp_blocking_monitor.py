#!/usr/bin/env python3
"""
FSDP初始化阻塞点诊断 - 实时捕获通信死锁
"""
import os
import sys
import time
import threading
import traceback
import logging
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

# 创建诊断目录
DIAG_DIR = Path('/ai/cqn/s3/tmp_logs/fsdp_diagnostics')
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# 配置日志
log_file = DIAG_DIR / f'fsdp_init_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BlockingPointMonitor:
    """监控FSDP初始化中的阻塞点"""
    
    def __init__(self, timeout_seconds=120):
        self.timeout_seconds = timeout_seconds
        self.current_operation = None
        self.operation_start_time = None
        self.monitor_thread = None
        self.should_stop = False
        self.lock = threading.Lock()
    
    def start_operation(self, op_name):
        """标记操作开始"""
        with self.lock:
            self.current_operation = op_name
            self.operation_start_time = time.time()
            logger.info(f"[OP_START] {op_name}")
    
    def end_operation(self):
        """标记操作结束"""
        with self.lock:
            if self.current_operation:
                elapsed = time.time() - self.operation_start_time
                logger.info(f"[OP_END] {self.current_operation} (耗时: {elapsed:.2f}s)")
                self.current_operation = None
    
    def start_monitoring(self):
        """启动后台监控线程"""
        self.should_stop = False
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("阻塞点监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.should_stop = True
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """监控循环 - 检测长时间操作"""
        while not self.should_stop:
            time.sleep(5)
            with self.lock:
                if self.current_operation:
                    elapsed = time.time() - self.operation_start_time
                    if elapsed > self.timeout_seconds:
                        self._save_blocking_info(elapsed)
    
    def _save_blocking_info(self, elapsed_time):
        """保存阻塞信息和堆栈跟踪"""
        import faulthandler
        
        block_file = DIAG_DIR / f'blocking_point_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.txt'
        logger.error(f"检测到长时间操作: {self.current_operation} (耗时: {elapsed_time:.2f}s)")
        
        with open(block_file, 'w') as f:
            f.write(f"=== 阻塞点诊断 ===\n")
            f.write(f"时间: {datetime.now()}\n")
            f.write(f"操作: {self.current_operation}\n")
            f.write(f"耗时: {elapsed_time:.2f}s (超时阈值: {self.timeout_seconds}s)\n\n")
            
            f.write("=== 所有线程堆栈跟踪 ===\n")
            for thread_id, frame in sys._current_frames().items():
                f.write(f"\n--- 线程 {thread_id} ---\n")
                traceback.print_tb(frame, file=f)
            
            f.write("\n=== Faulthandler 详细信息 ===\n")
            faulthandler.dump_traceback(f)
        
        logger.error(f"阻塞信息已保存到: {block_file}")

# 全局监控器
monitor = BlockingPointMonitor(timeout_seconds=120)

@contextmanager
def monitored_operation(op_name):
    """上下文管理器 - 监控操作"""
    monitor.start_operation(op_name)
    try:
        yield
    finally:
        monitor.end_operation()

def patch_fsdp_init():
    """补丁FSDP初始化以添加监控"""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    
    original_init = FSDP.__init__
    
    def monitored_init(self, *args, **kwargs):
        with monitored_operation(f"FSDP.__init__ (module={args[0].__class__.__name__ if args else 'unknown'})"):
            return original_init(self, *args, **kwargs)
    
    FSDP.__init__ = monitored_init
    logger.info("FSDP.__init__ 补丁已应用")

def patch_dist_operations():
    """补丁分布式操作"""
    import torch.distributed as dist
    
    # 补丁 barrier
    original_barrier = dist.barrier
    def monitored_barrier(*args, **kwargs):
        with monitored_operation("dist.barrier"):
            return original_barrier(*args, **kwargs)
    dist.barrier = monitored_barrier
    
    # 补丁 broadcast
    original_broadcast = dist.broadcast
    def monitored_broadcast(*args, **kwargs):
        with monitored_operation("dist.broadcast"):
            return original_broadcast(*args, **kwargs)
    dist.broadcast = monitored_broadcast
    
    # 补丁 all_reduce
    original_all_reduce = dist.all_reduce
    def monitored_all_reduce(*args, **kwargs):
        with monitored_operation("dist.all_reduce"):
            return original_all_reduce(*args, **kwargs)
    dist.all_reduce = monitored_all_reduce
    
    logger.info("分布式操作补丁已应用")

def main():
    logger.info("=" * 80)
    logger.info("FSDP初始化阻塞点诊断工具")
    logger.info(f"诊断目录: {DIAG_DIR}")
    logger.info("=" * 80)
    
    # 设置NCCL环境变量
    os.environ['NCCL_DEBUG'] = 'TRACE'
    os.environ['NCCL_TIMEOUT'] = '1800'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    
    # 启动监控
    monitor.start_monitoring()
    
    # 应用补丁
    patch_fsdp_init()
    patch_dist_operations()
    
    try:
        logger.info("启动主训练脚本...")
        from verl.trainer.main_ppo import main as ppo_main
        ppo_main()
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        raise
    finally:
        monitor.stop_monitoring()
        logger.info("诊断完成")

if __name__ == '__main__':
    main()
