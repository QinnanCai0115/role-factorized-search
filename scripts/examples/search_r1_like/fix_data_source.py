#!/usr/bin/env python3
"""
修复parquet数据 - 添加缺失的data_source列
"""
import sys
from pathlib import Path

def add_data_source_column():
    """为parquet文件添加data_source列"""
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
    except ImportError:
        print("错误: 需要安装pyarrow")
        print("运行: pip install pyarrow")
        return False
    
    parquet_file = Path('/ai/cqn/s3/data/hotpotqa_2wiki_musique_train/train_e5_s3.parquet')
    
    if not parquet_file.exists():
        print(f"错误: 文件不存在: {parquet_file}")
        return False
    
    print(f"读取文件: {parquet_file}")
    table = pq.read_table(parquet_file)
    
    print(f"原始列: {table.column_names}")
    
    # 检查是否已有data_source列
    if 'data_source' in table.column_names:
        print("✓ data_source列已存在")
        return True
    
    # 添加data_source列 - 默认值为'hotpotqa'
    data_source_col = pa.array(['hotpotqa'] * table.num_rows)
    table = table.append_column('data_source', data_source_col)
    
    print(f"新增列: data_source")
    print(f"更新后的列: {table.column_names}")
    
    # 备份原文件
    backup_file = parquet_file.with_suffix('.parquet.bak')
    print(f"备份原文件到: {backup_file}")
    import shutil
    shutil.copy(parquet_file, backup_file)
    
    # 保存更新后的文件
    print(f"保存更新后的文件...")
    pq.write_table(table, parquet_file)
    
    print(f"✓ 成功添加data_source列")
    print(f"  总行数: {table.num_rows}")
    print(f"  总列数: {table.num_columns}")
    
    return True

if __name__ == '__main__':
    success = add_data_source_column()
    sys.exit(0 if success else 1)
