#!/usr/bin/env python3
import os
import shutil
import glob
from datetime import datetime

try:
    # 切换到上一级目录
    os.chdir('../')
    
    # 获取当前日期和时间，格式为YYMMDD_HHMM
    current_datetime = datetime.now().strftime('%y%m%d_%H%M')
    backup_dir = f"agent_streamlit_study_bak_{current_datetime}"
    
    # 删除已存在的备份目录（如果存在的话）
    existing_backups = glob.glob("agent_streamlit_study_bak_*")
    for backup in existing_backups:
        if os.path.exists(backup):
            print(f"Removing existing backup: {backup}")
            # 处理只读文件（特别是.git目录中的文件）
            def handle_remove_readonly(func, path, exc):
                import stat
                os.chmod(path, stat.S_IWRITE)
                func(path)
            
            shutil.rmtree(backup, onerror=handle_remove_readonly)
    
    # 创建新的备份，但排除.git目录
    def ignore_git(src, names):
        return {'.git'} if '.git' in names else set()
    
    shutil.copytree("agent_streamlit_study", backup_dir, ignore=ignore_git)
    
    print(f"Backup completed: {backup_dir}")
    
except Exception as e:
    print(f"Error occurred: {e}")
