import os
import shutil
from datetime import datetime
import glob

def backup_project():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 切换到上一级目录
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
    
    # 获取当前日期，格式为YYMMDD
    current_date = datetime.now().strftime('%y%m%d')
    backup_dir = f"agent_streamlit_study_bak_{current_date}"
    
    # 删除已存在的备份目录
    for dir_to_remove in glob.glob("agent_streamlit_study_bak_*"):
        if os.path.exists(dir_to_remove):
            print(f"Removing existing backup: {dir_to_remove}")
            shutil.rmtree(dir_to_remove)
    
    # 创建新的备份
    source_dir = "agent_streamlit_study"
    print(f"Creating new backup: {backup_dir}")
    shutil.copytree(source_dir, backup_dir)
    
    print(f"Backup completed: {backup_dir}")

if __name__ == "__main__":
    try:
        backup_project()
    except Exception as e:
        print(f"Error occurred: {str(e)}")