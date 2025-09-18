#!/bin/bash

# 切换到上一级目录
cd ../

# 获取当前日期，格式为YYMMDD
current_date=$(date +'%y%m%d')
backup_dir="agent_streamlit_study_bak_${current_date}"

# 删除已存在的备份目录（如果存在的话）
rm -rf agent_streamlit_study_bak_*

# 创建新的备份
cp -r agent_streamlit_study "$backup_dir"

echo "Backup completed: $backup_dir"
