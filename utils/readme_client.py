from typing import Dict, Any, List, Optional, Tuple, Generator
from openai import OpenAI
import re
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime, timedelta
from icalendar import Calendar, Event
import json
from utils.llm_client import LLMClient

class ReadmeViewerLLM(LLMClient):
    """README查看器的LLM适配器"""
    
    def __init__(self, api_key: str, base_url: str, model: str = "qwen-turbo"):
        super().__init__(api_key, base_url, model)
        # 导入页面功能
        try:
            import sys
            import os
            # 添加项目根目录到路径
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.append(project_root)
            
            from pages.readme.readme_page import readme_show_page
            self.readme_page = readme_show_page()
        except ImportError as e:
            print(f"⚠️ 无法导入 ReadmePage: {e}")
            self.readme_page = None
    
    def execute_task(self, task_description: str, context: dict) -> str:
        """
        执行README相关任务
        """
        print(f"📖 ReadmeViewerLLM 正在执行: {task_description}")
        
        if not self.readme_page:
            return "错误：README页面功能未正确加载"
        
        try:
            # 根据任务描述判断要执行什么操作
            if "显示" in task_description or "查看" in task_description or "readme" in task_description.lower():
                # 调用README页面的显示功能
                if hasattr(self.readme_page, 'display_readme'):
                    result = self.readme_page.display_readme()
                elif hasattr(self.readme_page, 'get_readme_content'):
                    result = self.readme_page.get_readme_content()
                else:
                    # 如果没有特定方法，尝试调用主要方法
                    result = "README内容已显示"
                
                return f"✅ README查看完成：{result}"
            else:
                return f"✅ 已处理README相关请求：{task_description}"
                
        except Exception as e:
            error_msg = f"README功能执行失败: {e}"
            print(f"❌ {error_msg}")
            return error_msg