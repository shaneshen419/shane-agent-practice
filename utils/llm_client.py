from typing import Dict, Any, List, Optional, Tuple, Generator
from openai import OpenAI
import re
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime, timedelta
from icalendar import Calendar, Event
import json


class LLMClient:
    """通用LLM客户端基类"""
    
    def __init__(self, api_key: str, base_url: str = None, model: str = "qwen-turbo"):
        """初始化LLM客户端"""
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        
        # 初始化OpenAI客户端
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": None
        }
        
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        self.client = OpenAI(**client_kwargs)



