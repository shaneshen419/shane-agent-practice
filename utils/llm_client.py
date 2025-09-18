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


class TravelPlannerLLM(LLMClient):
    """旅行规划专用LLM客户端"""
    
    def generate_itinerary_stream(self, destination: str, num_days: int) -> Generator[str, None, None]:
        """流式生成旅行行程"""
        system_prompt = """你是一个专业的旅行规划师，具有丰富的全球旅行知识。
请为用户创建详细的、实用的旅行行程。

要求：
1. 为每一天创建详细的活动安排
2. 包括景点推荐、餐厅建议、住宿选择
3. 提供交通方式、最佳游览时间、预算估计
4. 确保行程节奏合理，不过于紧张
5. 使用清晰的"Day 1:", "Day 2:"等格式标题
6. 考虑当地文化和实际情况"""

        user_prompt = f"""请为{destination}创建一个{num_days}天的详细旅行行程。

请包括：
- 每日具体的景点和活动安排
- 餐厅和美食推荐
- 住宿建议
- 交通提示
- 预算估计
- 当地文化洞察

请确保每天都有明确的"Day X:"标题，方便转换为日历事件。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                top_p=0.9,
                max_tokens=4096,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise Exception(f"生成行程时发生错误: {str(e)}")
    
    def generate_itinerary(self, destination: str, num_days: int) -> str:
        """非流式生成旅行行程"""
        full_text = ""
        for chunk in self.generate_itinerary_stream(destination, num_days):
            full_text += chunk
        return full_text

    # +++ 新增：标准化的任务执行入口，用于被MCP调用 +++
    def execute_task(self, task_description: str, context: dict) -> str:
        """
        作为工具被MCP调用时执行的具体任务。
        task_description: MCP分配的具体指令, e.g., "为去巴黎的5日游制定一个行程"
        context: 任务上下文，可能包含前置任务的结果
        """
        print(f"✈️ TravelPlannerLLM 正在执行: {task_description}")
        # 使用LLM来解析任务描述中的目的地和天数信息
        parsed_info = self._parse_task_with_llm(task_description)
        
        destination = parsed_info.get("destination")
        days = parsed_info.get("days")
        
        # 如果LLM解析失败，则尝试使用正则表达式解析
        if not destination or not days:
            import re
            print("🔄 LLM解析失败，尝试使用正则表达式解析...")
            
            # 尝试提取目的地 - 使用更精确的模式
            destination_match = re.search(r'安排一下([\u4e00-\u9fa5]{2,5})(?=国庆)', task_description)
            destination = destination_match.group(1) if destination_match else None
            
            # 尝试提取天数 - 匹配中文数字或阿拉伯数字
            days_match = re.search(r'(?:(\d+)|([一二三四五六七八九十]))天', task_description)
            days = days_match.group(1) if days_match else None
            # 如果匹配到中文数字，需要转换为阿拉伯数字
            days_dict = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'}
            if not days and days_match and days_match.group(2):
                days = days_dict.get(days_match.group(2))
            
            if not destination or not days:
                return f"错误：无法从任务 '{task_description}' 中解析出目的地和天数。"
        
        num_days = int(days)

        # 使用流式方法来完成任务
        try:
            result = ""
            for chunk in self.generate_itinerary_stream(destination, num_days):
                result += chunk
            print(f"✈️ TravelPlannerLLM 完成任务。")
            return result
        except Exception as e:
            return f"旅行规划执行失败: {e}"

    def _parse_task_with_llm(self, task_description: str) -> dict:
        """使用LLM来智能解析任务描述中的目的地和天数信息"""
        try:
            parse_prompt = f"""
请从以下任务描述中提取旅行规划信息：

任务描述: \"{task_description}\"

请返回JSON格式的结果：
{{
    \"destination\": \"目的地城市名称\",
    \"days\": 天数（数字）
}}

如果无法确定某个信息，请返回null。
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": parse_prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            print(f"🧠 LLM解析结果: {result}")
            return result
            
        except Exception as e:
            print(f"⚠️ LLM解析失败: {e}")
            return {}

class VisionLLMClient(LLMClient):
    """视觉识别专用LLM客户端"""
    
    def __init__(self, api_key: str, base_url: str = None, model: str = "qwen-vl-plus"):
        super().__init__(api_key, base_url, model)
    
    def encode_image_to_base64(self, image: Image.Image, format: str = "JPEG") -> str:
        """将PIL图像转换为base64编码"""
        buffered = BytesIO()
        
        # 如果是PNG且有透明度，保持PNG格式
        if format.upper() == "PNG" or (hasattr(image, 'mode') and image.mode == 'RGBA'):
            format = "PNG"
        else:
            format = "JPEG"
            # 如果图像有透明度，转换为RGB
            if image.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
        
        image.save(buffered, format=format, quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{img_str}"
    
    def get_system_prompt(self, analysis_type: str) -> str:
        """根据分析类型获取系统提示"""
        system_prompts = {
            "comprehensive": """你是一个专业的图像分析专家，具有敏锐的视觉洞察力和丰富的知识背景。
请对图像进行全面深入的分析，包括但不限于：

📋 **基本信息识别**
- 主要对象和元素
- 场景和环境描述
- 颜色、光线、构图分析

🎨 **艺术和设计角度**
- 视觉风格和美学特点
- 构图技巧和视觉平衡
- 色彩搭配和氛围营造

🔍 **细节观察**
- 重要的细节和特征
- 背景信息和上下文
- 可能的象征意义或隐含信息

📊 **结构化输出**
请使用Markdown格式，包含适当的emoji和分段，让内容易读且美观。""",

            "simple": """你是一个图像识别助手，请简洁准确地描述图像的主要内容。

包括：
- 主要对象
- 场景描述  
- 关键特征

请用简洁的语言，重点突出最重要的信息。""",

            "detailed": """你是一个详细的图像分析专家，请提供深入全面的图像分析报告。

请从以下维度进行分析：
1. 视觉元素识别
2. 场景和环境分析
3. 技术参数评估（如可见）
4. 艺术和美学评价
5. 可能的用途和意义
6. 改进建议（如适用）

请使用专业的术语和结构化的格式。""",

            "creative": """你是一个富有创意的视觉故事家，请用生动有趣的方式描述这张图片。

可以包括：
- 富有想象力的场景描述
- 可能的故事背景
- 情感和氛围感受
- 创意联想和类比

让描述既准确又富有艺术感染力。"""
        }
        
        return system_prompts.get(analysis_type, system_prompts["comprehensive"])
    
    def analyze_image_stream(self, image: Image.Image, analysis_type: str = "comprehensive") -> Generator[str, None, None]:
        """流式分析图片内容"""
        
        system_prompt = self.get_system_prompt(analysis_type)
        
        user_prompt = f"""请对这张图片进行{analysis_type}分析。

请确保：
1. 准确识别图片中的所有重要元素
2. 使用清晰的结构和格式
3. 包含适当的emoji和标题来美化排版
4. 提供有价值的洞察和信息

请开始你的分析："""
        
        # 编码图像
        base64_image = self.encode_image_to_base64(image)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_image,
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.3,
                top_p=0.9,
                max_tokens=4096,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise Exception(f"图像分析时发生错误: {str(e)}")
    
    def analyze_image(self, image: Image.Image, analysis_type: str = "comprehensive") -> str:
        """非流式分析图片内容"""
        full_text = ""
        for chunk in self.analyze_image_stream(image, analysis_type):
            full_text += chunk
        return full_text
    
    # +++ 新增：标准化的任务执行入口，用于被MCP调用 +++
    def execute_task(self, task_description: str, context: dict) -> str:
        """
        作为工具被MCP调用时执行的具体任务。
        task_description: MCP分配的具体指令, e.g., "分析这张图片里的主要物体"
        context: 任务上下文，必须包含 image_path 或 image_url
        """
        print(f"👁️ VisionLLMClient 正在执行: {task_description}")
        
        # 从上下文中获取图片信息
        # 假设MCP会把需要处理的文件路径放入context
        image_path = context.get("image_path")
        if not image_path:
            return "错误：上下文中未找到需要分析的图片路径(image_path)。"
            
        try:
            image = Image.open(image_path)
            # 从任务描述中解析分析类型，如果没指定，就用默认的
            analysis_type = "comprehensive"
            if "简单" in task_description or "简洁" in task_description:
                analysis_type = "simple"
            elif "详细" in task_description:
                analysis_type = "detailed"
            
            # 调用已有的非流式方法来完成任务
            result = self.analyze_image(image, analysis_type)
            print(f"👁️ VisionLLMClient 完成任务。")
            return result
        except Exception as e:
            return f"图像分析执行失败: {e}"
        
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


