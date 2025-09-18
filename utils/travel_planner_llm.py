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
