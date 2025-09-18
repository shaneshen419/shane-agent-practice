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



# ==============================================================================
# +++ 新增: MCP Agent 的完整实现 +++
# ==============================================================================
class MCPAgentLLM(LLMClient):
    """
    Master Control Program, 负责规划和调度其他Agent。
    """
    def __init__(self, api_key: str, base_url: str = None, model: str = "qwen-max", tool_config: dict = None):
        super().__init__(api_key, base_url, model)
        self.tool_config = tool_config or {
            "travel_planner": {
                "description": "专业旅行规划师，能够制定详细的旅行计划、推荐景点和安排行程",
                "class": "TravelPlannerLLM", 
                "page": "travel_agent"
            },
            "vision_analyzer": {
                "description": "图像识别专家，能够分析图片内容、识别物体和场景",
                "class": "VisionLLMClient",
                "page": "image_recognition"  
            },
            "readme_viewer":{
                "description": "一个专业的文档介绍员，可以查看和解释项目说明文档内容。",
                "class": "ReadmeViewerLLM", # 假设这是图像识别的类名
                "page": "readme"
            }
        }
        # --- 修改: 动态加载工具类，确保类名与 config.yaml 中配置的一致 ---
        self.tool_classes = {
            "TravelPlannerLLM": TravelPlannerLLM,
            "VisionLLMClient": VisionLLMClient,
            "ReadmeViewerLLM": ReadmeViewerLLM
        }

    def execute_task(self, task_description: str, context: dict, config_manager=None) -> str:
        """
        作为工具被MCP调用时执行的具体任务。
        task_description: MCP分配的具体指令, e.g., "为去巴黎的5日游制定一个行程"
        context: 任务上下文，可能包含前置任务的结果
        config_manager: 配置管理器实例
        """
        print(f"🧠 MCPAgentLLM 正在执行: {task_description}")
        
        # 生成计划
        plan = self.plan(task_description, context)
        print(f"📋 生成的计划: {plan}")
        
        # 如果计划为空或失败，尝试使用LLM解析任务描述
        if not plan or (isinstance(plan, list) and len(plan) == 0):
            print("⚠️ 计划为空，尝试使用LLM解析任务描述...")
            parsed_info = self._parse_task_with_llm(task_description)
            if parsed_info and 'destination' in parsed_info and 'days' in parsed_info:
                destination = parsed_info['destination']
                days = parsed_info['days']
                if destination and days:
                    # 创建一个备用计划
                    plan = [{
                        "task_id": "task_1",
                        "description": f"为{destination}的{days}日游制定详细行程",
                        "tool": "travel_planner",
                        "dependencies": []
                    }]
                    print(f"✅ 使用LLM解析结果创建备用计划: {plan}")
        
        # 执行计划
        result = self.execute_plan(plan, config_manager, context)
        
        # 转换为JSON字符串返回
        return json.dumps(result, indent=2, ensure_ascii=False)

    def plan(self, goal: str, context: dict = None) -> list:
        """
        使用 LLM 将用户目标分解为具体任务步骤。
        """
        tool_descriptions = "".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tool_config.items()
        ])
        
        context_str = f"初始上下文: {json.dumps(context, ensure_ascii=False)}\n" if context else ""

        planning_prompt = f"""
你是超级智能助理 (MCP)，根据用户目标生成执行计划。

可用工具:
{tool_descriptions}

{context_str}
用户目标: "{goal}"

请严格按照以下JSON格式返回：

{{
"plan": [
    {{
    "task_id": "task_1",
    "description": "具体任务描述",
    "tool": "travel_planner",
    "dependencies": []
    }}
]
}}

重要要求：
1. 必须返回有效的JSON对象
2. 顶层必须有"plan"键
3. plan的值必须是数组
4. 每个任务必须包含所有4个字段
5. dependencies必须是数组格式
6. tool必须是: {', '.join(self.tool_config.keys())}

示例：对于旅行规划，返回：
{{
"plan": [
    {{
    "task_id": "task_1", 
    "description": "为合肥的2日游制定详细行程",
    "tool": "travel_planner",
    "dependencies": []
    }}
]
}}
"""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"🤖 尝试生成计划 (第{attempt + 1}次)...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": planning_prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content.strip()
                print(f"🤖 LLM 原始返回: {content}")
                
                # 尝试解析JSON
                try:
                    plan_data = json.loads(content)
                    print(f"✅ JSON解析成功: {type(plan_data)}")
                except json.JSONDecodeError as e:
                    print(f"❌ JSON 解析失败: {e}")
                    if attempt == max_retries - 1:
                        return self._create_fallback_plan(goal, context)
                    continue
                
                # 验证结构
                if not isinstance(plan_data, dict):
                    print(f"❌ 返回不是字典: {type(plan_data)}")
                    if attempt == max_retries - 1:
                        return self._create_fallback_plan(goal, context)
                    continue
                    
                if "plan" not in plan_data:
                    print("❌ 缺少 'plan' 键")
                    if attempt == max_retries - 1:
                        return self._create_fallback_plan(goal, context)
                    continue
                    
                plan_list = plan_data["plan"]
                if not isinstance(plan_list, list):
                    print(f"❌ 'plan' 不是数组: {type(plan_list)}")
                    if attempt == max_retries - 1:
                        return self._create_fallback_plan(goal, context)
                    continue
                
                # 验证每个任务
                validated_plan = []
                for i, task in enumerate(plan_list):
                    if not isinstance(task, dict):
                        print(f"❌ 任务 {i} 不是字典: {type(task)}")
                        continue
                        
                    required_fields = ['task_id', 'description', 'tool', 'dependencies']
                    missing_fields = [f for f in required_fields if f not in task]
                    if missing_fields:
                        print(f"❌ 任务 {i} 缺少字段: {missing_fields}")
                        continue
                    
                    # 验证并修正字段类型
                    task['task_id'] = str(task['task_id'])
                    task['description'] = str(task['description'])
                    task['tool'] = str(task['tool'])
                    
                    # 修正 dependencies
                    deps = task.get("dependencies", [])
                    if isinstance(deps, (int, str)):
                        task["dependencies"] = [] if deps in [0, "0", ""] else [str(deps)]
                    elif not isinstance(deps, list):
                        task["dependencies"] = []
                    else:
                        # 确保依赖项都是字符串
                        task["dependencies"] = [str(d) for d in deps]
                    
                    # 验证工具名称
                    if task["tool"] not in self.tool_config:
                        print(f"❌ 未知工具: {task['tool']}")
                        continue
                        
                    validated_plan.append(task)
                
                if validated_plan:
                    print(f"✅ 验证通过，共 {len(validated_plan)} 个任务")
                    return validated_plan
                else:
                    print("❌ 没有有效任务")
                    if attempt == max_retries - 1:
                        return self._create_fallback_plan(goal, context)
                        
            except Exception as e:
                print(f"❌ 计划生成失败 (尝试{attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return self._create_fallback_plan(goal, context)
        
        return self._create_fallback_plan(goal, context)



    def _create_fallback_plan(self, goal: str, context: dict = None) -> list:
        """创建备用计划，当LLM生成失败时使用"""
        print("🔄 生成备用计划...")
        
        # 根据目标关键词判断需要什么工具
        goal_lower = goal.lower()
        
        if any(keyword in goal_lower for keyword in ["旅行", "旅游", "行程", "travel"]):
            return [{
                "task_id": "task_1",
                "description": f"根据用户目标制定旅行计划: {goal}",
                "tool": "travel_planner",  # 修改：使用正确的工具名称
                "dependencies": []
            }]
        elif any(keyword in goal_lower for keyword in ["图片", "图像", "分析", "识别", "image"]):
            return [{
                "task_id": "task_1", 
                "description": f"分析用户上传的图像: {goal}",
                "tool": "vision_analyzer",  # 修改：使用正确的工具名称
                "dependencies": []
            }]
        else:
            # 默认使用旅行规划
            return [{
                "task_id": "task_1",
                "description": f"处理用户请求: {goal}",
                "tool": "travel_planner",  # 修改：使用正确的工具名称
                "dependencies": []
            }]


    def execute_plan(self, plan: list, config_manager, initial_context: dict = None) -> str:
        """
        按顺序执行计划，并调度相应的Agent工具。
        """
        if not isinstance(plan, list) or not plan:
            return json.dumps({"error": "计划为空或格式不正确"}, indent=2, ensure_ascii=False)
            
        context = initial_context or {}
        results = {"execution_summary": [], "tasks": {}}
        
        for i, task in enumerate(plan):
            if not isinstance(task, dict):
                error_msg = f"任务 {i} 格式错误: {task}"
                print(f"❌ {error_msg}")
                results["execution_summary"].append(error_msg)
                continue
                
            if 'task_id' not in task:
                error_msg = f"任务 {i} 缺少 task_id: {task}"
                print(f"❌ {error_msg}")
                results["execution_summary"].append(error_msg)
                continue
                
            task_id = task['task_id']
            description = task.get('description', '无描述')
            tool_name = task.get('tool', '')
            
            print(f"🤖 MCP: 开始执行 {task_id} - {description}")
            print(f"🔧 使用工具: {tool_name}")
            
            if not tool_name or tool_name not in self.tool_config:
                error_msg = f"任务 {task_id} 的工具 '{tool_name}' 不存在，可用工具: {list(self.tool_config.keys())}"
                print(f"❌ {error_msg}")
                results["tasks"][task_id] = {"error": error_msg}
                results["execution_summary"].append(error_msg)
                continue

            try:
                # 获取工具配置
                tool_info = self.tool_config[tool_name]
                tool_class_name = tool_info["class"]
                tool_page_config_key = tool_info["page"]
                
                print(f"🔍 工具类名: {tool_class_name}")
                print(f"📄 配置页面: {tool_page_config_key}")
                
                # 获取API配置
                api_key = config_manager.get_api_key(tool_page_config_key)
                base_url = config_manager.get_base_url(tool_page_config_key)
                
                try:
                    page_config = config_manager.get_page_config(tool_page_config_key)
                    model = page_config.get("default_model", "qwen-turbo")
                except:
                    model = "qwen-turbo"
                    
                print(f"⚙️ API配置: base_url={base_url}, model={model}")

                # 获取工具类并验证
                if tool_class_name not in self.tool_classes:
                    available_classes = list(self.tool_classes.keys())
                    error_msg = f"未找到工具类 {tool_class_name}，可用类: {available_classes}"
                    print(f"❌ {error_msg}")
                    results["tasks"][task_id] = {"error": error_msg}
                    results["execution_summary"].append(error_msg)
                    continue
                
                tool_class = self.tool_classes[tool_class_name]
                print(f"✅ 找到工具类: {tool_class}")
                    
                # 处理依赖
                task_context = context.copy()
                dependencies = task.get("dependencies", [])
                if isinstance(dependencies, (int, str)):
                    dependencies = [] if dependencies in [0, "0"] else [f"task_{dependencies}"]
                elif not isinstance(dependencies, list):
                    dependencies = []
                    
                # 合并依赖结果
                for dep_id in dependencies:
                    if dep_id in results["tasks"] and "result" in results["tasks"][dep_id]:
                        task_context[f"{dep_id}_result"] = results["tasks"][dep_id]["result"]

                # 实例化工具
                print(f"🏗️ 正在实例化工具...")
                tool_instance = tool_class(api_key=api_key, base_url=base_url, model=model)
                print(f"✅ 工具实例化成功: {type(tool_instance)}")
                
                # 验证工具实例是否有execute_task方法
                if not hasattr(tool_instance, 'execute_task'):
                    error_msg = f"工具实例 {tool_class_name} 没有 execute_task 方法"
                    print(f"❌ {error_msg}")
                    results["tasks"][task_id] = {"error": error_msg}
                    results["execution_summary"].append(error_msg)
                    continue
                
                # 执行任务
                print(f"🚀 开始执行任务...")
                result = tool_instance.execute_task(description, task_context)
                
                results["tasks"][task_id] = {"result": result}
                results["execution_summary"].append(f"✅ {task_id} 执行成功")
                print(f"✅ {task_id} 执行成功")
                
            except Exception as e:
                error_msg = f"执行 {task_id} 时发生错误: {str(e)}"
                print(f"❌ {error_msg}")
                print(f"🔍 错误详情: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()  # 打印完整的错误堆栈
                results["tasks"][task_id] = {"error": error_msg}
                results["execution_summary"].append(error_msg)
        
        return json.dumps(results, indent=2, ensure_ascii=False)


    def _parse_task_with_llm(self, task_description: str) -> dict:
        """使用LLM来智能解析任务描述中的目的地和天数信息"""
        try:
            parse_prompt = f"""
请从以下任务描述中提取旅行规划信息：

任务描述: "{task_description}"

请返回JSON格式的结果：
{{
    "destination": "目的地城市名称",
    "days": 天数（数字）
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
            # 当LLM解析失败时，使用正则表达式作为备选方案
            import re
            print("🔄 使用正则表达式作为备选方案...")
            
            # 尝试提取目的地
            destination_match = re.search(r'[去往|前往|到|去]?([\u4e00-\u9fa5]{2,10})[旅游|旅行|游玩]?', task_description)
            destination = destination_match.group(1) if destination_match else None
            
            # 尝试提取天数
            days_match = re.search(r'(\d+)[天日]', task_description)
            days = days_match.group(1) if days_match else None
            
            result = {
                "destination": destination,
                "days": days
            }
            print(f"🧠 正则表达式解析结果: {result}")
            return result
    def debug_tool_classes(self):
        """调试工具类配置"""
        print("🔍 调试工具类配置:")
        print(f"tool_classes 类型: {type(self.tool_classes)}")
        print(f"tool_classes 内容: {self.tool_classes}")
        
        for class_name, class_obj in self.tool_classes.items():
            print(f"  {class_name}: {class_obj}")
            print(f"    类型: {type(class_obj)}")
            if hasattr(class_obj, '__name__'):
                print(f"    名称: {class_obj.__name__}")
            
            # 尝试实例化测试
            try:
                test_instance = class_obj(api_key="test", base_url="test", model="test")
                print(f"    ✅ 可以实例化")
                print(f"    实例类型: {type(test_instance)}")
                print(f"    有execute_task方法: {hasattr(test_instance, 'execute_task')}")
            except Exception as e:
                print(f"    ❌ 实例化失败: {e}")
