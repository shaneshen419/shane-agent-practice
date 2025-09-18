
# ==============================================================================
# +++ 新增: MCP Agent 的完整实现 +++
# ==============================================================================

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
from utils.travel_planner_llm import TravelPlannerLLM
from utils.vision_llm_client import VisionLLMClient
from utils.readme_client import ReadmeViewerLLM

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
