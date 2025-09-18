import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.llm_client import TravelPlannerLLM

class MockTravelPlannerLLM(TravelPlannerLLM):
    """模拟的TravelPlannerLLM类，用于测试正则表达式解析功能"""
    
    def _parse_task_with_llm(self, task_description: str) -> dict:
        """模拟LLM解析失败的情况"""
        print("⚠️ 模拟LLM解析失败")
        return {}
    
    def generate_itinerary_stream(self, destination: str, num_days: int):
        """模拟流式生成行程"""
        print(f"✈️ 模拟生成{num_days}天的{destination}旅游计划...")
        # 模拟流式输出
        itinerary = f"**Day 1: 探索{destination}**\n\n**上午：**\n- **9:00 AM - 抵达{destination}**  \n  欢迎来到{destination}!\n\n**午餐：**\n- **12:00 PM - 特色餐厅**  \n  推荐尝试当地特色菜。\n\n**下午：**\n- **2:00 PM - 市中心游览**  \n  参观主要景点。\n\n**晚餐：**\n- **6:00 PM - 餐厅推荐**  \n  品尝当地美食。\n\n---\n\n**Day 2: 深度体验{destination}**\n\n**上午：**\n- **9:00 AM - 历史博物馆**  \n  了解{destination}的历史文化。\n\n**午餐：**\n- **12:00 PM - 文化街区**  \n  在文化街区享用午餐。\n\n**下午：**\n- **2:00 PM - 自然风光**  \n  游览{destination}的自然景观。\n\n**晚餐：**\n- **6:00 PM - 夜市小吃**  \n  体验当地夜市文化。\n\n---\n\n**Day 3: 周边游{destination}**\n\n**上午：**\n- **8:00 AM - 前往周边景点**  \n  前往{destination}周边的著名景点。\n\n**午餐：**\n- **12:00 PM - 景区餐厅**  \n  在景区内享用午餐。\n\n**下午：**\n- **2:00 PM - 继续游览**  \n  继续探索周边景点。\n\n**晚餐：**\n- **6:00 PM - 返回市区**  \n  返回市区享用晚餐。\n\n---\n\n**总结：**\n\n希望这份行程能帮助你更好地体验{destination}的魅力！"
        # 模拟流式输出，每次返回一小段文本
        for i in range(0, len(itinerary), 100):
            yield itinerary[i:i+100]


def test_regex_parsing():
    # 初始化MockTravelPlannerLLM实例
    travel_planner = MockTravelPlannerLLM(
        api_key="your_api_key_here",  # 替换为有效的API密钥
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo"
    )
    
    # 测试任务描述
    task_description = "帮我安排一下北京国庆七天旅游计划"
    
    # 执行任务
    result = travel_planner.execute_task(task_description, {})
    
    # 打印结果
    print("执行结果:")
    print(result)

if __name__ == "__main__":
    test_regex_parsing()