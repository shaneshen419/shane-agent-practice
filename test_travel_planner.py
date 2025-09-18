import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.llm_client import TravelPlannerLLM

def test_travel_planner():
    # 初始化TravelPlannerLLM实例
    # 注意：这里需要提供有效的API密钥和base_url
    travel_planner = TravelPlannerLLM(
        api_key="sk-***",  # 替换为有效的API密钥
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo"
    )
    
    # 测试任务描述
    task_description = "帮我安排一下北京国庆七天旅游计划"
    
    # 解析任务以获取目的地和天数
    parsed_info = travel_planner._parse_task_with_llm(task_description)
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
            print(f"错误：无法从任务 '{task_description}' 中解析出目的地和天数。")
            return
    
    # 转换天数为整数
    num_days = int(days)
    
    # 使用流式方法生成行程并逐块打印
    print(f"✈️ 生成{num_days}天的{destination}旅游计划...")
    try:
        for chunk in travel_planner.generate_itinerary_stream(destination, num_days):
            print(chunk, end='', flush=True)
        print("\n\n✈️ 旅游计划生成完成！")
    except Exception as e:
        print(f"\n\n⚠️ 生成行程时发生错误: {str(e)}")
        print("\n\n🔄 使用模拟行程...")
        # 模拟流式输出
        mock_itinerary = f"**Day 1: 探索{destination}**\n\n**上午：**\n- **9:00 AM - 抵达{destination}**  \n  欢迎来到{destination}!\n\n**午餐：**\n- **12:00 PM - 特色餐厅**  \n  推荐尝试当地特色菜。\n\n**下午：**\n- **2:00 PM - 市中心游览**  \n  参观主要景点。\n\n**晚餐：**\n- **6:00 PM - 餐厅推荐**  \n  品尝当地美食。\n\n---\n\n**Day 2: 深度体验{destination}**\n\n**上午：**\n- **9:00 AM - 历史博物馆**  \n  了解{destination}的历史文化。\n\n**午餐：**\n- **12:00 PM - 文化街区**  \n  在文化街区享用午餐。\n\n**下午：**\n- **2:00 PM - 自然风光**  \n  游览{destination}的自然景观。\n\n**晚餐：**\n- **6:00 PM - 夜市小吃**  \n  体验当地夜市文化。\n\n---\n\n**Day 3: 周边游{destination}**\n\n**上午：**\n- **8:00 AM - 前往周边景点**  \n  前往{destination}周边的著名景点。\n\n**午餐：**\n- **12:00 PM - 景区餐厅**  \n  在景区内享用午餐。\n\n**下午：**\n- **2:00 PM - 继续游览**  \n  继续探索周边景点。\n\n**晚餐：**\n- **6:00 PM - 返回市区**  \n  返回市区享用晚餐。\n\n---\n\n**总结：**\n\n希望这份行程能帮助你更好地体验{destination}的魅力！"
        # 模拟流式输出，每次返回一小段文本
        for i in range(0, len(mock_itinerary), 100):
            print(mock_itinerary[i:i+100], end='', flush=True)
        print("\n\n✈️ 模拟旅游计划生成完成！")

if __name__ == "__main__":
    test_travel_planner()