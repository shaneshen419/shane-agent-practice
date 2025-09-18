import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.llm_client import MCPAgentLLM
from config.config_manager import ConfigManager

def test_mcp_agent():
    # 创建配置管理器实例
    config_manager = ConfigManager()
    
    # 创建MCPAgentLLM实例
    mcp_agent = MCPAgentLLM(
        api_key="sk-xxx",  # 请替换为有效的API密钥
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 请替换为有效的base_url
        model="qwen-turbo"
    )
    
    # 测试任务描述
    task_description = "为合肥的4天游制定详细行程，包括景点推荐、餐饮建议及活动安排"
    
    # 执行任务
    result = mcp_agent.execute_task(task_description, {}, config_manager)
    print("执行结果:")
    print(result)

if __name__ == "__main__":
    test_mcp_agent()