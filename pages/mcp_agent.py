# pages/mcp_agent.py
import streamlit as st
from config import config_manager
from utils.mcp_client import MCPAgentLLM

def show_page():
    page_config = config_manager.get_page_config("mcp_agent")
    st.title(f"{page_config.get('icon', '')} {page_config.get('title')}")
    st.caption(page_config.get('description'))

    api_key = config_manager.get_api_key("mcp_agent")
    base_url = config_manager.get_base_url("mcp_agent")
    
    if not api_key:
        st.error("❌ 请先在全局设置或MCP Agent配置中提供API密钥。")
        return

    # 获取MCP可用的工具配置
    tool_config = page_config.get("available_tools", {})
    st.sidebar.subheader("可用专家工具")
    for name, info in tool_config.items():
        st.sidebar.info(f"**{name}**: {info['description']}")

    goal = st.text_area("请输入您希望实现的复杂目标：", height=150)

    if st.button("🚀 开始执行", use_container_width=True):
        if not goal:
            st.warning("请输入您的目标。")
            return

        try:
            mcp_client = MCPAgentLLM(
                api_key=api_key, 
                base_url=base_url, 
                model=page_config.get("default_model"),
                tool_config=tool_config
            )
            
            with st.status("🤖 MCP 正在分析目标并制定计划...", expanded=True) as status:
                plan = mcp_client.plan(goal)
                status.update(label="✅ 计划制定完成!", state="complete")
                st.subheader("行动计划")
                st.json(plan)
            
            with st.spinner("⏳ MCP 正在调度专家执行计划..."):
                final_result = mcp_client.execute_plan(plan, config_manager)
                st.subheader("执行结果汇总")
                st.code(final_result, language='json')

        except Exception as e:
            st.error(f"执行过程中发生错误: {e}")

