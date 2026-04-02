from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import asyncio
import sys
from pathlib import Path
from langchain.agents import create_agent 
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  
from model.factory import chat_model

client = MultiServerMCPClient(
    {
        "task": {
            "url": "http://127.0.0.1:8000/mcp",
            "transport": "http",
        }
    }
)
async def main():
    tools = await client.get_tools()
    agent = create_agent(
        model=chat_model,
        tools=tools,
    )
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "机器人是什么？"}]}
    )
    print(response)
if __name__ == "__main__":
    asyncio.run(main())
