import asyncio  
from fastmcp import Client  

async def main():  
    # Point the client at your server file  
    client = Client("http://127.0.0.1:8000/mcp")  
      
    # Connect to the server  
    async with client:  
        # List available tools  
        tools = await client.list_tools()  
        print("Available tools:")  
        for tool in tools:  
            print(f"  - {tool.name}: {tool.description}")  
          
        print("\n" + "="*50 + "\n")  
          
        # Call the weather tool  
        result = await client.call_tool(  
            "get_weather",   
            {"city": "Tokyo"}  
        )  
        print(f"Rag summarize result: {result}")
        result = await client.call_tool(  
            "rag_summarize",   
            {"query": "机器人","collection_name":"agent_rag"}  
        )  
        print(f"answer: {result}")  
        result = await client.call_tool(  
            "get_llm_answer",   
            {"question": "机器人是什么?","collection_name":"agent_rag"}  
        )  
        print(f"LLManswer: {result}")  
  

if __name__ == "__main__":  
     asyncio.run(main())