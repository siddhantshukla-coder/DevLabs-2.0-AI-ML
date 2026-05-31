import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from tools import search_games, get_game_details, get_game_reviews

load_dotenv()

def main():
    print("Starting Local Ollama Agent...")

    llm = ChatOllama(
        temperature=0.7, 
        model="llama3.1",
    )

    memory = MemorySaver()
    tools = [search_games, get_game_details, get_game_reviews]
    
    steam_agent = create_react_agent(
        llm,
        tools,
        checkpointer=memory
    )
    
    print("The Local Steam Agent is online\nType 'exit' or 'quit' to close the app.")
    
    config = {"configurable": {"thread_id": "steam_chat_session"}}

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("End of Conversation")
            break
            
        try:
            response = steam_agent.invoke(
                {"messages": [("human", user_input)]},
                config=config
            )
            
            final_message = response["messages"][-1].content
            
            if isinstance(final_message, list):
                final_message = final_message[0].get('text', str(final_message))
                
            print(f"\nAgent: {final_message}")

        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
