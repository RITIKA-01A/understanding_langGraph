from typing import TypedDict,List,Union
from langchain_core.messages import HumanMessage,AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages : List[Union[HumanMessage,AIMessage]]

llm = ChatGroq(model_name = "Gemma2-9b-it")

def process(state:AgentState)->AgentState:
    """this node will solve the request you input"""
    response = llm.invoke(state['messages'])

    state['messages'].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    print("CURRENT STATE",state['messages'])
    return state

graph = StateGraph(AgentState)
graph.add_node("process" , process)
graph.set_entry_point("process")
graph.set_finish_point("process")
agent = graph.compile()


conversation_history = []

user_input = input("enter :")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages" : conversation_history})
    conversation_history = result['messages']
    user_input = input("Enter : ")

with open("logging.txt","w") as file:
    file.write("Your conversation log:\n")

    for message in conversation_history:
        if isinstance(message,HumanMessage):
            file.write(f"you: {message.content}\n")
        elif isinstance(message,AIMessage):
            file.write(f"AI: {message.content}\n\n" )
            
    file.write("End of Conversaion")
print("Conversation saved to logging.txt")