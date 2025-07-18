from typing import TypedDict,List
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages : List[HumanMessage]

llm = ChatGroq(model_name='Gemma2-9b-it')

def process(state:AgentState)->AgentState:
    response = llm.invoke(state['messages'])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process" , process)
graph.set_entry_point("process")
graph.set_finish_point("process")
agent = graph.compile()

user_input = input("enter :")
while user_input != "exit":
    agent.invoke({"messages" : [HumanMessage(content=user_input)]})
    user_input = input("Enter : ")

