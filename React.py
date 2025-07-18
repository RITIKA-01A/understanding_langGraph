from typing import Annotated,Sequence,TypedDict    
from typing import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage# foundational cls of all mesages
from langchain_core.messages import ToolMessage #
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,END
from langgraph.prebuilt import ToolNode

load_dotenv()

class Agentstate(TypedDict):
    messages:Annotated[Sequence[BaseMessage] , add_messages]


@tool
def add(a:int , b:int)->int:
    """This tool adds the two numbers"""
    return a+b

@tool
def substract(a:int , b:int):
    """substraction function"""
    return a-b

@tool
def multiplication(a:int , b:int):
    """multiplication tool"""
    return a*b

tools = [add,substract,multiplication]

llm = ChatGroq(model_name="Gemma2-9b-it").bind_tools(tools)

def model_call(state:Agentstate)->Agentstate:
    system_prompt=SystemMessage(content=
        "u are my assistant ,pls answer my query to the best or ur ability"
    )
    response = llm.invoke([system_prompt]+ state["messages"])

    ## Updating the state
    return {"messages":[response]}

def should_continue(state:Agentstate)->Agentstate:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        ## If there is no tool callig left then we end the loop
        return "end"
    else:
        return "continue"


graph = StateGraph(Agentstate)
graph.add_node("our_agent",model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools" , tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "end":END,
        "continue":"tools"
    }
)

graph.add_edge("tools" , "our_agent")
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message , tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages":[("user","get the sum of 2 and 10.get the difference between 45 and 23.Tell me a joke related to maths")]}
print_stream(app.stream(inputs,stream_mode="values"))
