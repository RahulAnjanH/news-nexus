import operator
from typing import Annotated,List,TypedDict
from langgraph.graph import StateGraph,END
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,AIMessage
from langchain_ollama import ChatOllama

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage],operator.add]
    researcher_data :List[str] # researcher findings
    chart_data :List[dict]  # After analysis chart is generated

def researcher_node(state:AgentState):
    print("\n---(Agent:Researcher) is gathering data---")
    last_message