from typing import TypedDict, Annotated, Literal
import os
from langgraph.graph import StateGraph, START, END, add_messages
from agents import init_state, travel_agent, reviewer_agent, summary_state

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    quality: Annotated[int, None]
    iterations: Annotated[int, None]
    final_script: Annotated[str, None]

def quality_gate_condition(state) -> Literal["travel_agent", "summary"]:

        if state['iterations'] >= int(os.getenv('MAX_ITERATIONS')):
            return 'summary'
        
        if state['quality'] < int(os.getenv('QUALITY_TRESHOLD')):
             return 'travel_agent'
        else:
            return 'summary'

def build_rag_graph():
    builder = StateGraph(AgentState)  # Passa a classe TypedDict como schema
    
    builder.add_node("init", init_state)
    builder.add_node("travel_agent", travel_agent)
    builder.add_node("reviewer_agent", reviewer_agent)
    builder.add_node("summary", summary_state)
    
    builder.add_edge(START, "init")
    builder.add_edge("init", "travel_agent")
    builder.add_edge("travel_agent", "reviewer_agent")
    builder.add_edge("summary", END)    

    builder.add_conditional_edges("reviewer_agent", quality_gate_condition)
    
    return builder.compile()
