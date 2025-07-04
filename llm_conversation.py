from typing import Dict, Any, List, TypedDict, Annotated
from langchain_community.chat_models import BedrockChat
from langchain_core.runnables import add
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessageGraph
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import boto3

class ExperimentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    turn_count: int
    current_speaker: str
    max_turns: int

DEFAULT_REGION = "us-east-1"
INTERROGATOR_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
SUBJECT_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"

class CensorshipExperiment:
    def __init__(self):
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=DEFAULT_REGION
        )
        
        self.interrogator = BedrockChat(
            client=self.bedrock_client,
            model_id=INTERROGATOR_MODEL,
            model_kwargs={"temperature": 0.7}
        )
        
        self.subject = BedrockChat(
            client=self.bedrock_client,
            model_id=SUBJECT_MODEL,
            model_kwargs={"temperature": 0.7}
        )
        
        self.conversation_graph = self.create_state_graph()
    
    def create_state_graph(self) -> StateGraph:
        graph = StateGraph(ExperimentState)
        
        graph.add_node("llm1_turn", self._llm1_turn)
        graph.add_node("llm2_turn", self._llm2_turn)
        
        graph.add_conditional_edges(
            "llm1_turn",
            self._should_continue,
            {
                "llm2": "llm2_turn",
                "end": END
            }
        )
        
        graph.add_conditional_edges(
            "llm2_turn", 
            self._should_continue,
            {
                "llm1": "llm1_turn",
                "end": END
            }
        )
        
        graph.set_entry_point("llm1_turn")
        
        return graph.compile()
    
    def _llm1_turn(self, state: ExperimentState) -> ExperimentState:
        response = self.llm1.invoke(state["messages"])
        state["messages"].append(response)
        state["turn_count"] += 1
        state["current_speaker"] = "llm1"
        return state
    
    def _llm2_turn(self, state: ExperimentState) -> ExperimentState:
        response = self.llm2.invoke(state["messages"])
        state["messages"].append(response)
        state["turn_count"] += 1
        state["current_speaker"] = "llm2"
        return state
    
    def _should_continue(self, state: ExperimentState) -> str:
        if state["turn_count"] >= state["max_turns"]:
            return "end"
        
        if state["current_speaker"] == "llm1":
            return "llm2"
        else:
            return "llm1"
    
    def start_conversation(self, initial_prompt: str) -> List[BaseMessage]:
        initial_state = ExperimentState(
            messages=[HumanMessage(content=initial_prompt)],
            turn_count=0,
            current_speaker="",
            max_turns=self.max_turns
        )
        
        final_state = self.conversation_graph.invoke(initial_state)
        return final_state["messages"]
    
    def print_conversation(self, messages: List[BaseMessage]):
        for i, message in enumerate(messages):
            if isinstance(message, HumanMessage):
                print(f"Initial Prompt: {message.content}")
            elif isinstance(message, AIMessage):
                speaker = "LLM1" if i % 2 == 1 else "LLM2"
                print(f"{speaker}: {message.content}")
            print("-" * 50)


if __name__ == "__main__":
    conversation = CensorshipExperiment()
    messages = conversation.start_conversation()