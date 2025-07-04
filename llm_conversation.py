import os
from typing import Dict, Any, List, TypedDict
from langchain_community.chat_models import BedrockChat
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessageGraph
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import boto3
from dotenv import load_dotenv

load_dotenv()


class ConversationState(TypedDict):
    messages: List[BaseMessage]
    turn_count: int
    current_speaker: str
    max_turns: int


class LLMConversation:
    def __init__(self, 
                 model_id_1: str = "anthropic.claude-3-sonnet-20240229-v1:0",
                 model_id_2: str = "anthropic.claude-3-haiku-20240307-v1:0",
                 region_name: str = "us-east-1",
                 max_turns: int = 10):
        
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name
        )
        
        self.llm1 = BedrockChat(
            client=self.bedrock_client,
            model_id=model_id_1,
            model_kwargs={"temperature": 0.7, "max_tokens": 1000}
        )
        
        self.llm2 = BedrockChat(
            client=self.bedrock_client,
            model_id=model_id_2,
            model_kwargs={"temperature": 0.7, "max_tokens": 1000}
        )
        
        self.max_turns = max_turns
        self.conversation_graph = self._create_conversation_graph()
    
    def _create_conversation_graph(self) -> StateGraph:
        graph = StateGraph(ConversationState)
        
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
    
    def _llm1_turn(self, state: ConversationState) -> ConversationState:
        response = self.llm1.invoke(state["messages"])
        state["messages"].append(response)
        state["turn_count"] += 1
        state["current_speaker"] = "llm1"
        return state
    
    def _llm2_turn(self, state: ConversationState) -> ConversationState:
        response = self.llm2.invoke(state["messages"])
        state["messages"].append(response)
        state["turn_count"] += 1
        state["current_speaker"] = "llm2"
        return state
    
    def _should_continue(self, state: ConversationState) -> str:
        if state["turn_count"] >= state["max_turns"]:
            return "end"
        
        if state["current_speaker"] == "llm1":
            return "llm2"
        else:
            return "llm1"
    
    def start_conversation(self, initial_prompt: str) -> List[BaseMessage]:
        initial_state = ConversationState(
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


def main():
    conversation = LLMConversation(max_turns=6)
    
    initial_prompt = """You are having a conversation with another AI. 
    The topic is: 'The future of artificial intelligence and its impact on society.'
    Please share your thoughts and engage in a thoughtful discussion."""
    
    print("Starting LLM conversation...")
    messages = conversation.start_conversation(initial_prompt)
    
    print("\nConversation Results:")
    print("=" * 60)
    conversation.print_conversation(messages)


if __name__ == "__main__":
    main()