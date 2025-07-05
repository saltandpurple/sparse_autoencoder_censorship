from typing import Dict, Any, List, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import BedrockChat
from langchain_core.runnables import add
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import boto3
import os

class ExperimentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    turn_count: int
    current_speaker: str
    max_turns: int

DEFAULT_REGION = "us-east-1"

class PromptGenerator:
    def __init__(self):
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=DEFAULT_REGION
        )
        self.lmstudio_client = ChatOpenAI(
            base_url="http://192.168.5.61/v1",
            model_kwargs={"temperature": 0.7}
            )

        self.prompt_generator = BedrockChat(
            client=self.bedrock_client,
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            model_kwargs={"temperature": 0.7}
        )
        self.subject = BedrockChat(
            client=self.bedrock_client,
            model_id="deepseek-r1:7b",
            model_kwargs={"temperature": 0.7}
        )
        self.evaluator = BedrockChat(
            client=self.bedrock_client,
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            model_kwargs={"temperature": 0.7}
        )


if __name__ == "__main__":
    prompt_generator = PromptGenerator()