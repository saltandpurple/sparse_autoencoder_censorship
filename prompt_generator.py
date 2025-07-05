from typing import Dict, Any, List, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
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
        # clients
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=DEFAULT_REGION
        )

        # models
        self.prompt_generator = ChatBedrock(
            client=self.bedrock_client,
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            model_kwargs={"temperature": 0.7}
        )
        self.subject = ChatOpenAI(
            base_url="http://192.168.178.61:1234/v1",
            model="deepseek/deepseek-r1-0528-qwen3-8b",
            temperature=0.7
        )
        self.evaluator = ChatBedrock(
            client=self.bedrock_client,
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            model_kwargs={"temperature": 0.7}
        )

    def test_local(self):
        prompt = "What is the capital of France?"
        response = self.subject.invoke([HumanMessage(content=prompt)])
        print(response.content)

if __name__ == "__main__":
    prompt_generator = PromptGenerator()
    prompt_generator.test_local()