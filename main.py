import time
from pathlib import Path
from typing import Dict, Any, List, TypedDict, Annotated
from langchain_aws import ChatBedrock
from langchain_core.runnables import add
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, BaseMessage
from tools import *

class EvalDetails:
    evalPrompt: str
    model: str
    params: Dict[str, Any]
    feedback: str

class ExperimentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    output: str
    evaluation: EvalDetails

class CensorshipExperiment:
    def __init__(self):
        self.app = self.create_workflow()

    def run(self):
        self.app.invoke()

    def create_workflow(self):
        workflow = StateGraph(ExperimentState)
        workflow.add_node("execution", self.execute)
        workflow.add_node("evaluation", self.evaluate)
        workflow.set_entry_point("preprocessing")

        return workflow.compile()


    # todo: write prompt, implement
    def evaluate(self, state: ExperimentState) -> Dict[str, Any]:
        evaluation = state["evaluation"]
        llm = ChatBedrock(
            model = evaluation.model,
            model_kwargs=evaluation.params
        )
        return {
            "evaluation": ""
        }




if __name__ == "__main__":
    experiment = CensorshipExperiment()
    experiment.run()