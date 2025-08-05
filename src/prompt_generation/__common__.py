from typing import List
from pydantic import BaseModel


class Evaluation(BaseModel):
    censored: bool = False
    censorship_category: str = ""

class Response(BaseModel):
    response: str = ""
    thought: str = ""
    response_embedding: str = ""
    Evaluation: str = ""

class Question(BaseModel):
    question: str
    response: Response
    embedding: List[float] = []
    id: str = ""

class Questionnaire(BaseModel):
    questions: List[Question]
    subject: str = ""
