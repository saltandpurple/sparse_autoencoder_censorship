import generation
from src.config import *
from typing import List
from datetime import datetime

BATCH_SIZE = 5

def retrieve_unanswered_questions(count: int) -> List[generation.Question]:
    results = collection.get(
        limit=count,
        where={
            "response": {"$eq": ""},
        },
        include=['metadatas', 'embeddings']
    )

    questions = []
    if results and results['metadatas']:
        for metadata in results['metadatas']:
            question = generation.Question(
                question=metadata['question'],
                response=generation.Response(),
                id=results['ids'][0],
            )
            questions.append(question)

    logging.info(f"Retrieved {len(questions)} questions for interrogation")
    return questions


def update_results(questionnaire: generation.Questionnaire) -> None:
    for question in questionnaire.questions:
        metadata = {
            "question": question.question,
            "subject": questionnaire.subject,
            "thought": question.response.thought,
            "response": question.response.response,
            "response_embedding": question.response.response_embedding,
            "censored": question.response.Evaluation.censored if hasattr(question.response.Evaluation, "censored") else False,
            "censorship_category": question.response.Evaluation.censorship_category if hasattr(question.response.Evaluation, "censorship_category") else "none",
            "timestamp": datetime.now().isoformat()
        }

        collection.update(
            ids=[question.id],
            metadatas=[metadata]
        )

def run():
    questions = retrieve_unanswered_questions(BATCH_SIZE)

    questionnaire = generation.Questionnaire(
        questions=questions,
        subject=subject.model_name
    )

    logging.info("Beginning interrogation...")
    generation.interrogate_subject(questionnaire.questions)

    logging.info("Generating response embeddings...")
    generation.generate_response_embeddings(questionnaire)

    logging.info("Evaluating responses...")
    generation.evaluate_responses(questionnaire.questions)

    logging.info("Storing results...")
    update_results(questionnaire)

    logging.info("Interrogation process completed!")


if __name__ == "__main__":
    run()
