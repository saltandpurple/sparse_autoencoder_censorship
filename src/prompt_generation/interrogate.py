import generation
from src.config import *
from typing import List


def retrieve_questions_for_interrogation() -> List[generation.Question]:
    results = collection.get(
        where={
            "response": {"$eq": ""},
        }
    )

    questions = []
    if results and results['metadatas']:
        for metadata in results['metadatas']:
            question = generation.Question(
                question=metadata['question'],
                response=generation.Response()
            )
            questions.append(question)

    logging.info(f"Retrieved {len(questions)} questions for interrogation")
    return questions


def run():
    questions = retrieve_questions_for_interrogation()

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
    generation.store_results(questionnaire)

    logging.info("Interrogation process completed!")


if __name__ == "__main__":
    run()
