import generation
from src.config import *


def reset_responses():
    results = collection.get(
        where={
            "response": {"$ne": ""}
        }
    )
    logging.info(f"Found {len(results['metadatas'])} questions with responses")
    questions = []
    for metadata in results['metadatas']:
        # create fresh Response object with empty data
        empty_response = generation.Response()

        question = generation.Question(
            question=metadata['question'],
            response=empty_response,
            embedding=metadata.get('embedding', [])
        )
        questions.append(question)

    # replace existing questions with updated ones (without response data)
    for i, question in enumerate(questions):
        question_id = results['ids'][i]

        updated_metadata = {
            "question": question.question,
            "subject": "",
            "thought": "",
            "response": "",
            "response_embedding": "",
            "censored": False,
            "censorship_category": "none",
            "timestamp": results['metadatas'][i].get('timestamp', '')
        }

        collection.update(
            ids=[question_id],
            metadatas=[updated_metadata]
        )

    logging.info(f"Reset {len(questions)} questions - cleared all response data")


if __name__ == "__main__":
    reset_responses()