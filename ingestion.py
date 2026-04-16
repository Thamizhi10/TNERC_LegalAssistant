import os
import fitz
from utils import split_into_chunks, translate_to_english
from logger import get_logger

logger = get_logger()

BASE_PATH = "data/rulings/Ombudsman Orders/subject wise Orders up to 2020"
REGULATIONS_PATH = "data/regulations"

def ingest_regulations():
    all_chunks = []

    for file in os.listdir(REGULATIONS_PATH):
        if not file.lower().endswith(".pdf"):
            continue

        logger.info(f"Processing regulation: {file}")

        file_path = os.path.join(REGULATIONS_PATH, file)

        try:
            doc = fitz.open(file_path)

            full_text = ""
            for page in doc:
                full_text += page.get_text()

            raw_chunks = split_into_chunks(full_text)

            for chunk in raw_chunks:
                translated_chunk = translate_to_english(chunk)

                all_chunks.append({
                    "text": translated_chunk,
                    "original_text": chunk,
                    "type": "regulation",
                    "source": file
                })

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    return all_chunks


def ingest_subject(subject):
    subject_path = os.path.join(BASE_PATH, subject)
    all_chunks = []

    for file in os.listdir(subject_path):
        if not file.endswith(".pdf"):
            continue

        logger.info(f"Processing: {subject} → {file}")

        file_path = os.path.join(subject_path, file)

        try:
            doc = fitz.open(file_path)

            full_text = ""
            for page in doc:
                full_text += page.get_text()

            raw_chunks = split_into_chunks(full_text)

            for chunk in raw_chunks:
                translated_chunk = translate_to_english(chunk)

                all_chunks.append({
                    "text": translated_chunk,
                    "original_text": chunk,
                    "type": "ruling",
                    "subject": subject,
                    "source": file
                })

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    return all_chunks