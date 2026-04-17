import os
import fitz
import docx

from utils import split_into_chunks, translate_to_english
from logger import get_logger

logger = get_logger()

BASE_PATH = "data/rulings/Ombudsman Orders"
REGULATIONS_PATH = "data/regulations"


# ---------------- TEXT EXTRACTION ----------------
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text(file_path):
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)

    elif file_path.lower().endswith(".docx"):
        return extract_text_from_docx(file_path)

    return ""


# ---------------- REGULATIONS INGESTION ----------------
def ingest_regulations():
    all_chunks = []

    for file in os.listdir(REGULATIONS_PATH):

        if not (file.lower().endswith(".pdf") or file.lower().endswith(".docx")):
            continue

        logger.info(f"Processing regulation: {file}")

        file_path = os.path.join(REGULATIONS_PATH, file)

        try:
            full_text = extract_text(file_path)

            full_text = full_text.strip()
            if not full_text:
                continue

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


# ---------------- SUBJECT-WISE RULINGS INGESTION ----------------
def ingest_subject(subject):
    subject_path = os.path.join(BASE_PATH, subject)
    all_chunks = []

    for file in os.listdir(subject_path):

        if not (file.lower().endswith(".pdf") or file.lower().endswith(".docx")):
            continue

        logger.info(f"Processing: {subject} → {file}")

        file_path = os.path.join(subject_path, file)

        try:
            full_text = extract_text(file_path)

            full_text = full_text.strip()
            if not full_text:
                continue

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