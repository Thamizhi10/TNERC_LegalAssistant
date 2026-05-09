import re
from deep_translator import GoogleTranslator
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_regulation_chunks(text):
    text = re.sub(r'\s+', ' ', text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=[
            "\nCHAPTER",
            "\nChapter",
            "\nRegulation",
            "\nClause",
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )
    return splitter.split_text(text)

def split_ruling_chunks(text):
    text = re.sub(r'\s+', ' ', text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=[
            "\nORDER",
            "\nFindings",
            "\nConclusion",
            "\nArguments",
            "\nPrayer",
            "\nFacts",
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )
    return splitter.split_text(text)
'''def split_into_chunks(text, chunk_size=800):
    #text = re.sub(r'\s+', ' ', text)
    #return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    text = re.sub(r'\s+', ' ', text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    return splitter.split_text(text)'''


def translate_to_english(text):
    return text
    '''try:
        if all(ord(c) < 128 for c in text):
            return text
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text'''