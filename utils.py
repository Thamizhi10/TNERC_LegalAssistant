import re
from deep_translator import GoogleTranslator

def split_into_chunks(text, chunk_size=800):
    text = re.sub(r'\s+', ' ', text)
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def translate_to_english(text):
    try:
        if all(ord(c) < 128 for c in text):
            return text
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text