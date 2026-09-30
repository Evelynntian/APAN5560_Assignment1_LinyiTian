import spacy

nlp = spacy.load("en_core_web_lg")


def calculate_embedding(input_word: str) -> list[float]:
    word = nlp(input_word)
    return word.vector.tolist()