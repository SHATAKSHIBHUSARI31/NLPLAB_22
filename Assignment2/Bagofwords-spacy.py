import spacy
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load("en_core_web_sm")

documents = [
    "I love machine learning",
    "Machine learning is great for AI",
    "I love AI"
]

def spacy_tokenizer(text):
    doc = nlp(text.lower())
    return [token.text for token in doc if not token.is_stop and not token.is_punct]

vectorizer_spacy = CountVectorizer(tokenizer=spacy_tokenizer)
X_spacy = vectorizer_spacy.fit_transform(documents)

print("Vocabulary (spaCy):", vectorizer_spacy.vocabulary_)
print("\nDense Matrix (spaCy):\n", X_spacy.toarray())
print("\nFeature Names (spaCy):", vectorizer_spacy.get_feature_names_out())
