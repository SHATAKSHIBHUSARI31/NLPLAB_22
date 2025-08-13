import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('stopwords')

documents = [
    "I love machine learning",
    "Machine learning is great for AI",
    "I love AI",
    "AI is great"
]

stop_words = set(stopwords.words('english'))

def nltk_tokenizer(text):
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t not in stop_words and t not in string.punctuation]

vectorizer_nltk = CountVectorizer(tokenizer=nltk_tokenizer)
X_nltk = vectorizer_nltk.fit_transform(documents)

print("Vocabulary (NLTK):", vectorizer_nltk.vocabulary_)
print("\nDense Matrix (NLTK):\n", X_nltk.toarray())
print("\nFeature Names (NLTK):", vectorizer_nltk.get_feature_names_out())
