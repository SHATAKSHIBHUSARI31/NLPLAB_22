import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
example_string = "My name is Shatakshi. My father name is Satish. My mother name is Aarti. I am in final year of Engineering."
print("Sentence Tokenization:")
sentences = sent_tokenize(example_string)
print(sentences)
print("\nWord Tokenization:")
words = word_tokenize(example_string)
print(words)
print("\nFiltered Words (without stopwords):")
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]
print(filtered_words)
print("\nLemmatization:")
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
print(lemmatized_words)