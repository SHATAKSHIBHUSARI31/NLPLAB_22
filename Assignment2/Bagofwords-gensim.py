import gensim
from gensim import corpora
from gensim.utils import simple_preprocess

text2 = [""""The government announced a new economic policy", "Global markets reacted positively to the news", and "Experts predict economic growth in the coming year", we first tokenize the text into words and create a Dictionary that assigns each word a unique ID."""]

tokens2 = [[item for item in line.split()] for line in text2]
g_dict2 = corpora.Dictionary(tokens2)

print("The dictionary has: " + str(len(g_dict2)) + " tokens\n")
print(g_dict2.token2id)

g_bow2 = [g_dict2.doc2bow(token, allow_update=True) for token in tokens2]
print("Bag of Words : ", g_bow2)

text3 = [""""The government announced a new economic policy", "Global markets reacted positively to the news", and "Experts predict economic growth in the coming year", we first tokenize the text into words and create a Dictionary that assigns each word a unique ID."""]

tokens2 = [[item for item in line.split()] for line in text2]

g_dict3 = corpora.Dictionary([simple_preprocess(line) for line in text3])
g_bow3 = [g_dict3.doc2bow(simple_preprocess(line)) for line in text3]

print("\nDictionary : ")
for item in g_bow3:
    print([[g_dict3[id], freq] for id, freq in item])