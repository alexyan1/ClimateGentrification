# preprocessing.py

# preprocessing tweets so that we can classify them using the nlp

"""
steps:
convert all text to lowercase
remove Stopwords: i.e. ("is", "the")
tokenization: split text into individual words.
lemmatization/stemming: reduce words to their root form (i.e. "helping" turns into "help").
process or remove emojis (idk yet we could flag certain emojis as important)
"""

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# downloading
# nltk.download('punkt_tab')
# nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# spaCy model for lemmatization
nlp = spacy.load("en_core_web_sm")

def process_tweet(tweet):
    # tokenization
    tokens = nltk.word_tokenize(tweet)
    # process with spacy
    doc = nlp(" ".join(tokens))
    # lemmatize and remove stop words
    lemmatized_tokens = [token.lemma_ for token in doc if token.text.lower() not in stop_words]

    return lemmatized_tokens

example_tweet = "hello world! Helping with nlp processing"
print(process_tweet(example_tweet))
