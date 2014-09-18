from nltk.corpus import stopwords
import nltk, string

__author__ = 'Jasneet Sabharwal'


def pre_process(phrase):
    #phrase = phrase.lower()
    #phrase = phrase.translate(None, string.punctuation)
    tokens = nltk.word_tokenize(phrase)
    #clean_tokens = [token for token in tokens if not token in stopwords.words(
    #    'english')]
    return tokens