from nltk.tag.stanford import POSTagger
from nltk.corpus import stopwords
from collections import defaultdict
from src.utils.sentiwordnet import SentiWordNetCorpusReader, SentiSynset
import sys

__author__ = 'Jasneet Sabharwal'

POS_TAGGER = POSTagger('/Users/KonceptGeek/Documents/Projects/Kaggle'
                       '/SentimentAnalysis/lib/english-bidirectional-distsim'
                       '.tagger',
                       '/Users/KonceptGeek/Documents/Projects/Kaggle'
                       '/SentimentAnalysis/lib/stanford-postagger.jar')
SENTI_WORDNET = SentiWordNetCorpusReader('/Users/KonceptGeek/Documents/'
                                         'Projects/Kaggle/SentimentAnalysis/'
                                         'lib/SentiWordNet_3.0.0_20130122.txt')


def _pos_features(tokens):
    pos_tags = POS_TAGGER.tag(tokens)
    pos_tags = [(word,tag) for (word,tag) in pos_tags if not word in
                stopwords.words('english')]
    features = defaultdict(int)
    for (word, tag) in pos_tags:
        if 'NN' in tag:
            features['countNoun'] += 1
        elif 'VB' in tag:
            features['countVerb'] += 1
        elif 'RB' in tag:
            features['countAdv'] += 1
        elif 'JJ' in tag:
            features['countAdj'] += 1
    return features



def extract_features(tokens):
    features = _pos_features(tokens)
    return features