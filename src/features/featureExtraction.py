from nltk.tag.stanford import POSTagger
from nltk.corpus import stopwords
from collections import defaultdict
from src.utils.sentiwordnet import SentiWordNetCorpusReader, SentiSynset
from collections import defaultdict
import os

__author__ = 'Jasneet Sabharwal'

_POS_TAGGER_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'lib/english-bidirectional-distsim.tagger')
_POS_TAGGER_JAR_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'lib/stanford-postagger.jar')
_SENTI_WORDNET_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'lib/SentiWordNet_3.0.0_20130122.txt')

POS_TAGGER = POSTagger(_POS_TAGGER_MODEL_PATH, _POS_TAGGER_JAR_PATH)
SENTI_WORDNET = SentiWordNetCorpusReader(_SENTI_WORDNET_FILE_PATH)


def _pos_features(pos_tags):
    pos_tags = [(word, tag) for (word, tag) in pos_tags if not word.lower() in
                stopwords.words('english')]
    features = defaultdict(int)
    for (word, tag) in pos_tags:
        if 'NN' in tag:
            features['countNN'] += 1
        elif 'VB' in tag:
            features['countVB'] += 1
        elif 'RB' in tag:
            features['countRB'] += 1
        elif 'JJ' in tag:
            features['countJJ'] += 1
    return features


def _tag_records(allTokens):
    pos_tags = POS_TAGGER.batch_tag(allTokens)
    posTagMap = _parse_pos_tags(pos_tags)
    return posTagMap


def _parse_pos_tags(allPosTags):
    result = defaultdict(list)
    for posTag in allPosTags:
        words = []
        for word, tag in posTag:
            words.append(word)
        result[' '.join(words)] = posTag
    return result


def extract_features(allTokens):
    result = []

    print "TAGGING DATA"
    posTagMap = _tag_records(allTokens)

    print "EXTRACTING FEATURES FROM TAGGED DATA"
    for i, tokens in enumerate(allTokens):
        posFeatures = _pos_features(posTagMap[' '.join(tokens)])
        result.append(posFeatures)
    return result