from nltk.tag.stanford import POSTagger
from nltk.corpus import stopwords
from src.utils.sentiwordnet import SentiWordNetCorpusReader, SentiSynset
from src.utils import utils
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
import os

__author__ = 'Jasneet Sabharwal'

_POS_TAGGER_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'lib/english-bidirectional-distsim.tagger')
_POS_TAGGER_JAR_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'lib/stanford-postagger.jar')
_SENTI_WORDNET_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'lib/SentiWordNet_3.0.0_20130122.txt')
_BOW_VOCAB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'lib/bow_vocab')

POS_TAGGER = POSTagger(_POS_TAGGER_MODEL_PATH, _POS_TAGGER_JAR_PATH)
SENTI_WORDNET = SentiWordNetCorpusReader(_SENTI_WORDNET_FILE_PATH)
BOW_VECTORIZER = CountVectorizer(min_df=1, binary=True, dtype='float64', lowercase=True, ngram_range=(1, 1),
                                 stop_words=stopwords.words('english'), vocabulary=utils.get_bow_vocab(_BOW_VOCAB_PATH))


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


def _bow_features(records):
    bow_features = BOW_VECTORIZER.transform(records)
    return bow_features


def _vectorize_and_merge(posFeatures, bowFeatureMatrix):
    vec = DictVectorizer()
    posFeatureMatrix = vec.fit_transform(posFeatures).toarray()

    featureMatrix = hstack([posFeatureMatrix, bowFeatureMatrix])

    return featureMatrix


def extract_features(allTokens):
    allPosFeatures = []

    print "TAGGING DATA"
    posTagMap = _tag_records(allTokens)

    print "EXTRACTING FEATURES FROM TAGGED DATA"
    for i, tokens in enumerate(allTokens):
        posFeatures = _pos_features(posTagMap[' '.join(tokens)])
        allPosFeatures.append(posFeatures)

    print "EXTRACTING BOW FEATURES"
    bowRecords = [' '.join(token) for token in allTokens]
    bowFeatureMatrix = _bow_features(bowRecords)

    featureMatrix = _vectorize_and_merge(allPosFeatures, bowFeatureMatrix)
    return featureMatrix