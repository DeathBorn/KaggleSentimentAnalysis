from src.features import featureExtraction
from src.preprocessing import preProcess
from src.machinelearning import training, classification
from src.utils import utils
import os
import sys

__author__ = 'Jasneet Sabharwal'

_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model/model.svm'))
_TRAIN_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data/train.tsv'))
_TEST_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data/test.tsv'))


def extract_features_data(data):
    """
    Perform feature extraction in batch. Need to do this as Stanford POS tagger works faster in batch mode due to
    model initialization times.
    :param data:
    :return:
    """
    allPreProcessedTokens = [record['preProcessed'] for record in data]
    allFeatures = featureExtraction.extract_features(allPreProcessedTokens)

    for i, (record, features) in enumerate(zip(data, allFeatures)):
        record['features'] = features
        data[i] = record
    return data


def pre_process_data(data):
    for i, record in enumerate(data):
        record['preProcessed'] = preProcess.pre_process(record['Phrase'])
        data[i] = record
    return data


def perform_training(data):
    model = training.train_data(data)
    return model


def classify_data(data):
    for i, record in enumerate(data):
        record['prediction'] = classification.classify(record['features'])
        data[i] = record
    return data


def main(train):

    data = utils.load_train_data(_TRAIN_FILE_PATH)

    print "PRE-PROCESSING DATA"
    data = pre_process_data(data)

    print "EXTRACTING FEATURES"
    data = extract_features_data(data)

    if train:
        print 'TRAINING MODEL'
        model = perform_training(data)
        utils.save_model(model, _MODEL_PATH)
    else:
        model = utils.load_model(_MODEL_PATH)
        #data = classify_data(data)
        #utils.save_classification(data, '')
        pass


if __name__ == '__main__':
    TRAINING = True
    main(TRAINING)