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
_OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data/output.csv'))


def extract_features_data(data):
    """
    Perform feature extraction in batch. Need to do this as Stanford POS tagger works faster in batch mode due to
    model initialization times.
    :param data:
    :return:
    """
    allPreProcessedTokens = [record['preProcessed'] for record in data]
    featureMatrix = featureExtraction.extract_features(allPreProcessedTokens)

    #for (record, features) in zip(data, allFeatures):
    #    record['features'] = features
    return featureMatrix


def pre_process_data(data):
    for record in data:
        record['preProcessed'] = preProcess.pre_process(record['Phrase'])
    return data


def perform_training(featureMatrix, data):
    model = training.train_data(featureMatrix, [record['Sentiment'] for record in data])
    return model


def classify_data(data, model):
    for i, record in enumerate(data):
        record['prediction'] = classification.classify(record['features'], model)
    return data


def main(train, filePath, outFilePath):
    if train:
        data = utils.load_train_data(filePath)
    else:
        data = utils.load_test_data(filePath)

    print "PRE-PROCESSING DATA"
    data = pre_process_data(data)

    print "EXTRACTING FEATURES"
    featureMatrix = extract_features_data(data)

    if train:
        print 'TRAINING MODEL'
        model = perform_training(featureMatrix, data)
        utils.save_model(model, _MODEL_PATH)
    else:
        print "CLASSIFYING"
        model = utils.load_model(_MODEL_PATH)
        data = classify_data(data, model)
        utils.save_classification(data, outFilePath)


if __name__ == '__main__':
    TRAINING = True
    main(TRAINING, _TRAIN_FILE_PATH, _OUTPUT_FILE)