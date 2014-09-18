from src.features import featureExtraction
from src.preprocessing import preProcess
from src.machinelearning import training, classification
from src.utils import utils
import sys

__author__ = 'Jasneet Sabharwal'


def extract_features_data(data):
    for i, record in enumerate(data):
        record['features'] = featureExtraction.extract_features(record['preProcessed'])
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
    data = utils.load_train_data(
        '../../data/train.tsv')[:2]

    print "PRE-PROCESSING DATA"
    data = pre_process_data(data)

    print "EXTRACTING FEATURES"
    data = extract_features_data(data)

    if train:
        model = perform_training(data)
        utils.save_model(model, '')
    else:
        data = classify_data(data)
        utils.save_classification(data, '')


if __name__ == '__main__':
    TRAINING = True
    main(TRAINING)