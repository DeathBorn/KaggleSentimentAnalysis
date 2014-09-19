from sklearn.svm import libsvm
from sklearn.feature_extraction import DictVectorizer
from numpy import array


__author__ = 'Jasneet Sabharwal'


def train_data(featureMatrix, labels):

    #Need to transform feature vector into something recognized by scikit learn
    #try to keep it a sparse vector so that memory utilization is low. Can we
    #do feature hashing? (what is it and how will it help?)


    model = libsvm.fit(featureMatrix.toarray(), array(labels, dtype='float64'))

    return model