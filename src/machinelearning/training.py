from sklearn.svm import libsvm
from sklearn.feature_extraction import DictVectorizer
from numpy import array


__author__ = 'Jasneet Sabharwal'


def train_data(data):
    records, labels = zip(*[(record['features'], record['Sentiment']) for
                            record in data])

    #Need to transform feature vector into something recognized by scikit learn
    #try to keep it a sparse vector so that memory utilization is low. Can we
    #do feature hashing? (what is it and how will it help?)
    print labels
    vec = DictVectorizer()
    vec.fit_transform(records, labels)
    print vec
    #model = libsvm.fit(records, labels)

    #print model
    return None