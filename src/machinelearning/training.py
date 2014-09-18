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

    vec = DictVectorizer()
    X = vec.fit_transform(records)
    model = libsvm.fit(X.toarray(), array(labels, dtype='float64'))

    return model