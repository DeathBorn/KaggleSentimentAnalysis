from sklearn.svm import libsvm
from sklearn.feature_extraction import DictVectorizer

__author__ = 'Jasneet Sabharwal'


def classify(features, model):
    vec = DictVectorizer()
    X = vec.fit_transform(features)
    prediction = libsvm.predict(X.toarray(), *model)
    return int(prediction[0])
