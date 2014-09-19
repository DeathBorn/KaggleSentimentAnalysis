from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

__author__ = 'Jasneet Sabharwal'


def load_train_data(file_name):
    data = []
    with open(file_name, 'r') as inFile:
        for line in inFile:
            line = line.strip('\n\r')
            if line:
                linesplit = line.split('\t')
                if linesplit[0].strip() == 'PhraseId':
                    continue
                else:
                    data.append({'PhraseId': linesplit[0],
                                'SentenceId': linesplit[1],
                                'Phrase': linesplit[2],
                                'Sentiment': int(linesplit[3])})
    return data


def load_test_data(file_name):
    data = []
    with open(file_name, 'r') as inFile:
        for line in inFile:
            line = line.strip('\n\r')
            if line:
                linesplit = line.split('\t')
                if linesplit[0].strip() == 'PhraseId':
                    continue
                else:
                    data.append({'PhraseId': linesplit[0],
                                 'SentenceId': linesplit[1],
                                 'Phrase': linesplit[2]})
    return data


def save_model(model, filename):
    joblib.dump(model, filename)


def load_model(filename):
    model = joblib.load(filename)
    return model


def save_classification(data, filename):
    with open(filename, 'w') as outFile:
        outFile.write('PhraseId,Sentiment\n')
        for record in data:
            outFile.write(record['PhraseId'] + ',' + str(record['prediction'])+'\n')


def create_bow_vocabulary(data):
    vec = CountVectorizer(min_df=1, binary=True, dtype='float64', lowercase=True, ngram_range=(1, 1),
                          stop_words=stopwords.words('english'))
    X = vec.fit_transform(data)
    features = vec.get_feature_names()
    return features


def get_bow_vocab(filename):
    bow_vocab = joblib.load(filename)
    return bow_vocab

if __name__ == '__main__':
    data = load_train_data('../../data/train.tsv')
    bow_vocab = create_bow_vocabulary([record['Phrase'] for record in data])
    joblib.dump(bow_vocab, '../../lib/bow_vocab')
