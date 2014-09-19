from sklearn.externals import joblib

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


if __name__ == '__main__':
    print len(load_train_data('../../data/train.tsv'))