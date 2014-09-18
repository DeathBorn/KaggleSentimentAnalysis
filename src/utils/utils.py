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
                                'Sentiment': linesplit[3]})
    return data


def load_test_data(filename):
    pass


def save_model(model, filename):
    pass


def save_classification(data, filename):
    pass


if __name__ == '__main__':
    print len(load_train_data('../../data/train.tsv'))