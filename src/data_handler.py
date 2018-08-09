# -*- coding: utf-8 -*-
'''
Created on May 14 2018

@author : Felipe Aparecido Garcia
@github: https://github.com/felipeagarcia/

Open and prepare the data
content: train data
test_content: test data
'''

import numpy as np
import math

# openning data file
with open('../data/X_train.txt', 'r') as data:
    # separating lines
    content = data.readlines()
# removing whitespaces
content = [list(map(float, x.split())) for x in content]

# openning label file
with open('../data/y_train.txt', 'r') as data:
    # separating lines
    labels = data.readlines()
# removing whitespaces
labels = [list(map(int, x.split())) for x in labels]

with open('../data/X_test.txt', 'r') as data:
    # separating lines
    test_content = data.readlines()
# removing whitespaces
test_content = [list(map(float, x.split())) for x in test_content]

# openning label file
with open('../data/y_test.txt', 'r') as data:
    # separating lines
    test_labels = data.readlines()
# removing whitespaces
test_labels = [list(map(int, x.split())) for x in test_labels]


def prepare_data(data, labels, length):
    '''
    This function makes every line of the data to be of the same length
    @param data: raw_data to be prepared
    @param labels: labels of the data
    @param length: desired time step
    '''
    prepared_data = []
    prepared_data.append([])
    current_label = labels[0][0]
    pos = 0
    prepared_labels = []
    aux = np.zeros(6)
    aux[current_label - 1] = 1
    prepared_labels.append(list(aux))
    for i in range(len(data)):
        if(current_label == labels[i][0]):
            prepared_data[pos].append(data[i])
        else:
            current_label = labels[i][0]
            # fill the first positions of the data with zeros
            while(len(prepared_data[pos]) < length):
                prepared_data[pos].insert(0, list(np.zeros(561)))
            prepared_data.append([])
            aux = np.zeros(6)
            aux[current_label - 1] = 1
            prepared_labels.append(list(aux))
            pos += 1
    while(len(prepared_data[pos]) < length):
        prepared_data[pos].insert(0, list(np.zeros(561)))
    aux = list(zip(prepared_data, prepared_labels))
    np.random.shuffle(aux)
    prepared_data[:], prepared_labels[:] = zip(*aux)
    return prepared_data, prepared_labels
