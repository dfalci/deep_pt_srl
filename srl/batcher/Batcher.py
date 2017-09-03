# -*- coding: utf-8 -*-

import numpy as np

class Batcher(object):

    def __init__(self):
        """
        Creates mini-batches grouped according to the sentence size
        :return:
        """
        self.nBatches = 0
        self.batches = []
        self.struct = {}
        self.currentBatch = 0

    def __getTag(self, sent):
        size = sent.shape[1]
        try:
            return self.struct[size]
        except:
            self.nBatches += 1
            self.struct[size] = []
        return self.struct[size]

    def add(self, sent, pred, aux, label):
        #print sent.shape, label.shape
        self.__getTag(sent).append((sent, pred, aux, label,))

    def printStats(self):
        print 'Quantidade de batches na parada ', self.nBatches

    def __transformBatch(self, key, arr):
        sent = []
        pred = []
        aux = []
        label = []
        for i in xrange(0, len(arr)):
            sent.extend(arr[i][0])
            pred.extend(arr[i][1])
            aux.extend(arr[i][2])
            label.extend(arr[i][3])
        return np.array(sent), np.array(pred), np.array(aux), np.array(label)

    def getBatches(self):
        container = []
        for key in self.struct:
            sent, pred, aux, label = self.__transformBatch(key, self.struct[key])
            container.append((sent, pred, aux, label,))
        return container

        #print sent.shape, pred.shape, aux.shape, label.shape
    def open(self, containerLine):
        return containerLine[0], containerLine[1], containerLine[2], containerLine[3]


""" def convertToTrainFormat(sent, pred, aux, label, debug = False):
    sent = np.array(sent)[np.newaxis, :]
    pred = np.array(pred)[np.newaxis, :]
    aux = np.array(aux)[np.newaxis, :]
    label = np.array(label)[np.newaxis, :]
    if debug:
        print sent.shape, pred.shape, aux.shape, label.shape
    return sent, pred, aux, label

numIterations = len(x_train)
batcher = Batcher()
for i in xrange(0, numIterations):
    sent, pred, aux, label = convertToTrainFormat(x_train[i], x_train_predicates[i], xAuxTrain[i], y_train[i])
    batcher.add(sent, pred, aux, label)
container = batcher.getBatches()

"""

