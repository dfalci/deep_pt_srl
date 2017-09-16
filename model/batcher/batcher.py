# -*- coding: utf-8; -*-
# Copyright (c) 2017, Daniel Falci - danielfalci@gmail.com
# Laboratory for Advanced Information Systems - LAIS
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither the name of deep_pt_srl nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

    def addAll(self, sentences, predicates, auxiliar, roles):
        for (sent, pred, aux, label) in zip(sentences, predicates, auxiliar, roles):
            self.add(sent, pred, aux, label)

