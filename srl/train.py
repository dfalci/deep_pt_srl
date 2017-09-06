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

from model import LSTMModel
from inference import SRLInference
from batcher import Batcher
from embeddings import EmbeddingLoader
from embeddings import W2VModel
from parser import CorpusConverter
import time
import sys


def showProgress(currentStep, totalSteps):
    perc = (float(currentStep)/float(totalSteps)) * 100.0
    temp = perc/10
    sys.stdout.write('\r[{0}] {1}% - {2}/{3}'.format('#'*int(temp), (perc), currentStep, totalSteps))
    sys.stdout.flush()

options = {
    "npzFile":"../resources/embeddings/wordEmbeddings.npy",
    "npzModel":"../resources/embeddings/wordEmbeddings",
    "vecFile":"../resources/embeddings/model.vec",
    "w2idxFile":"../resources/embeddings/vocabulary.json"
}
w2v = W2VModel()
w2v.setResources(options)
loader = EmbeddingLoader(w2v)
word2idx, idx2word, weights = loader.process()
csvFiles = ['../resources/corpus/converted/propbank_training.csv', '../resources/corpus/converted/propbank_test.csv']
converter = CorpusConverter(csvFiles, loader)
data = converter.load('../resources/feature_file.npy')
tagMap = converter.tagMap
tagList = converter.tagList

trainingData = data[0]
testData = data[1]

batcher = Batcher()
batcher.addAll(trainingData[0], trainingData[1], trainingData[2], trainingData[3])
container = batcher.getBatches()

sent, pred, aux, label = batcher.open(container[0])


model = LSTMModel('../config/srl-config.json')
nn = model.create(weights, weights)
nn.summary()

number_of_epochs = 2
for epoch in xrange(number_of_epochs):
    print "--------- Epoch %d -----------" % (epoch+1)
    start_time = time.time()
    numIterations = len(container)
    print 'Running in {} batches'.format((numIterations,))
    for i in xrange(0, numIterations):
        sent, pred, aux, label = batcher.open(container[i])
        showProgress(i, numIterations)
        nn.fit([sent, pred, aux], label)

    showProgress(numIterations, numIterations)
    print '\n'
    print 'end of epoch... evaluating'

    print "%.2f sec for evaluation" % (time.time() - start_time)
    print ""




