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

import sys
import time

import numpy as np

from corpus.corpus_converter import CorpusConverter
from embeddings.emb_utils import getEmbeddings
from model.auxiliar.lr_reducer import PatienceBaseLrReducer
from model.batcher import Batcher
from model.configuration import Config
from model.configuration.model_config import ModelConfig
from model.evaluation.training_evaluation import Evaluator
from model.inference import SRLInference
from model.lstm_model import LSTMModel
from model.persistence.model_persistence import ModelEvaluation
from utils.function_utils import Utils
from utils.nn_utils import NNUtils


def showProgress(currentStep, totalSteps):
    perc = (float(currentStep)/float(totalSteps)) * 100.0
    temp = perc/10
    sys.stdout.write('\r[{0}] {1}% - {2}/{3}'.format('#'*int(temp), (perc), currentStep, totalSteps))
    sys.stdout.flush()

np.random.seed(13)

print 'loading configuration'
config = Config.Instance()
config.prepare(Utils.getWorkingDirectory())

modelConfig = ModelConfig.Instance()
modelConfig.prepare(config.srlConfig+'/srl-config.json')
print 'configuration loaded'



print 'loading word embeddings {}'.format(modelConfig.embeddingType)
sentenceLoader, predicateLoader = getEmbeddings(config, modelConfig.embeddingType)
nnUtils = NNUtils.Instance()
nnUtils.setWordUtils(sentenceLoader.word2idx, sentenceLoader.idx2word)
print 'loaded'


print 'loading corpus'
csvFiles = [config.convertedCorpusDir+'/propbank_training.csv', config.convertedCorpusDir+'/propbank_test.csv']
converter = CorpusConverter(csvFiles, sentenceLoader, predicateLoader)
data = converter.load(config.resourceDir+'/feature_file.npy')
tagMap = converter.tagMap
tagList = converter.tagList
nnUtils.setTagList(tagMap, tagList)
print 'loaded'

print 'preparing data for training'
trainingData = data[0]
testData = data[1]

batcher = Batcher()
batcher.addAll(trainingData[0], trainingData[1], trainingData[2], trainingData[3])
container = batcher.getBatches()

inference = SRLInference(tagMap, tagList)
evaluator = Evaluator(testData, inference, nnUtils, config.resultsDir+'/finalResult.json')
lrReducer = PatienceBaseLrReducer(modelConfig.trainingEpochs)
msaver = ModelEvaluation(modelConfig.checkpointsToKeep)
print 'prepared'

print 'creating neural network model'
model = LSTMModel(ModelConfig.Instance())
nn = model.create(sentenceLoader.weights, predicateLoader.weights)
nn.summary()
lrReducer.setNetwork(nn)
print 'model loaded'


print 'start training'

number_of_epochs = ModelConfig.Instance().trainingEpochs
for epoch in xrange(number_of_epochs):
    print "--------- Epoch %d -----------" % (epoch+1)
    start_time = time.time()
    numIterations = len(container)

    indexes = np.arange(len(container))
    np.random.shuffle(indexes)

    print indexes

    print 'Running in {} batches'.format(numIterations)
    for i in xrange(0, numIterations):
        z = indexes[i]

        sent, pred, aux, label = batcher.open(container[z])
        showProgress(i, numIterations)
        nn.fit([sent, pred, aux], label)

    showProgress(numIterations, numIterations)
    print '\n'
    print 'end of epoch in  {}... evaluating'.format((time.time() - start_time))
    start_time = time.time()
    evaluator.prepare(nn, config.resultsDir+'/epoch_'+str(epoch+1), config.resourceDir+'/model-eval.pl')
    evaluation = evaluator.evaluate()
    f1 = evaluation["macroF1"]
    print 'F1-SCORE : {}'.format(f1)
    lrReducer.onEpochEnd(f1, epoch+1)
    print "%.2f sec for evaluation" % (time.time() - start_time)

    print "saving checkpoint if needed"
    msaver.update(nn, f1, epoch+1)


print 'ended training'
