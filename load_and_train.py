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
#set the seed for reproductability
np.random.seed(4)

from corpus.corpus_converter import CorpusConverter
from embeddings.emb_utils import getEmbeddings
from model.auxiliar.lr_reducer import PatienceBaseLrReducer, CyclicLearningRate, FixedBasedLrReducer
from model.auxiliar.early_stopper import EarlyStopper
from model.batcher import Batcher
from model.configuration.model_config import ModelConfig
from model.configuration import Config
from model.evaluation.training_evaluation import Evaluator
from model.inference import SRLInference
from model.lstm_model import LSTMModel
from model.persistence.model_persistence import ModelEvaluation
from utils.nn_utils import NNUtils
from utils.config_loader import readConfig


def showProgress(currentStep, totalSteps, epoch):
    perc = (float(currentStep)/float(totalSteps)) * 100.0
    temp = perc/10
    sys.stdout.write('\r[{0}] {1}% - {2}/{3} - Epoch {4}'.format('#'*int(temp), (perc), currentStep, totalSteps, epoch))
    sys.stdout.flush()


print 'loading configuration'
config, modelConfig = readConfig()
print 'configuration loaded'



print 'loading word embeddings : {} - embedding size : {}'.format(modelConfig.embeddingType, modelConfig.embeddingSize)
sentenceLoader, predicateLoader = getEmbeddings()

print 'sentenceLoader shape {}'.format(sentenceLoader.weights.shape)

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
#lrReducer = PatienceBaseLrReducer(modelConfig.trainingEpochs)
lrReducer = FixedBasedLrReducer(modelConfig.trainingEpochs)
clr = CyclicLearningRate(base_lr=0.00020, max_lr=0.0012, step_size=(204.*3), mode='exp_range', gamma=0.99996)
msaver = ModelEvaluation()
print 'prepared'

print 'creating neural network model'
file = str(sys.argv[1])
startingEpoch = int(sys.argv[2])
lrDefault = float(sys.argv[3])
print 'model to load : {} - epoch : {} - lr : {}'.format(file, startingEpoch, lrDefault)
model = LSTMModel(ModelConfig.Instance())
nn = model.load(Config.Instance().resultsDir+'/'+file+'.json', Config.Instance().resultsDir+'/'+file+'.h5py')
nn.summary()
lrReducer.setNetwork(nn)
es = EarlyStopper()
lrReducer.setLearningRate(lrDefault)
print 'model loaded'


print 'start training'

number_of_epochs = ModelConfig.Instance().trainingEpochs
for epoch in xrange(startingEpoch, number_of_epochs):
    print "--------- Epoch %d -----------" % (epoch)
    start_time = time.time()
    numIterations = len(container)

    print 'shuffling training data'
    indexes = np.arange(len(container))
    np.random.shuffle(indexes)
    print indexes

    print 'Running in {} batches'.format(numIterations)
    for i in xrange(0, numIterations):
        z = indexes[i]

        sent, pred, aux, label = batcher.open(container[z])
        showProgress(i, numIterations, epoch)
        nn.fit([sent, pred, aux], label, shuffle=False, verbose=0)

    showProgress(numIterations, numIterations, epoch)
    print '\n end of epoch in  {}... evaluating'.format((time.time() - start_time))
    start_time = time.time()

    print '\n number of iterations : {}'.format(clr.clr_iterations)

    evaluator.prepare(nn, config.resultsDir+'/epoch_'+str(epoch), config.resourceDir+'/srl-eval.pl')
    evaluation = evaluator.evaluate()

    tokenf1 = evaluation["tokenMacroF1"]
    officialf1 = evaluation["officialF1"]


    print 'TOKEN F1-SCORE : {}'.format(tokenf1)
    print 'OFFICIAL F1-SCORE : {}'.format(officialf1)

    lrReducer.onEpochEnd(officialf1, epoch)


    print "%.2f sec for evaluation" % (time.time() - start_time)

    print "saving checkpoint if needed"
    msaver.update(nn, officialf1, epoch)

    if es.shouldStop(officialf1):
        print 'Early stopper decided to quit on epoch {} - best value {}'.format(epoch, es.best)
        break


print 'ended training'



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


from corpus.corpus_converter import CorpusConverter
from embeddings.emb_utils import getEmbeddings
from model.configuration import Config
from model.configuration.model_config import ModelConfig
from model.inference import SRLInference
from model.inference import Predictor
from model import LSTMModel
from utils.function_utils import Utils
from utils.nn_utils import NNUtils
from utils import extractFeaturesFromSentence, toNNFormat
import pandas as pd
import unicodecsv as csv
import sys
from utils.config_loader import readConfig


"""
This script is responsible for loading the data from wiki.csv file, annotating it with semantic roles inferred with a given trained model.
"""

def showProgress(currentStep, totalSteps, length):
    sys.stdout.write('\r{0} of {1} - {2}'.format(currentStep, totalSteps, length))
    sys.stdout.flush()

def checkVerb(roles, distance):
    i = 0
    for d in distance:
        if d == 0:
            return roles[i] == u'V'
        i +=1

def formatItems(items):
    ret = ''
    for i in items:
        ret += str(i)+ ' '
    return ret.encode('utf-8').strip()

print 'loading configuration'
config, modelConfig = readConfig()
print 'configuration loaded'



print 'loading word embeddings {}'.format(modelConfig.embeddingType)
sentenceLoader, predicateLoader = getEmbeddings(config, modelConfig.embeddingType)
nnUtils = NNUtils.Instance()
nnUtils.setWordUtils(sentenceLoader.word2idx, sentenceLoader.idx2word)
print 'loaded'


print 'loading corpus'
csvFiles = [config.convertedCorpusDir+'/propbank_training.csv', config.convertedCorpusDir+'/propbank_test.csv']
converter = CorpusConverter(csvFiles, sentenceLoader, predicateLoader)
data = converter.load(config.resourceDir+'/wiki_feature_file.npy')
tagMap = converter.tagMap
tagList = converter.tagList
nnUtils.setTagList(tagMap, tagList)
print 'loaded'

print 'preparing data for training'
trainingData = data[0]
testData = data[1]


print 'loading neural network model'
inference = SRLInference(tagMap, tagList)
model = LSTMModel(ModelConfig.Instance())
nn = model.load(Config.Instance().resultsDir+'/best/wiki_model.json', Config.Instance().resultsDir+'/best/wiki_model.h5py')
nn.summary()
print 'model loaded'

prediction = Predictor(nn, tagList, inference)


wikiFile = pd.read_csv(config.convertedCorpusDir+'/wiki.csv')

results = []
iterations = len(wikiFile)
for i in xrange(0, iterations):
    try:
        propositionId = wikiFile['propositionId'][i]
        predicate = wikiFile['predicate'][i]
        sentence = wikiFile['sentence'][i]
        convertedSentence, convertedPredicate, allCaps, firstCaps, noCaps, context, distance = extractFeaturesFromSentence(sentence, predicate, sentenceLoader.word2idx, predicateLoader.word2idx)
        inputSentence, inputPredicate, inputAux = toNNFormat(convertedSentence, convertedPredicate, allCaps, firstCaps, noCaps, context, distance)

        roles = prediction.predict(inputSentence, inputPredicate, inputAux)

        if checkVerb(roles, distance):
            results.append((propositionId,predicate,sentence,formatItems(roles),formatItems(allCaps),formatItems(firstCaps),formatItems(noCaps),formatItems(distance),formatItems(context)))
        showProgress(i, iterations, len(results))
    except:
        print 'error in line {}'.format(i)



with open(config.convertedCorpusDir+'/semi_sup_wiki.csv', 'w') as f:
    writer = csv.writer(f, encoding='utf-8')
    writer.writerow(['propositionId', 'predicate', 'sentence', 'roles', 'allCapitalized', 'firstCapitalized', 'noCapitalized', 'distance', 'predicateContext'])
    for p in results:
        writer.writerow(p)
f.close()





