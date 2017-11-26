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





