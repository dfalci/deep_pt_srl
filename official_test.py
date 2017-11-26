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
from model.auxiliar.lr_reducer import RateBasedLrReducer
from model.configuration import Config
from model.configuration.model_config import ModelConfig
from model.evaluation.training_evaluation import Evaluator
from model.inference import SRLInference
from model import LSTMModel
from model.persistence.model_persistence import ModelEvaluation
from utils.function_utils import Utils
from utils.nn_utils import NNUtils



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
testData = data[1]

inference = SRLInference(tagMap, tagList)

evaluator = Evaluator(testData, None, nnUtils, config.resultsDir+'/finalResult.json')

lrReducer = RateBasedLrReducer(modelConfig.trainingEpochs)
msaver = ModelEvaluation()
print 'prepared'

print 'loading neural network model'
model = LSTMModel(ModelConfig.Instance())
nn = model.load(Config.Instance().resultsDir+'/best/wiki_model.json', Config.Instance().resultsDir+'/best/wiki_model.h5py')
nn.summary()
print 'model loaded'

print 'evaluating...'
evaluator.prepare(nn, config.resultsDir+'/official_no', config.resourceDir+'/srl-eval.pl')
evaluation = evaluator.evaluate()
print 'done'



