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

from corpus.corpus_converter import CorpusConverter
from corpus.propbankbr_parser import PropBankParser
from embeddings.emb_utils import prepareEmbeddings
from utils.config_loader import readConfig
from utils.data_clean import deleteTrainingResources

seed = 27
np.random.seed(seed)


print 'loading configuration'
config, modelConfig = readConfig()
print 'configuration loaded'

print 'deleting old data'
deleteTrainingResources(20)
print 'data removed'


print 'converting from propbank format : partition 0.95, 0.05. no development set'
parser = PropBankParser(config.corpusDir+'/PropBankBr_v1.1_Const.conll.txt', config.corpusDir+'/verbnet_gold.csv', seed=seed)
parser.prepare()
#parser.generateFeatures(outputDirectory=config.convertedCorpusDir, partition=(0.95, 0, 0.05))
parser.generateKFold(outputDirectory=config.foldsDir, k=20)
print 'conversion ready'

print 'preparing the embedding model - {} - {}'.format(modelConfig.embeddingType, modelConfig.embeddingSize)
sentLoader, predLoader = prepareEmbeddings()
print 'embedding prepared'


print 'creating features'
for i in xrange(1, 21):
    csvFiles = [config.foldsDir+'/train_fold_'+str(i)+'.csv', config.foldsDir+'/test_fold_'+str(i)+'.csv']
    converter = CorpusConverter(csvFiles, sentLoader, predLoader)
    converter.convertAndSave(config.resourceDir+'/feature_file_'+str(i))
    data = converter.load(config.resourceDir+'/feature_file_'+str(i)+'.npy')
print 'features created'