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


from emb_loader import EmbeddingLoader, W2VModel
from nn_corpus_loader import CorpusConverter
from propbankbr_parser import PropBankParser

import numpy as np

from config import Config
from model_config import ModelConfig
from function_utils import Utils
from prepare_hybrid_embeddings import prepareEmbeddings


np.random.seed(4)


print 'loading configuration'
config = Config.Instance()
config.prepare(Utils.getWorkingDirectory())
print 'base directory : {}'.format(config.baseDir)
ModelConfig.Instance().prepare(config.srlConfig+'/srl-config.json')
print 'configuration loaded'


print 'converting from propbank format : partition 0.95, 0.05. no development set'
parser = PropBankParser(config.corpusDir+'/PropBankBr_v1.1_Const.conll.txt', config.corpusDir+'/verbnet_gold.csv')
parser.prepare()
parser.generateFeatures(outputDirectory=config.convertedCorpusDir, partition=(0.95, 0, 0.05))
print 'conversion ready'

print 'loading embedding model'
sentLoader, predLoader = prepareEmbeddings(config, 'hybrid')
print 'embeddings loaded'


print 'creating features'
csvFiles = [config.convertedCorpusDir+'/propbank_training.csv', config.convertedCorpusDir+'/propbank_test.csv']
converter = CorpusConverter(csvFiles, sentLoader, predLoader)
converter.convertAndSave(config.resourceDir+'/feature_file')
data = converter.load(config.resourceDir+'/feature_file.npy')
print 'features created'