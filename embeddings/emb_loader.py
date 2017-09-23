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
from gensim.models import Word2Vec
from os.path import isfile
import json

class Embedding(object):

    def __init__(self):
        pass

    def setResources(self, options):
        pass

    def __loadModel(self):
        pass

    def __processFile(self):
        pass

    def prepare(self):
        pass

    def getWeights(self):
        pass

    def getVocabulary(self):
        pass

class HybridModel(Embedding):

    def __init__(self):
        super(HybridModel, self).__init__()
        self.npzModel = None
        self.npzFile = None
        self.resourcesReady = False
        self.model = None
        self.weights = None


    def setResources(self, options):
        self.npzModel = options["npzModel"]
        self.npzFile = options["npzFile"]
        self.word2idxFile = options["w2idxFile"]
        self.resourcesReady = True


    def getVocabulary(self):
        assert self.resourcesReady
        return self.word2idxFile

    def getWeights(self):
        assert self.resourcesReady
        return self.weights

    def __processFile(self):
        with open(self.word2idxFile, 'w') as f:
            f.write(json.dumps(self.word2idx))
        f.close()

        np.save(self.npzModel, self.weights)

    def __loadModel(self):
        """
        In this model the files are
        :param files:
        :return:
        """
        assert self.resourcesReady
        with open(self.word2idxFile, 'r') as f:
            vocab = json.loads(f.read())
        weightMatrix = np.load(self.npzFile)
        return (vocab, weightMatrix)

    def prepare(self):
        if not isfile(self.npzFile):
            self.__processFile()
        self.word2idxFile, self.weights = self.__loadModel()

    def generateCorpus(self, tokens, originalWeights, originalWord2idx):
        unkWords = []
        unkCount = 0
        self.word2idx={}
        if originalWord2idx != None:
            dimensions = originalWeights.shape[1]
            self.weights = np.zeros((len(tokens), dimensions), dtype=float)

        currIdx = 0
        for t in tokens:
            self.word2idx[t] = currIdx
            try:
                originalIdx = originalWord2idx[t]
                #self.weights[currIdx] = np.vstack((self.weights, originalWeights[originalIdx]))
                self.weights[currIdx] = originalWeights[originalIdx]
            except:
                #self.weights = np.vstack((self.weights, np.random.rand(1, dimensions)))
                self.weights[currIdx] = np.random.rand(1, dimensions)
                unkCount+=1
                unkWords.append(t)
            currIdx+=1

        print 'palavras desconhecidas - {}'.format(unkCount)
        print unkWords





class W2VModel(Embedding):

    def __init__(self):
        super(W2VModel, self).__init__()
        self.npzModel = None
        self.npzFile = None
        self.vecFile = None
        self.resourcesReady = False
        self.model = None
        self.weights = None
        self.word2idx = None


    def setResources(self, options):
        self.npzModel = options["npzModel"]
        self.npzFile = options["npzFile"]
        self.word2idxFile = options["w2idxFile"]
        self.vecFile = options["vecFile"]
        self.resourcesReady = True

    def __processFile(self, addUNK=True, dimensions=150):
        """
        The .vec file is the input and the npz file without the extension
        :param inputFile:
        :param outputFile:
        :return:
        """
        assert self.resourcesReady
        wikiModel = Word2Vec.load(self.vecFile)
        weightMatrix = wikiModel.wv.syn0

        vocab = dict([(k, v.index) for k, v in wikiModel.wv.vocab.items()])
        print len(vocab)
        print weightMatrix.shape
        if addUNK:
            vocab['UNK_TK'] = len(vocab)
            weightMatrix = np.vstack((weightMatrix, np.random.rand(1, dimensions)))
            vocab['UNK_PRED'] = len(vocab)
            weightMatrix = np.vstack((weightMatrix, np.random.rand(1, dimensions)))
        print len(vocab)
        print weightMatrix.shape

        with open(self.word2idxFile, 'w') as f:
            f.write(json.dumps(vocab))


        np.save(self.npzModel, weightMatrix)

        f.close()


    def __loadModel(self):
        """
        In this model the files are
        :param files:
        :return:
        """
        assert self.resourcesReady
        with open(self.word2idxFile, 'r') as f:
            vocab = json.loads(f.read())
        weightMatrix = np.load(self.npzFile)
        #filename = weightMatrix.files[0]
        #weightMatrix = weightMatrix[filename]
        return (vocab, weightMatrix)


    def prepare(self):
        if not isfile(self.npzFile):
            self.__processFile()
        self.word2idx, self.weights = self.__loadModel()


    def getVocabulary(self):
        assert self.resourcesReady
        return self.word2idx

    def getWeights(self):
        assert self.resourcesReady
        return self.weights


class EmbeddingLoader(object):

    def __init__(self, model):
        self.model = model

    def process(self, addPadding=True, dimensions=150):
        """
        Loads the embedding model. If necessary adds a padding member (array with zeros) at the end
        :param addPadding:
        :param dimensions:
        :return:
        """
        self.model.prepare()
        self.word2idx = self.model.getVocabulary()
        self.weights = self.model.getWeights()
        if addPadding:
            self.weights = np.vstack((self.weights, np.zeros((1, dimensions))))

        self.idx2word = self.__createInvertedIndex(self.word2idx)

        return self.word2idx, self.idx2word, self.weights

    def __createInvertedIndex(self, word2idx):
        return dict([(v, k) for k, v in word2idx.items()])


    def getValues(self):
        return self.word2idx, self.idx2word, self.weights



if __name__ == '__main__':
    options = {
        "npzFile":"../resources/embeddings/wordEmbeddings.npy",
        "npzModel":"../resources/embeddings/wordEmbeddings",
        "vecFile":"../resources/embeddings/model.vec",
        "w2idxFile":"../resources/embeddings/vocabulary.json"
    }
    model = W2VModel()
    model.setResources(options)
    loader = EmbeddingLoader(model)
    word2idx, idx2word, weights = loader.process()
    print 'index of word \'eu\': {}'.format(word2idx[u'eu'])
    print 'word test : {}'.format(idx2word[word2idx[u'eu']])


