# -*- coding: utf-8 -*-

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



class W2VModel(Embedding):

    def __init__(self):
        super(W2VModel, self).__init__()
        self.npzModel = None
        self.npzFile = None
        self.vecFile = None
        self.resourcesReady = False
        self.model = None
        self.weights = None


    def setResources(self, options):
        self.npzModel = options["npzModel"]
        self.npzFile = options["npzFile"]
        self.word2idxFile = options["w2idxFile"]
        self.vecFile = options["vecFile"]
        self.resourcesReady = True

    def __processFile(self):
        """
        The .vec file is the input and the npz file without the extension
        :param inputFile:
        :param outputFile:
        :return:
        """
        assert self.resourcesReady
        wikiModel = Word2Vec.load(self.vecFile)
        weightMatrix = wikiModel.wv.syn0
        np.savez(self.npzModel, weightMatrix)
        vocab = dict([(k, v.index) for k, v in wikiModel.wv.vocab.items()])
        with open(self.word2idxFile, 'w') as f:
            f.write(json.dumps(vocab))
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
        filename = weightMatrix.files[0]
        weightMatrix = weightMatrix[filename]
        return (vocab, weightMatrix)


    def prepare(self):
        if not isfile(self.npzFile):
            self.__processFile()
        self.word2idxFile, self.weights = self.__loadModel()


    def getVocabulary(self):
        assert self.resourcesReady
        return self.word2idxFile

    def getWeights(self):
        assert self.resourcesReady
        return self.weights


class EmbeddingLoader(object):

    def __init__(self, model):
        self.model = model

    def process(self):
        self.model.prepare()
        self.word2idx = self.model.getVocabulary()
        self.weights = self.model.getWeights()
        self.idx2word = self.__createInvertedIndex(self.word2idx)

        return self.word2idx, self.idx2word, self.weights

    def __createInvertedIndex(self, word2idx):
        return dict([(v, k) for k, v in word2idx.items()])




if __name__ == '__main__':
    options = {
        "npzFile":"../resources/embeddings/wordEmbeddings.npz",
        "npzModel":"../resources/embeddings/wordEmbeddings",
        "vecFile":"../resources/embeddings/model.vec",
        "w2idxFile":"../resources/embeddings/vocabulary.json"
    }
    model = W2VModel()
    model.setResources(options)
    loader = EmbeddingLoader(model)
    word2idx, idx2word, weights = loader.process()
    print 'index of word \'eu\': {}'.format(word2idx[u'eu'])
    print 'Palavra eu a partir do indice da palavra: {}'.format(idx2word[word2idx[u'eu']])


