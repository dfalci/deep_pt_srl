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
import pandas as pd

from embeddings import EmbeddingLoader
from utils import parseCSVLine


class CorpusConverter(object):

    def __init__(self, csvFiles, sentenceEmbeddingLoader, predicateEmbeddingLoader=None):
        """
        This class is responsible for transforming the data from the csv file to the format expected by our lstm model
        :param csvFile: the csv file to be transformed
        :param embeddingLoader: The initialized embedding loader
        :return:
        """
        self.csvFiles = csvFiles

        self.sentenceEmbeddingLoader = sentenceEmbeddingLoader
        if predicateEmbeddingLoader == None:
            self.predicateEmbeddingLoader = sentenceEmbeddingLoader
        else:
            self.predicateEmbeddingLoader = predicateEmbeddingLoader

        self.wordStat = {'biggestSize':0, 'unknown':0, 'unktokens':[], 'total':0}
        self.predStat = {'unknown':0, 'unktokens':[], 'total':0}
        self.tagCount = {}
        self.tagMap = {}
        self.tagList = []


    def __loadFile(self, f):
        return pd.read_csv(f)


    def __preparePredicate(self, x, word2idx, defaultPredicate=u'UNK_PRED'):
        token = parseCSVLine(x)
        try:
            return word2idx[token]
        except:
            self.predStat['unknown'] += 1
            self.predStat['unktokens'].append(token)
            return word2idx[defaultPredicate]
        finally:
            self.predStat['total'] += 1


    def __prepareTokens(self, x, word2idx, defaultWord=u'UNK_TK', defaultPredicate=u'UNK_PRED'):
        temp = parseCSVLine(x)
        tokens = temp.split(' ')
        retorno = []
        for t in tokens:
            try:
                retorno.append(word2idx[t])
            except:

                if t in self.predStat['unktokens']:
                    retorno.append(word2idx[defaultPredicate])
                else:
                    retorno.append(word2idx[defaultWord])

                self.wordStat['unknown']+=1
                self.wordStat['unktokens'].append(t)

        val = len(retorno)
        if val > self.wordStat['biggestSize']:
            self.wordStat['biggestSize'] = val
        self.wordStat['total']+=val
        return retorno

    def __prepareRoles(self, roles):
        tags = roles.split(' ')
        retorno = []
        for t in tags:
            try:
                self.tagCount[t] = self.tagCount[t]+1
            except:
                self.tagCount[t] = 1
                self.tagList.append(t)
                self.tagMap[t] = len(self.tagList)-1
            retorno.append(self.tagMap[t])
        return retorno

    def __getIntegerFeatures(self, x):
        x = x.strip().split(' ')
        for i in xrange(0, len(x)):
            x[i] = int(x[i])
        return x

    def __prepareAuxiliaryFeatures(self, row):
        allCaps = self.__getIntegerFeatures(row['allCapitalized'])
        firstCaps = self.__getIntegerFeatures(row['firstCapitalized'])
        noCaps = self.__getIntegerFeatures(row['noCapitalized'])
        distance = self.__getIntegerFeatures(row['distance'])
        predicateContext = self.__getIntegerFeatures(row['predicateContext'])
        ret = []
        for i in xrange(0, len(allCaps)):
            ret.append([allCaps[i], firstCaps[i], noCaps[i], distance[i], predicateContext[i]])
        return ret

    def __auxiliaryToNP(self, ret):
        final = []
        for i in xrange(0, len(ret)):
            linha = []
            for j in xrange(0, len(ret[i])):
                linha.append(np.array(ret[i][j]))
            final.append(np.array(linha))
        return np.array(final)

    def __expandPredicates(self, predicates, sentences):
        """
        Responsible for expanding predicates when necessary for as many timesteps as necessary
        :param predicates:
        :param sentences:
        :return:
        """
        ret = []
        for i in xrange(0, len(predicates)):
            line = []
            for este in xrange(0, len(sentences[i])):
                line.append(predicates[i])
            ret.append(line)
        return np.array(ret)

    def __expandRoles(self, roles):
        """
        Expand the role into binarized feature : class 1 from 10 possible classes will be converted to an array such as - [0, 1, 0, 0, 0 ... 0]
        :param roles:
        :return:
        """
        nClasses = len(self.tagList)
        ret = []
        for row in range(0, len(roles)):
            nrow = []
            for column in range(0, len(roles[row])):
                nrow.append(np.eye(nClasses)[roles[row][column]])
            ret.append(nrow)
        return np.array(ret)

    def __toTrainFormat(self, sentences, predicates, auxiliar, roles):
        retSentences = []
        retPredicates = []
        retAuxiliar = []
        retRoles = []
        for (sent, pred, aux, label) in zip(sentences, predicates, auxiliar, roles):
            retSentences.append(np.array(sent)[np.newaxis, :])
            retPredicates.append(np.array(pred)[np.newaxis, :])
            retAuxiliar.append(np.array(aux)[np.newaxis, :])
            retRoles.append(np.array(label)[np.newaxis, :])


        return retSentences, retPredicates, retAuxiliar, retRoles


    def printStats(self):
        print self.wordStat
        print self.predStat


    def __validate(self, sentence, roles, predicates, aux):
        assert sentence.shape == roles.shape == predicates.shape == aux.shape
        for (s, r, a) in zip(sentence, roles, aux):
            assert len(s) == len(r) == a.shape[0] and a.shape[1] == 5


    def save(self, structure, featureFile):
        data = {}
        data["tagMap"] = self.tagMap
        data["tagList"] = self.tagList
        data["data"] = structure
        np.save(featureFile, data)

    def load(self, featureFile):
        data = np.load(featureFile)
        self.tagMap = data[()]["tagMap"]
        self.tagList = data[()]["tagList"]
        return data[()]["data"]

    def convert(self):
        structure = []
        for f in self.csvFiles:
            originalData = self.__loadFile(f)

            predicates = np.array(originalData['predicate'].apply(lambda x: self.__preparePredicate(x, self.predicateEmbeddingLoader.word2idx)))

            #handle all the final transformations needed
            sentences = np.array(originalData['sentence'].apply(lambda x: self.__prepareTokens(x, self.sentenceEmbeddingLoader.word2idx)))
            roles = self.__expandRoles(np.array(originalData['roles'].apply(lambda x: self.__prepareRoles(x))))
            predicates = self.__expandPredicates(predicates, sentences)
            aux = self.__auxiliaryToNP(originalData.apply(lambda x: self.__prepareAuxiliaryFeatures(x), axis=1))

            self.__validate(sentences, roles, predicates, aux)

            sentences, predicates, aux, roles = self.__toTrainFormat(sentences, predicates, aux, roles)
            # the order is important
            structure.append((sentences, predicates, aux, roles))
        return structure

    def convertAndSave(self, featureFile):
        self.save(self.convert(), featureFile)




if __name__ == '__main__':

    from embeddings.emb_loader import W2VModel

    options = {
        "npzFile":"../../resources/embeddings/wordEmbeddings.npy",
        "npzModel":"../../resources/embeddings/wordEmbeddings",
        "vecFile":"../../resources/embeddings/model.vec",
        "w2idxFile":"../../resources/embeddings/vocabulary.json"
    }
    model = W2VModel()
    model.setResources(options)
    loader = EmbeddingLoader(model)
    loader.process()
    csvFiles = ['../../resources/corpus/converted/propbank_training.csv', '../../resources/corpus/converted/propbank_test.csv']
    converter = CorpusConverter(csvFiles, loader)
    converter.convertAndSave('../../resources/feature_file')
    #converter.load('../../resources/feature_file.npy')

    #converter.printStats()


