# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.token_regex import parseCSVLine
from embeddings import W2VModel, EmbeddingLoader

class CorpusConverter(object):

    def __init__(self, csvFiles, embeddingLoader):
        """
        This class is responsible for transforming the data from the csv file to the format expected by our lstm model
        :param csvFile: the csv file to be transformed
        :param embeddingLoader: The initialized embedding loader
        :return:
        """
        self.csvFiles = csvFiles

        self.embeddings = embeddingLoader
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


    def __prepareTokens(self, x, word2idx, defaultWord=u'UNK_TK'):
        temp = parseCSVLine(x)
        tokens = temp.split(' ')
        retorno = []
        for t in tokens:
            try:
                retorno.append(word2idx[t])
            except:
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

    def printStats(self):
        print self.wordStat
        print self.predStat


    def __validate(self, sentence, roles, predicates, aux):
        assert sentence.shape == roles.shape == predicates.shape == aux.shape
        for (s, r, a) in zip(sentence, roles, aux):
            assert len(s) == len(r) == a.shape[0] and a.shape[1] == 5


    def save(self, structure, featureFile):
        np.save(featureFile, structure)

    def convert(self):
        structure = []
        for f in self.csvFiles:
            originalData = self.__loadFile(f)

            sentences = np.array(originalData['sentence'].apply(lambda x: self.__prepareTokens(x, self.embeddings.word2idx)))
            roles = np.array(originalData['roles'].apply(lambda x: self.__prepareRoles(x)))
            predicates = np.array(originalData['predicate'].apply(lambda x: self.__preparePredicate(x, self.embeddings.word2idx)))
            aux = self.__auxiliaryToNP(originalData.apply(lambda x: self.__prepareAuxiliaryFeatures(x), axis=1))
            self.__validate(sentences, roles, predicates, aux)
            # the order is important
            structure.append((sentences, predicates, aux, roles))
        return structure

    def convertAndSave(self, featureFile):
        self.save(self.convert(), featureFile)



if __name__ == '__main__':
    options = {
        "npzFile":"../../resources/embeddings/wordEmbeddings.npz",
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
    converter.printStats()


