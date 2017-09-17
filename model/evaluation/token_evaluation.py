__author__ = 'daniel'
import numpy as np
import pandas as pd

class TokenEvaluation(object):

    def __init__(self, tagMap, tagList, ignoreList=[]):
        """
        Creates a token evaluator under which the neural network is primarily evaluated
        :param tagMap: The tag map
        :param tagList: The tag list
        :param ignoreList: The list of tokens that must be discarded from the evaluation procedure
        :return:
        """
        self.tagMap = tagMap
        self.tagList = tagList
        self.nClasses = len(self.tagList)
        self.nValidClasses = self.nClasses - len(self.tagList)
        self.matrix = np.zeros((self.nClasses, self.nClasses))
        self.nExamples = 0
        self.ignoreList = []
        for i in xrange(0, len(ignoreList)):
            self.ignoreList.append(tagMap[ignoreList[i]])

    def reset(self):
        self.matrix = np.zeros((self.nClasses, self.nClasses))
        self.nExamples


    def addSample(self, y_true, y_pred):
        for i in xrange(0, len(y_true)):
            idxTrue = y_true[i]
            idxPred = y_pred[i]
            if not idxTrue in self.ignoreList:
                self.matrix[idxPred][idxTrue] +=1
                self.nExamples += 1

    def getConfusionMatrix(self):
        return pd.DataFrame(self.matrix, columns=self.tagList, index=self.tagList)


    def calculate(self, lr=0, column=None):
        correct = 0
        totalRows = 0
        totalColumns = 0
        sumRows = self.matrix.sum(axis=1)
        sumColumns = self.matrix.sum(axis=0)

        precision = np.zeros((self.nClasses, ))
        recall = np.zeros((self.nClasses, ))
        f1 = np.zeros((self.nClasses,))

        tagResult = {}

        discount = 0

        for i in xrange(0, self.nClasses):
            totalRows += sumRows[i]
            totalColumns += sumColumns[i]
            correct += self.matrix[i][i]

            precision[i] = 0 if sumRows[i] == 0 else self.matrix[i][i] / sumRows[i]

            recall[i] = 0 if sumColumns[i] == 0 else self.matrix[i][i]/sumColumns[i]

            f1[i] = 0 if (precision[i] + recall[i]) == 0 else (2 * (precision[i] * recall[i]))/(precision[i] + recall[i])

            if sumRows[i] == 0 and sumColumns[i] == 0:
                discount += 1



            tagResult[self.tagList[i]] = {'precision':precision[i], 'recall':recall[i], 'f1':f1[i]}


        macroPrecision= precision.sum(axis=0)/float(self.nClasses - discount)
        macroRecall =  recall.sum(axis=0)/float(self.nClasses - discount)
        macroF1 = f1.sum(axis=0)/float(self.nClasses - discount)

        return {'tokenMacroPrecision':macroPrecision, 'tokenMacroRecall':macroRecall, 'tokenMacroF1':macroF1, 'tokenMeasuresPerTag':tagResult, 'lr':float(lr)}




if __name__ == '__main__':
    from sklearn.metrics import precision_recall_fscore_support
    tagMap = {'A0':0, 'A1':1, 'A2':2, 'A3': 3, 'V':4, 'O':5}
    tagList = ['A0', 'A1', 'A2', 'A3', 'V', 'O']
    ev = TokenEvaluation(tagMap, tagList, [])

    y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 2, 4, 3])
    ev.addSample(y_true, y_pred)
    print ev.getConfusionMatrix()

    print 'Accuracy measures'
    print ev.calculate()

    #print 'teste regular : '
    print precision_recall_fscore_support(y_true, y_pred, average='macro')






