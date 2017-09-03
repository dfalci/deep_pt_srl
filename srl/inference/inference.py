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
import logging

class Node(object):
    def __init__(self, parent, value, position, tagPredicted):
        self.propCons = set(['V'])
        self.propIm = set(['A0', 'A1', 'A2', 'A3', 'A4', 'A5'])
        self.parent = parent
        self.value = value
        self.tagPredicted = tagPredicted
        self.position = position
        self.f = 0
        self.cumulativeCost = 0 if self.parent==None else self.parent.f
        self.openingCost = 0
        self.iob = 0
        self.prop = 0
        self.cost = 0
        self.calculate()


    def __str__(self):
        """
        for debugging purposes
        :return:
        """
        return 'Tag index {0} - {1} \nvalue : {2}\ncost : {3} : {4} - {5}\ntotal : {6}\n'.format(self.position, self.tagPredicted, self.value, self.openingCost, self.iob, self.prop, self.f)

    def __repr__(self):
        """
        for debugging purposes
        :return:
        """
        return 'Tag index {0} - {1} \nvalue : {2}\ncost : {3} : {4} - {5}\ntotal : {6}\n'.format(self.position, self.tagPredicted, self.value, self.openingCost, self.iob, self.prop, self.f)

    def setChildren(self, children):
        self.children = children

    def __calculateBIO(self):
        if self.parent == None:
            return float(0)
        if self.tagPredicted.startswith('V') or self.tagPredicted.startswith('O'):
            return float(0)

        if self.tagPredicted.startswith('I-'):
            if (self.parent.tagPredicted.startswith('B-') or self.parent.tagPredicted.startswith('I-')) and self.tagPredicted.endswith(self.parent.tagPredicted[self.parent.tagPredicted.find('-'):]):
                return float(0)
            else:
                return float(100)
        if self.tagPredicted.startswith('B-'):
            if self.parent.tagPredicted.startswith('B-') and self.tagPredicted.endswith(self.parent.tagPredicted[self.parent.tagPredicted.find('-'):]):
                return float(100)
        return float(0)


    def __calculatePropBank(self):
        role = self.tagPredicted
        if self.tagPredicted.find('-') != -1:
            role = self.tagPredicted[self.tagPredicted.find('-')+1:]

        checkCons = role in self.propCons
        checkIm = role in self.propIm

        current = self.parent
        found = False
        together = True

        while current != None:
            currentRole = current.tagPredicted
            if current.tagPredicted.find('-') != -1:
                currentRole = current.tagPredicted[current.tagPredicted.find('-')+1:]

            if currentRole != role:
                together = False
            else:
                if checkIm and self.tagPredicted.startswith('B-') and current.tagPredicted.startswith('B-'):
                    together = False

            if checkCons and currentRole == role:
                found = True
                break

            if checkIm and currentRole == role and together == False:
                found = True
                break

            current = current.parent

        if found:
            return float(100)
        return float(0)

    def calculate(self):
        self.iob = self.__calculateBIO()
        self.prop = self.__calculatePropBank()
        self.cost =  self.iob + self.prop
        self.openingCost = self.value - self.cost;
        self.f = self.cumulativeCost + self.openingCost;

class SRLInference(object):

    def __init__(self, tagMap, tagList):
        """
        The inference model employed in our project
        :param tagMap:
        :param tagList:
        :return:
        """
        self.tagMap = tagMap
        self.tagList = tagList

    def __generateChildren(self, parent, nBest=10000):
        children = []
        currentIndex = parent.position[0] + 1
        if (currentIndex < len(self.predictionMatrix)):
            bestIndex, bestValues = self.__getNBestIndex(self.predictionMatrix[currentIndex], nBest)
            for i in xrange(0, len(bestIndex)):
                children.append(Node(parent, bestValues[i], (currentIndex, bestIndex[i]), self.tagList[bestIndex[i]]))
        return children

    def predict(self, predictionMatrix):
        """
        Executa o search e converte o resultado para o formato utilizado no resto do sistema
        :param predictionMatrix:
        :param nBestSolutions:
        :return:
        """
        solucao = self.search(predictionMatrix, 1)[0]
        predicao = solucao
        resultado = np.zeros(predictionMatrix.shape)
        while predicao != None:
            resultado[predicao.position[0]][predicao.position[1]] = 1
            predicao = predicao.parent
        return resultado, self.__getPath(solucao)



    def search(self, predictionMatrix, nBestSolutions=1, debug=False):
        """
        Approach :

        - Create the first elementos
        add the first elements
        while the openset is still opened
            selects the element with the higher score in openlist
        finish later

        :param predictions:
        :return:
        """
        self.predictionMatrix = predictionMatrix
        openset = set(self.createFirstNodes(1000))

        def getCost(elem):
            return elem.f

        def getValue(elem):
            return elem.openingCost

        closedset = set()
        solutions = []
        iterations = 0
        while len(openset) != 0:
            iterations+=1
            currentNode = sorted(openset, key=getValue, reverse=True)[0]
            if debug:
                logging.debug('opening : {0} - cumulative = {1}, openinig = {2}'.format(currentNode.position, currentNode.f, currentNode.openingCost))
            if currentNode.position[0] == self.predictionMatrix.shape[0] -1:
                solutions.append(currentNode)
                if len(solutions) == nBestSolutions:
                    break

            openset.remove(currentNode)
            #logging.debug('removing : {0} - Length : {1}'.format(currentNode.position, len(openset)))
            closedset.add(currentNode)
            children = self.__generateChildren(currentNode)

            for child in children:

                if currentNode.position == (1, 1):
                    logging.debug(child.__str__())
                #if child.cost == 0:
                openset.add(child)
        if debug:
            logging.debug('ended in '+str(iterations)+' iterations')
            for x in solutions:
                self.__printPath(x)
        return solutions


    def __printPath(self, node):
        temp = self.__getPath(node)
        logging.debug(temp.__str__()+ ' - '+ str(node.f))

    def __getPath(self, node):
        current = node
        temp = []
        while current != None:
            temp.append(current.tagPredicted)
            current = current.parent
        return list(reversed(temp))


    def createFirstNodes(self, num):
        bestIndex, predictions = self.__getNBestIndex(self.predictionMatrix[0], num, False)
        firstNodes = []
        for i in xrange(0, len(bestIndex)):
            firstNodes.append(Node(None, predictions[i], (0, bestIndex[i]), self.tagList[bestIndex[i]]))
        return firstNodes


    def __getNBestIndex(self, arr, n, show=False):
        '''
        Captura os indices dos x maiores elementos de um dado array
        :param arr:
        :param n:
        :return: os indices e seus respectivos valores
        '''

        n = len(arr) if n > len(arr) else n
        index = np.argpartition(arr, -n)[-n:]
        if show:
            print index
            print arr[index]
        return index, arr[index]

if __name__ == '__main__':
    sentenca = ['o', 'rato', 'roeu', 'a', 'roupa', 'do', 'rei']
    tagList = ['B-ARG0', 'I-ARG0', 'V', 'B-ARG1', 'I-ARG1', 'O', 'B-ARG2']
    tagMap = {'B-ARG0':0, 'I-ARG0':1, 'V':2, 'B-ARG1':3, 'I-ARG1':4, 'O':5, 'B-ARG2':6}
    predictionMatrix = np.array([
    #    ba0  ia0     v    ba1    ia1    o     ba2
        [0.34, 0.25, 0.01, 0.11, 0.30, 0.33, 0.12], # o - ba0
        [0.30, 0.39, 0.05, 0.34, 0.36, 0.05, 0.05], # rato - ia0
        [0.12, 0.10, 0.22, 0.07, 0.09, 0.22, 0.15], # roeu - v
        [0.11, 0.12, 0.05, 0.12, 0.13, 0.11, 0.11], # a - ba1
        [0.8, 0.11, 0.05, 0.15, 0.16, 0.10, 0.11], # roupa - ia1
        [0.14, 0.10, 0.10, 0.11, 0.05, 0.15, 0.13], # do - o
        [0.5, 0.25, 0.27, 0.05, 0.05, 0.05, 0.27], # rei - ba2
        [0.5, 0.25, 0.27, 0.05, 0.05, 0.30, 0.27], # rei - ba2
    ])
    #'B-ARG0'
    #'I-ARG0'
    #'V', \
    #'B-ARG1', \
    #'I-ARG1', \
    #'O'
    #'B-ARG2'
    #logging.basicConfig(filename='output.log',level=logging.DEBUG)
    inference = SRLInference(tagMap, tagList)
    print inference.predict(predictionMatrix)




