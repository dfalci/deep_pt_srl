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

from token_evaluation import TokenEvaluation
from conll_evaluator import CoNLLEvaluator
from utils import RoleConverter
import numpy as np
import os
import json


class Evaluator(object):

    def __init__(self, testData, globalInference, tagList, tagMap, generalResultFile):
        """
        Makes the comparison process
        :param testData:
        :param globalInference:
        :return:
        """
        self.testData = testData
        self.globalInference = globalInference
        self.tagList = tagList
        self.tagMap = tagMap
        self.generalResultFile = generalResultFile
        self.nn = None
        self.evaluations = []



    def __createDirectoryIfNeeded(self, directory):
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)

    def prepare(self, nn, targetDirectory, conllFile):
        self.nn = nn
        self.tokenEvaluation = TokenEvaluation(self.tagMap, self.tagList)
        self.officialEvaluation = CoNLLEvaluator(conllFile)
        self.targetDirectory = targetDirectory
        self.__createDirectoryIfNeeded(self.targetDirectory)


    def evaluate(self):
        teste = 0
        for sent, pred, aux, role in zip(self.testData[0], self.testData[1], self.testData[2], self.testData[3]):
            print 'fazendo predicao'
            y = self.nn.predict([sent, pred, aux])
            print 'predicao encerrada'
            print 'fzendo inferencia'
            y, tags = self.globalInference.predict(y[0])
            print 'inferencia encerrada'

            #create an index array for predictions and golden labels
            print 'ajustando dados'
            y = np.argmax(y, axis=1)
            golden = np.argmax(role[0], axis=1)
            print 'dados ajustados'

            print 'adicionando sample'
            self.tokenEvaluation.addSample(golden, y)
            self.officialEvaluation.addSentence(RoleConverter.fromIndexToRoles(golden, self.tagList), RoleConverter.fromIndexToRoles(y, self.tagList), pred[0][0])
            print 'registrando sample'
            teste +=1
            if teste > 3:
                break

        print 'calculando'
        evaluation = self.tokenEvaluation.calculate()
        self.evaluations.append(evaluation)
        print 'calculado'
        print 'escrevendo resultados'
        self.__writeResults()
        print 'terminou'
        return evaluation

    def __writeResults(self):
        with open(self.generalResultFile, 'w') as f:
            f.write(json.dumps(self.evaluations))
            f.close()
        self.officialEvaluation.write(self.targetDirectory+'/official.txt', self.targetDirectory+'/goldprops.props', self.targetDirectory+'/pred.props')