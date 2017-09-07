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

import subprocess

class CoNLLEvaluator(object):
    """
    This class converts role predictions made using the IOB format to the format expected by "srl-eval.pl" - the official evaluation script of CoNLL.
    Notice that this script only works in unix based systems.
    """

    def __init__(self, srlEvalScript):
        self.golden = []
        self.predicted = []
        self.predicates = []
        self.srlEvalScript = srlEvalScript;


    def addSentence(self, golden, predicted, predicate):
        """
        Adds propositions to the end of the list
        :param golden: The golden predictions from the test set
        :param predicted: The predictions from the neural network model
        :param predicate: The predicate of a proposition
        :return:
        """
        self.golden.append(golden)
        self.predicted.append(predicted)
        self.predicates.append(predicate)


    def __write(self, list, file):
        file = open(file, "w")
        for sentIdx in xrange(0, len(list)):
            sentence = list[sentIdx]
            for i in xrange(0, len(sentence)):
                item = {'predicate':'-', 'end':'', 'start':''}
                currentTag = sentence[i]
                nextTag = None
                if i < len(sentence) - 1:
                    nextTag = sentence[i+1]

                if currentTag == 'O':
                    item['start'] = '*'


                if currentTag.startswith('B-'):
                    item['start'] = '('+currentTag[2:]+'*'
                    if nextTag == None or not nextTag.startswith('I-'):
                        item['end'] = currentTag[2:]+')'

                if currentTag.startswith('I-'):
                    item['start'] = '*'
                    if nextTag == None or nextTag != currentTag:
                        item['end'] = currentTag[2:]+')'

                if currentTag == 'V':
                    item['start'] = '(V*'
                    item['end'] = 'V)'
                    item['predicate'] = self.predicates[sentIdx]
                file.write('{0}\t{1}\n'.format(item['predicate'], item['start']+item['end']))
            file.write('\n')
        file.close()


    def write(self, outputFile='../results/official.eval', goldfile='../results/gold.props', predictedfile='../results/pred.props'):
        """
        writes down results on the given files
        :param goldfile:
        :param predictedfile:
        :return:
        """
        self.__write(self.golden, goldfile)
        self.__write(self.predicted, predictedfile)
        if self.srlEvalScript!= None:
            with open(outputFile, 'w') as file:
                subprocess.call(['perl', self.srlEvalScript, goldfile, predictedfile], stdout=file)
            file.close()


if __name__ == '__main__':
    gold = [
        ['B-A0', 'I-A0', 'I-A0', 'O', 'V', 'B-A1', 'I-A1', 'I-A1', 'O'],
        ['B-A0', 'I-A0', 'I-A0', 'O', 'V', 'B-A1', 'I-A1', 'I-A1', 'O']
    ]
    predicted = [
        ['B-A0', 'I-A0', 'O', 'O', 'V', 'B-A1', 'I-A1', 'I-A1', 'O'],
        ['B-A0', 'I-A0', 'O', 'O', 'V', 'B-A1', 'I-A1', 'I-A1', 'O']
    ]

    predicates = ['test', 'test2']

    converter = CoNLLEvaluator('../resources/srl-eval.pl')
    for i in xrange(0, len(gold)):
        converter.addSentence(gold[i], predicted[i], predicates[i])

    converter.write()