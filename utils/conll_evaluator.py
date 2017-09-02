# -*- coding: utf-8 -*-
import subprocess

class CoNLLEvaluator(object):
    """
    This class converts role predictions made using the IOB format to the format expected by "srl-eval.pl" - the official evaluation script of CoNLL.
    Notice that this script only works in unix based systems.
    """

    def __init__(self, srlEvalScript='../resources/srl-eval.pl'):
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

    converter = CoNLLEvaluator()
    for i in xrange(0, len(gold)):
        converter.addSentence(gold[i], predicted[i], predicates[i])

    converter.write()