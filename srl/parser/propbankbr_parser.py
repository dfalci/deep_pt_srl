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


import codecs
import random
import math
import unicodecsv as csv
from os.path import join
from function_utils import ContractionHandler

class PropBankParser(object):

    def __init__(self, propBankPath, verbNetPath, outputDirectory='../resources/corpus/converted'):
        """
        PropBank.br v1.1 - Parser to the format employed in our approach
        :param propBankPath:
        :param verbNetPath:
        :param outputDirectory:
        :return:
        """
        self.propBankPath = propBankPath
        self.verbNetPath = verbNetPath
        self.outputDirectory = outputDirectory
        self.verbNet = None
        self.contractionHandler = ContractionHandler()

    def prepare(self):
        """
        Prepare the internal structure to load the propbank
        :return:
        """
        self.__loadVerbNet()
        self.coRefProps= 0
        self.overflowProps = 0

    def head(self, nLines=10):
        """
        show the first n lines of the propbank file
        :param nLines:
        :return:
        """
        fileStream  = codecs.open(self.propBankPath, 'r', 'utf-8')
        i = 0
        for l in fileStream:
            print l.rstrip()
            i+=1
            if i > nLines:
                break
        fileStream.close()

    def __loadVerbNet(self):
        fileStream  = codecs.open(self.verbNetPath, 'r', 'utf-8')
        self.verbNet = {}
        for l in fileStream:
            items = l.split(u',')
            self.verbNet[items[0]] = items[1]


    def __extractFields(self, line):
        """
        Given a line from the corpus, extracts the fields of interest to our analysis
        :param line: The line from the corpus
        :return:
        """
        retorno = []
        columns = line.split('\t')
        word = columns[1].strip()
        predicate = columns[9].strip() # the predicate column
        roles = []
        for i in range(10, len(columns)):# after the predicate column, all columns are semantic role description
            roles.append(columns[i].strip())
        return (word, predicate, roles)

    def __extractAllPredicates(self, sentence):
        """
        Given a sentence, return all of its predicates
        :param verbNet:
        :return: (infinitive, original, verbnetClass)
        """
        retorno = []
        for l in sentence:
            if l[1] != '-':
                verbnetClass = -1
                try:
                    verbnetClass = self.verbNet[l[1]] # from the infinitive form, captures the verbnetClass
                except:
                    pass
                retorno.append((l[1], l[0], int(verbnetClass)))
        return retorno

    def __mustSkip(self, sentence, propIdx):
        """
        Determines contains co-reference roles, and therefore, must be skipped
        :param sentence:
        :param propIdx:
        :return: a Boolean
        """
        for i in range(0, len(sentence)):
            l = sentence[i]
            tag = l[2][propIdx]
            if tag.find('C-')!=-1:
                return True
        return False

    def __extractPredicateFeatures(self, proposition):
        """
        Extract the predicate related features : context
        :param proposition:
        :return:
        """
        for index in range(0, len(proposition)):
            if proposition[index][1] == u'V':
                propSurroundings = []
                for i in range(index - 2, index +2):
                    propSurroundings.append(proposition[i][0] if i >= 0 and i < len(proposition) else 'PAD')
                return (index, propSurroundings)

    def __fromCorpusToPropositions(self, corpus):
        retorno = []
        for sent in corpus:
            retorno.extend(sent)
        return retorno

    def __findPredicateFromRoles(self, tokens, roles):
        for (token, role) in zip(tokens, roles):
            if role == 'V':
                return token
        raise AssertionError('Cant find the predicate in the sentence')

    def __handleSentence(self, sentenceId, propositionId, sentence, verbNet):
        """
        Transforms a complete sentence read from the propbank into a collection of propositions
        :param sentenceId:
        :param propositionId:
        :param sentence:
        :param verbNet:
        :return:
        """
        propositions = []
        verbs = self.__extractAllPredicates(sentence)
        for i in range(0, len(verbs)): # for each predicate found
            proposition = []
            tokenTemp = []
            roleTemp = []
            predTemp = []
            currentVerb = verbs[i]
            inside = False
            temSobreposicao = False
            currentTag = ''
            if self.__mustSkip(sentence, i): # verifies if the sentence contains
                self.coRefProps += 1
                continue
            for j in range(0, len(sentence)):
                l = sentence[j]
                word = l[0]
                predicate = currentVerb
                tag = l[2][i]
                val = 'O'
                # faz o tagging do verbo
                if tag == u'(V*)':
                    val = 'V'
                    inside = False
                    currentTag = ''
                else:
                    # analisa se é um outside ou se é a continuação de um role já aberto
                    if tag == u'*':
                        if inside:
                            val = u'I-'+currentTag
                        else:
                            val = u'O'
                    else:
                        # analisando abertura de um role
                        autocontido = False
                        if tag.find(u'(')!=-1:
                            inside = True
                            currentTag = tag[1:tag.find(u'*')]
                            val = u'B-'+currentTag
                            if tag.find(u')')!=-1:
                                autocontido = True
                                inside = False
                                currentTag = ''
                        # se for fechamento de um role
                        if tag.find(u')')!=-1  and not autocontido:
                            val = u'I-'+currentTag
                            inside = False
                            currentTag = ''

                #quebra as palavras
                subWords = word.split('_')
                for k in range(0, len(subWords)):
                    if k>0 and val!= u'O':
                        correctTag = u'I-'+val[val.find('-')+1:]
                    else:
                        correctTag = val
                    if correctTag == 'I-':
                        temSobreposicao = True
                        correctTag = 'O'
                        break
                    tokenTemp.append(subWords[k])
                    roleTemp.append(correctTag)

            #correct the corpus tokens, doing contractions
            tokenTemp, roleTemp = self.contractionHandler.execute(tokenTemp, roleTemp)
            predTemp = self.__findPredicateFromRoles(tokenTemp, roleTemp) # as it is not possible to trust in predicates from the original corpus, captures the real predicate

            for (t, r) in zip(tokenTemp, roleTemp):
                proposition.append((t, r, (currentVerb[0], predTemp, currentVerb[1])))

            if temSobreposicao:
                print 'Overflow detected at sentence \'{}\' - Discarding it from the final version '.format(sentenceId)
                self.overflowProps +=1
                continue

            # create the predicate features
            predicateIndex, propSurroundings =  self.__extractPredicateFeatures(proposition)
            for k in range(0, len(proposition)):
                distance = ((predicateIndex - k)*-1)
                isPredicateContext = 1 if distance >= -2 and distance <= 2 else 0
                tokenPosition = k+1
                proposition[k] = proposition[k]+(distance, propSurroundings, isPredicateContext, propositionId, sentenceId, tokenPosition) # complementary features
            propositionId+=1
            propositions.append(proposition)
        return propositions

    def __readCorpus(self):
        sentenceId = 1 # sentence ids, generate automatically
        propositionId = 1 # proposition ids, generated automatically
        currentSentence = []
        sentences = []
        fileStream  = codecs.open(self.propBankPath, 'r', 'utf-8')
        for l in fileStream:
            if len(l.strip()) == 0:
                propositions = self.__handleSentence(sentenceId, propositionId, currentSentence, self.verbNet) # get all the propositions for each sentence

                sentences.append(propositions)
                currentSentence = []
                sentenceId+=1
                propositionId += len(propositions)
            else:
                currentSentence.append(self.__extractFields(l))
        fileStream.close()
        return sentences

    def printStats(self):
        print 'Co-reference propositions skipped : {}\nOverflow propositions skipped {} : '.format(self.coRefProps, self.overflowProps)

    def __extractCapitalizationFeatures(self, targetToken):
        """
        Extract capitalization features
        :param targetToken:
        :return: allcaps, firstCaps, noCaps
        """
        allCapitalized = 1 if targetToken.isupper() else 0
        firstCapitalized = 0
        noCapitalized = 0
        if not allCapitalized:
            firstCapitalized = 1 if targetToken[0].isupper() else 0
        if allCapitalized == False and firstCapitalized == False:
            noCapitalized = 1
        return (allCapitalized, firstCapitalized, noCapitalized)

    def __toNetworkFormat(self, proposition):
        """
        Given a proposition object, convert it to the format expected to our network
        :param proposition:
        :return:
        """
        sentence = ''
        results = ''
        allCaps = ''
        firstCaps = ''
        noCaps = ''
        distances = ''
        contexts = ''
        propositionId = proposition[0][6]
        predicate = proposition[0][2][1]
        for t in proposition:
            allCapitalized, firstCapitalized, noCapitalized = self.__extractCapitalizationFeatures(t[0])

            sentence += t[0]+' '
            results += t[1]+' '

            allCaps += str(allCapitalized)+' '
            noCaps += str(noCapitalized)+' '
            firstCaps += str(firstCapitalized)+' '

            distances += str(t[3])+' '

            contexts += str(t[5])+' '

        sentence = sentence.encode('utf-8').strip()
        results = results.encode('utf-8').strip()
        allCaps = allCaps.encode('utf-8').strip()
        firstCaps = firstCaps.encode('utf-8').strip()
        noCaps = noCaps.encode('utf-8').strip()
        distances = distances.encode('utf-8').strip()
        contexts = contexts.encode('utf-8').strip()

        return (propositionId, predicate, sentence, results, allCaps, firstCaps, noCaps, distances, contexts)


    def __export(self, propositions, filename):
        with open(filename, 'w') as f:
            writer = csv.writer(f, encoding='utf-8')
            writer.writerow(['propositionId', 'predicate', 'sentence', 'roles', 'allCapitalized', 'firstCapitalized', 'noCapitalized', 'distance', 'predicateContext'])
            for p in propositions:
                writer.writerow(p)
        f.close()

    def __generatePartitions(self, propositions, partitions, shuffles=True):
        """
        Partitions the database according to a split method
        :param propositions:
        :param partitions:
        :return:
        """
        random.seed(13)
        if shuffles:
            random.shuffle(propositions)# shuffles propositions and split it
        devIdx = int(math.ceil((len(propositions) * partitions[1])))
        testIdx = devIdx + int(math.ceil((len(propositions) * partitions[2])))
        devSet = propositions[:devIdx]
        testSet = propositions[devIdx:testIdx]
        trainingSet = propositions[testIdx:]

        print 'TrainingSet : {} - TestSet : {}'.format(len(trainingSet), len(testSet))

        return (trainingSet, devSet, testSet)

    def generateFeatures(self, outputDirectory=None, partition=(0, 0, 0)):
        props = self.__fromCorpusToPropositions(self.__readCorpus())
        final = []
        for p in props:
            final.append(self.__toNetworkFormat(p))
        if outputDirectory!=None:
            self.__export(final, join(outputDirectory, "propbank_full.csv"))
        if partition != (0, 0, 0):
            trainingSet, devSet, testSet = self.__generatePartitions(final, partition)
            if len(trainingSet) > 0:
                self.__export(trainingSet, join(outputDirectory, 'propbank_training.csv'))
            if len(testSet) > 0:
                self.__export(testSet, join(outputDirectory, 'propbank_test.csv'))
            if len(devSet) > 0:
                self.__export(devSet, join(outputDirectory, 'propbank_dev.csv'))


        return final




if __name__ == '__main__':
    parser = PropBankParser('../../resources/corpus/PropBankBr_v1.1_Const.conll.txt', '../../resources/corpus/verbnet_gold.csv')
    parser.prepare()
    parser.head()
    parser.generateFeatures(outputDirectory='../../resources/corpus/converted', partition=(0.9, 0, 0.10))
    parser.printStats()



