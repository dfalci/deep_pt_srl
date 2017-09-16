# encoding: utf-8
import os
import io
import re
import sys
from nltk.tokenize import RegexpTokenizer
import unicodedata
import nltk
import unicodecsv as csv

class CorpusCreator(object):

    def __init__(self, numberOfSentences, targetTokens, word2idx, dirname='/Users/danielfalci/development/pycharm/nlp/', fname='wiki_00', tokenCoverage=1, maxLength=50, minLength=7):
        """
        Read the wikipedia dump file generating a new sentence corpus
        :param numberOfSentences:
        :param targetTokens:
        :param word2idx:
        :param dirname:
        :param fname:
        :param tokenCoverage:
        :param maxLength:
        :param minLength:
        :return:
        """
        self.nSentences = numberOfSentences
        self.targetTokens = targetTokens
        self.word2idx = word2idx
        self.tokenCoverage = tokenCoverage
        self.sentenceGenerator = SentenceGenerator(dirname, fname)
        self.maxLength = maxLength
        self.minLength = minLength
        self.blackList = [u'é', u'está', u'e', u'esta', u'dá', u'da', u'pôr', u'por']


    def containTokenOfInterest(self, sentence):
        if not sentence[0][0].isupper():
            return None
        if sentence[len(sentence)-1] != '.':
            return None
        for t in sentence:
            if t.lower() in self.targetTokens:
                return t.lower()
        return None

    def checkTokenCoverage(self, sentence):
        tokens = len(sentence)
        errors = 0
        for t in sentence:
            try:
                nfkd = unicodedata.normalize('NFKD', t)
                e = u"".join([c for c in nfkd if not unicodedata.combining(c)])
                e = e.lower()
                t = self.word2idx[e]
            except:
                errors += 1
        if errors > 0:
            index = float(errors) / float(tokens)
            return index >= self.tokenCoverage
        return True

    def __toSentence(self, sentence):
        ret = ''
        for i, t in enumerate(sentence):
            ret += t
            if i < len(sentence) -1:
                ret +=' '
        return ret

    def scanWikipediaCorpus(self, outputFile):
        sentDb = []
        nSentences = 0
        for sentence_no, sentence in enumerate(self.sentenceGenerator):
            if len(sentence) >= self.minLength and len(sentence) <= self.maxLength:
                pred = self.containTokenOfInterest(sentence)
                if pred != None and not pred in self.blackList and self.checkTokenCoverage(sentence):
                    #print sentence
                    sentDb.append((sentence_no, pred, self.__toSentence(sentence)))
                    nSentences +=1
            if nSentences >= self.nSentences:
                break
        self.__export(sentDb, outputFile)
        return sentDb

    def __export(self, propositions, filename):
        with open(filename, 'w') as f:
            writer = csv.writer(f, encoding='utf-8')
            writer.writerow(['propositionId', 'predicate', 'sentence'])
            for p in propositions:
                writer.writerow(p)
        f.close()

class SentenceGenerator:

    def __init__(self, dirname='/Users/danielfalci/development/pycharm/nlp/', fname='wiki_00', nGramModel=None):
        self.dirname = dirname
        self.fname = fname
        self.nGramModel = nGramModel
        self.sentenceBroker = SentenceBroker()
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        self.sentNum = 0
        self.tokenNum = 0
        self.artNum = 0

    def __iter__(self):
        for line in io.open(os.path.join(self.dirname, self.fname), encoding="utf-8"):
            if not self.sentenceBroker.mustSkip(line):
                #value = self.sentenceBroker.transformSentence(line)
                for frase in self.sent_tokenizer.tokenize(line):
                    value = self.sentenceBroker.transformSentence(frase)
                    #if self.nGramModel != None:
                    #    value = self.nGramModel[value]
                    #self.sentNum = self.sentNum + 1
                    #self.tokenNum = self.tokenNum + len(value)
                    yield value

class SentenceBroker:

    def __init__(self):
        self.skipElements= (
            '<doc',
            '</doc'
        )

    def mustSkip(self, line):
        for este in self.skipElements:
            if este in line:
                return True
        return False

    def captureSentences(self, line):
        #return re.split(r' *[\.\?\;\!!][\'"\)\]]* *', line)
        return re.split(r' (?<=[\.!\?\;\:])\s+', line)


    def tokenize(self, text):
        """
        Primeira camada de tokenizacao
        """
        tokenizer_regexp = r'''(?ux)
        # a ordem e importante!!!!!!
        # Os padroes estruturados vem primeiro
        [a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+|    # emails
        (?:https?://)?\w+(?:\.\w+)+(?:/\w+)*|                  # urls
        (?:[\#@]\w+)|                     # Hashtagds
        (?:[^\W\d_]\.)+|                  # Abreviacoes de pais : U.S.A., E.U.A.
        (?:[DSds][Rr][Aa]?)\.|            # Abreviacoes do tipo Sr. Sra., Dr. Dra....
        (?:\B-)?\d+(?:[:.,]\d+)*(?:-?\w)*|
            # numeros no formato numerico
            # \B- evita pegar tipo f-15 como um valor numerico negativo
        \.{3,}|                           # tres pontinhos
        \w+|                              # palavras
        -+|                               # qualquer sequencia de palavras separadas por -
        \S                                # qualquer caractere nao espaco
        '''
        tokenizer = RegexpTokenizer(tokenizer_regexp)

        return tokenizer.tokenize(text)

    def prepareLine(self, line):
        line = line.rstrip('\n')
        line = re.sub(r'\d+(\d)*(\.\d+)*(\,\d+)*(\ \d+)?', '#', line)


        #separa as palavras com virgula
        line = re.sub(r'[ ]*,[ ]*', ' , ', line)
        line = re.sub(r'[ ]*\?[ ]*', ' ? ', line)
        line = re.sub(r'[ ]*\![ ]*', ' ! ', line)
        line = re.sub(r'[ ]*\.[ ]*', ' . ', line)
        line = re.sub(r'[ ]*\:[ ]*', ' : ', line)
        line = re.sub(r'[ ]*\"[ ]*', ' " ', line)
        line = re.sub(r'[ ]*\'[ ]*', " ' ", line)
        line = re.sub(r'[ ]*\([ ]*', ' ( ', line)
        line = re.sub(r'[ ]*\)[ ]*', ' ) ', line)
        line = re.sub(r'[ ]*\;[ ]*', ' ; ', line)
        return line

    def transformSentence(self, line):
        line = self.prepareLine(line)
        return line.strip().split(' ')

    #captura todas as sentencas de uma linha
    def splitSentence(self, line):
        yield line.split(' ')
        #yield self.tokenize(line)


if __name__ == '__main__':
    c = CorpusCreator(50, ['fazer', 'remediar'], {})
    c.scanWikipediaCorpus()