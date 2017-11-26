# -*- coding: utf-8; -*-
import nlpnet
import pandas as pd

nlpnet.set_data_dir('/Users/danielfalci/Downloads/srl-pt')
tagger = nlpnet.SRLTagger()


def getByPredicate(predicate, result):
    #quando nao acha nada
    if len(result.arg_structures) == 0:
        return {}, []
    for este in result.arg_structures:
        if este[0] == predicate:
            return este[1], result.tokens
    # quando nao acha o predicado
    return {}, []

def handleTag(tag):
    if tag.startswith('V:') or (tag.startswith('A') and ':' in tag):
        quantidade = int(tag[tag.find(':')+1:])
        tagFinal = tag[:tag.find(':')]
        if quantidade == 1:
            return [u'('+tagFinal+u'*'+tagFinal+u')']
        else:
            temp = []
            for i in xrange(0, quantidade):
                if i == 0:
                    temp.append(u'('+tagFinal+u'*')
                elif i < quantidade -1:
                    temp.append(u'*')
                else:
                    temp.append(u'*'+tagFinal+u')')
            return temp
    else:
        return [u'*']

def tagSentence(tagger, predicate, sentence):
    print sentence, predicate
    temporary = sentence
    quant = 0
    mapa, tokens = getByPredicate(predicate, tagger.tag(sentence)[0])
    if len(mapa)>0:
        for key in mapa:
            quant = len(mapa[key])
            trecho = u' '.join(mapa[key])
            if quant == 1:
                trecho = ''+trecho+' '
                temporary = temporary.replace(trecho, key+':'+str(quant)+' ')
            else:
                temporary = temporary.replace(trecho, key+':'+str(quant))

    temporary = temporary.split(' ')
    finalTags = []
    for item in temporary:
        finalTags.extend(handleTag(item))
    return finalTags, tokens


def tagGoldSentence(predicate, goldRoles):
    roles = goldRoles.split(' ')



file = open('./results/erick/testenlp.props', "w")
#goldFile = open('./results/erick/gold.props', "w")
data = pd.read_csv('./resources/corpus/converted/propbank_test.csv', encoding='utf-8')
iterations = len(data)
for i in xrange(0, iterations):
    sentence = data['sentence'][i]
    predicate = data['predicate'][i]
    goldRoles = data['roles'][i]
    sent, tokens = tagSentence(tagger, predicate, sentence)
    for x, t in enumerate(sent, tokens):
        file.write('{0}\t{1}\n'.format(t, x))
    file.write('\n')
file.close()
