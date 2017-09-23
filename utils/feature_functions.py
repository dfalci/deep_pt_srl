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


def extractCaps(sentence):
    tokens = sentence.split(' ')
    allCaps = []
    firstCaps = []
    noCaps = []
    for t in tokens:
        a = 1 if t.isupper() else 0
        f = 0
        n = 0
        if not a:
            f = 1 if t[0].isupper() else 0
        if a == False and f == False:
            n = 1
        allCaps.append(a)
        firstCaps.append(f)
        noCaps.append(n)
    return allCaps, firstCaps, noCaps

def extractTokenPositions(sentence, predicate):
    tokens = sentence.split(' ')
    predicateIndex = -1
    for i in xrange(0, len(tokens)):
        if tokens[i] == predicate:
            predicateIndex = i
            break

    assert predicateIndex !=-1
    distanceArray = []
    contextArray = []

    for k in range(0, len(tokens)):
        distance = ((predicateIndex - k)*-1)
        isPredicateContext = 1 if distance >= -2 and distance <= 2 else 0
        distanceArray.append(distance)
        contextArray.append(isPredicateContext)
    return distanceArray, contextArray

def translateToken(token, word2idx, defaultToken = None):
    try:
        return word2idx[token]
    except:
        if defaultToken != None:
            return defaultToken
        else:
            raise AssertionError('Token {} is undefined'.format(token))

def translatePredicates(sentence, predicate, word2idx, defaultToken=None):
    pred = translateToken(predicate, word2idx, defaultToken)
    sentenceLen = len(sentence.split(' '))
    predicates = []
    for i in xrange(0, sentenceLen):
        predicates.append(pred)
    return predicates

def translateSentence(sentence, word2idx, defaultToken = None):
    tokens = sentence.split(' ')
    translation = [translateToken(t,word2idx) for t in tokens]
    return translation


def extractFeaturesFromSentence(sentence, predicate, word2idx, unkToken=None, unkPred=None):
    """
    Given a sentence, its predicate and the word2idx dictionary returns the indexes
    :param sentence:
    :param predicate:
    :param word2idx:
    :return:
    """
    allCaps, firstCaps, noCaps = extractCaps(sentence)
    distance, context = extractTokenPositions(sentence, predicate)
    convertedSentence = translateSentence(sentence, word2idx)
    convertedPredicate = translatePredicates(sentence, predicate, word2idx)
    return convertedSentence, convertedPredicate, allCaps, firstCaps, noCaps, context, distance




if __name__ == '__main__':
    sentence = u'Este e e e mais um pequeno teste : FUMEC'
    predicate = u'mais'
    word2idx = { u'Este':0, u'e':1, u'mais':2, u'um':3, u'pequeno':4, u'teste':5, u':':6, u'FUMEC':7}
    print extractFeaturesFromSentence(sentence, predicate, word2idx)
