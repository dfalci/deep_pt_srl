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

import unicodedata
import re
import pandas as pd


def splitTokens(line, items):
    temp = parseCSVLine(line).split(' ')
    for t in temp:
        items.add(t)

def extractAllTokens(csvFile, columnName='sentence'):
    originalData = pd.read_csv(csvFile)
    items = set()
    originalData[columnName].apply(lambda x: splitTokens(x, items))
    return items


def parseCSVLine(line, toLower=True, numberTransform='#', removeAccents=True, strip=True):
    """
    This method is designed to be executed after the processing and may not remove tokens, but just transform them
    :param line:
    :return:
    """

    if removeAccents:
        nfkd = unicodedata.normalize('NFKD', unicode(line, 'utf-8'))
        line = u"".join([c for c in nfkd if not unicodedata.combining(c)])

    if toLower:
        line = line.lower()

    #line = re.sub(r'\d+(\d)*(\.\d+)*(\,\d+)*(\ \d+)?', numberTransform, line)
    line = re.sub(r'\d+(\d)*(\.\d+)*(\,\d+)?', numberTransform, line)

    if strip:
        return correct(line.strip())

    return correct(line)

def correct(line):
    # substituir estes camaradas abaixo por uma expressão regular
    line = line.replace('dr. ', 'dr ')
    line = line.replace('dra. ', 'dra ')
    line = line.replace('sr. ', 'sr ')
    line = line.replace('sra. ', 'sra ')
    line = line.replace('mr. ', 'mr ')
    line = line.replace('ilmo. ', 'ilmo ')
    line = line.replace('av. ', 'av ')
    line = line.replace('fund. ', 'fund ')
    line = line.replace('pag. ', 'pag ')
    line = line.replace('terra,', 'terra')
    line = line.replace('sexta-feira,', 'sexta-feira')
    line = line.replace('ed. ', 'ed ')
    line = line.replace('no. ', 'no ')
    line = line.replace('etc. ', 'etc ')
    line = line.replace('tel. ', 'tel ')
    line = line.replace('...', '.')
    line = line.replace('mais! ', 'mais ')
    line = line.replace('jr. ', 'junior ')
    line = line.replace('cia. ', 'cia ')
    line = line.replace('inc. ', 'inc ')
    line = line.replace('a.p.s.', 'aps')
    line = line.replace('arts. ', 'artigos ')
    line = line.replace('art. ', 'artigo ')
    line = line.replace('fazendo.', 'fazendo')
    line = line.replace('d. ', 'd ')
    line = line.replace('$/', '$')
    line = line.replace('$%', '$')
    line = line.replace('#.', '#')
    line = line.replace('cosmetica\'#,', 'cosmetica')
    line = line.replace('c.f. ', 'cf ')

    return line


if __name__ == '__main__':
    temp = u"""Às É Este é apenas um sr. Dr.  dr. joao doutor. teste para identificar se o parser está funcionando apropriadamente : R$ 200 00 .""".encode('utf-8')
    print parseCSVLine(temp)