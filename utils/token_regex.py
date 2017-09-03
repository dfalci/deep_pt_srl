# -*- coding: utf-8 -*-

import unicodedata
import re

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