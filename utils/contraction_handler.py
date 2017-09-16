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

class ContractionHandler(object):
    """
    Handle contractions in the Portuguese Language within a semantic role labeling context.
    It performs token contraction, eliminating the semantic roles when necessary

    """

    def __init__(self):
        self.article = ['a', 'as', 'o', 'os']
        self.article_masc = ['o', 'os']
        self.adverb = [u'aí', 'aqui', 'ali']
        self.pronoun = ['ele', 'eles', 'ela', 'elas', 'esse', 'esses', 'essa', 'essas', 'isso', 'este', 'estes', 'esta', 'estas', 'isto']
        self.pronoun_crass = ['aquele', 'aqueles', 'aquela', 'aquelas', 'aquilo']
        self.ref = ['o', 'os', 'a', 'as', 'me', 'se', 'te', 'vos', 'lhe', 'lho', 'lhas', 'lhos', 'lha', 'lo', 'la', 'los', 'las', 'lhes', 'no', 'na', 'nos']

    def mark(self, lowerPattern, upperPattern, currentToken, nextToken, nextTokenAddition, currentRole, nextRole, tokenList, roleList):
        role = currentRole
        if currentRole == u'O' and nextRole != u'O':
            role = nextRole
        start = upperPattern if currentToken[0].isupper() else lowerPattern
        tokenList.append(start+nextTokenAddition)
        roleList.append(role)
        return True

    def markSimple(self, currentToken, nextToken, currentRole, nextRole, tokenList, roleList):
        role = currentRole
        if currentRole == u'O' and nextRole != u'O':
            role = nextRole
        tokenList.append(currentToken+nextToken)
        roleList.append(role)
        return True


    def execute(self, tokens, roles):
        """
        Executes token contractions
        :param tokens: A list of tokens in a sentence
        :param roles: A list of corresponding semantic roles
        :return:
        """
        assert (len(tokens) == len(roles))

        return_tokens = []
        return_roles = []
        contracted = False
        for i in xrange(0, len(tokens)):
            try:
                currentToken = tokens[i]
                currentRole = roles[i]
                nextRole = roles[i+1]
                nextToken = tokens[i+1]


            except IndexError:
                if not contracted:
                    return_tokens.append(currentToken)
                    return_roles.append(currentRole)
                break

            if contracted :
                contracted = False
                continue

            if currentToken.lower() == 'de' and nextToken.lower() in (self.article + self.pronoun + self.pronoun_crass + self.adverb):
                contracted = self.mark('d', 'D', currentToken, nextToken, nextToken.lower(), currentRole, nextRole, return_tokens, return_roles)

            elif currentToken.lower() == 'em' and nextToken.lower() in (self.article + self.pronoun + self.pronoun_crass):
                contracted = self.mark('n', 'N', currentToken, nextToken, nextToken.lower(), currentRole, nextRole, return_tokens, return_roles)

            elif currentToken.lower() == 'por' and nextToken in self.article:
                contracted = self.mark('pel', 'Pel', currentToken, nextToken, nextToken.lower(), currentRole, nextRole, return_tokens, return_roles)

            elif currentToken.lower() == 'a':
                if nextToken.lower() in self.pronoun_crass:
                    contracted = self.mark(u'à', u'À', currentToken, nextToken, nextToken[1:].lower(), currentRole, nextRole, return_tokens, return_roles)
                elif nextToken.lower() in self.article_masc:
                    contracted = self.mark('a', 'A', currentToken, nextToken, nextToken.lower(), currentRole, nextRole, return_tokens, return_roles)
                elif nextToken.lower() == 'a':
                    contracted = self.mark(u'à', u'À', currentToken, nextToken, '', currentRole, nextRole, return_tokens, return_roles)
                elif nextToken.lower() == 'as':
                    contracted = self.mark(u'às', u'Às', currentToken, nextToken, '', currentRole, nextRole, return_tokens, return_roles)
            elif currentToken[len(currentToken)-1] == '-':
                if nextToken.lower() in self.ref:
                    contracted = self.markSimple(currentToken, nextToken, currentRole, nextRole, return_tokens, return_roles)

            if contracted == False:
                return_tokens.append(currentToken)
                return_roles.append(currentRole)
                contracted = False

        assert len(return_tokens) == len(return_roles)

        return return_tokens, return_roles


if __name__ == '__main__':
    #tokens = [u'\xab', u'C\xe2mera', u'Manchete', u'\xbb', u'\xe9', u'o', u'nome', u'de', u'o', u'novo', u'programa', u'jornal\xedstico', u'que', u'estr\xe9ia', u'quarta-feira', u',', u'a', u'as', u'22h30', u',', u'em', u'a', u'Rede', u'Manchete', u'.']

    #teste = ContractionHandler().execute(tokens, tokens)

    #print teste[0]
    #print teste[1]

    texto = u'nao há hipótese de o plano real ser lançado'
    roles = u'O O O O B-A1 I-A1 O O V'


    tokens = texto.split(' ')
    roles = roles.split(' ')

    teste = ContractionHandler().execute(tokens, roles)

    print teste[0]
    print teste[1]

