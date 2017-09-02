# -*- coding: utf-8 -*-
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

    def mark(self, lowerPattern, upperPattern, currentToken, nextToken, nextTokenAddition, currentRole, tokenList, roleList):
        start = upperPattern if currentToken[0].isupper() else lowerPattern
        tokenList.append(start+nextToken.lower())
        roleList.append(currentRole)
        return True


    def execute(self, tokens, roles):
        """
        Executes token contractions
        :param tokens: A list of tokens in a sentence
        :param roles: A list of corresponding semantic roles
        :return:
        """
        if (len(tokens) != len(roles)):
            raise Exception("erroneus sentence")
        return_tokens = []
        return_roles = []
        contracted = False
        for i in xrange(0, len(tokens)):
            try:
                currentToken = tokens[i]
                currentRole = roles[i]
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
                contracted = self.mark('d', 'D', currentToken, nextToken, nextToken.lower(), currentRole, return_tokens, return_roles)

            elif currentToken.lower() == 'em' and nextToken.lower() in (self.article + self.pronoun + self.pronoun_crass):
                contracted = self.mark('n', 'N', currentToken, nextToken, nextToken.lower(), currentRole, return_tokens, return_roles)

            elif currentToken.lower() == 'por' and nextToken in self.article:
                contracted = self.mark('pel', 'Pel', currentToken, nextToken, nextToken.lower(), currentRole, return_tokens, return_roles)

            elif currentToken.lower() == 'a':
                if nextToken.lower() in self.pronoun_crass:
                    contracted = self.mark(u'à', u'À', currentToken, nextToken, nextToken[1:].lower(), currentRole, return_tokens, return_roles)
                elif nextToken.lower() in self.article_masc:
                    contracted = self.mark('a', 'A', currentToken, nextToken, nextToken.lower(), currentRole, return_tokens, return_roles)
                elif nextToken.lower() == 'a':
                    contracted = self.mark(u'à', u'À', currentToken, nextToken, '', currentRole, return_tokens, return_roles)
                elif nextToken.lower() == 'as':
                    contracted = self.mark(u'às', u'Às', currentToken, nextToken, '', currentRole, return_tokens, return_roles)

            if contracted == False:
                return_tokens.append(currentToken)
                return_roles.append(currentRole)
                contracted = False

        return return_tokens, return_roles


if __name__ == '__main__':
    tokens = ['joao', 'de', u'os', 'em', u'estas', 'por', 'a', 'a', 'aquela']
    papeis = ['joao', 'de', u'os', 'em', u'estas', 'por', 'a', 'aquela']

    print ContractionHandler().execute(tokens, tokens)
