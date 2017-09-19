import numpy as np
from utils import ConverterUtils

class Predictor(object):
    """
    Given a sentence, convert it to the format used in our neural network and then uses it
    """

    def __init__(self, nn, tagList, inference=None):
        self.nn = nn
        self.tagList = tagList
        self.inference = inference


    def __predict(self, sent, pred, aux):
        y = self.nn.predict([sent, pred, aux])
        if self.inference != None:
            y, tags = self.globalInference.predict(y[0])
        y = np.argmax(y, axis=1)
        return ConverterUtils.fromIndexToRoles(y, self.tagList)