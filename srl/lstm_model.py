import json
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.merge import Concatenate
from keras.engine import Input



class LSTMModel(object):

    def __init__(self, configFile):
        """
        Creates a neural network model
        :param configFile:
        :return:
        """
        self.configFile = configFile
        self.config = None

    def __loadConfig(self):
        with open(self.configFile, 'r') as f:
            self.config = json.loads(f.read())

    def create(self, tokenMatrix, predMatrix):
        self.__loadConfig()

        EMBED_SIZE = self.config['embeddingSize']
        LSTM_CELLS = self.config['lstmCells']
        RECURRENT_ACTIVATION = self.config["recurrentActivation"]
        ACTIVATION = self.config["activation"]
        DROPOUT = self.config["dropout"]
        RECURRENT_DROPOUT = self.config["recurrentDropout"]
        CLASSES = self.config["classes"]
        OPTIMIZER = self.config["optimizer"]
        LOSS_FUNCTION = self.config["lossFunction"]

        nn = None

        inputSentence = Input(shape=(None,), dtype='int32', name='inputSentence')
        inputAux = Input(shape=(None,), batch_shape=(None, None, 5), name='InputAux')
        inputPredicate = Input(shape=(None,), dtype='int32', name='InputPredicate')

        embedding = Embedding(tokenMatrix.shape[0], EMBED_SIZE, weights=[tokenMatrix], trainable=False, name='Embedding')(inputSentence)
        embeddingPredicate = Embedding(predMatrix.shape[1], EMBED_SIZE,  weights=[predMatrix], trainable=False, name='EmbeddingPred')(inputPredicate)
        conc = Concatenate(axis=-1, name='concatenate')([embedding, embeddingPredicate, inputAux])

        bi = Bidirectional(LSTM(LSTM_CELLS, activation=ACTIVATION, recurrent_activation=RECURRENT_ACTIVATION, recurrent_dropout=RECURRENT_DROPOUT, dropout=DROPOUT, return_sequences=True))(conc)

        bi = Bidirectional(LSTM(LSTM_CELLS, activation=ACTIVATION, recurrent_activation=RECURRENT_ACTIVATION, recurrent_dropout=RECURRENT_DROPOUT, dropout=DROPOUT, return_sequences=True))(bi)

        bi = Bidirectional(LSTM(LSTM_CELLS, activation=ACTIVATION, recurrent_activation=RECURRENT_ACTIVATION, recurrent_dropout=RECURRENT_DROPOUT, dropout=DROPOUT, return_sequences=True))(bi)

        bi = Bidirectional(LSTM(LSTM_CELLS, activation=ACTIVATION, recurrent_activation=RECURRENT_ACTIVATION, recurrent_dropout=RECURRENT_DROPOUT, dropout=DROPOUT, return_sequences=True))(bi)

        output = TimeDistributed(Dense(units=CLASSES, activation='softmax'), name='output')(bi)

        nn = Model(inputs=[inputSentence, inputPredicate, inputAux], outputs=[output])

        nn.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
        return nn


if __name__ == '__main__':
    pass