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

from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.merge import Concatenate
from keras.engine import Input
from model.persistence import ModelPersistence
from keras import backend as K



class LSTMModel(object):

    def __init__(self, modelConfig):
        """
        Creates a neural network model
        :param configFile:
        :return:
        """
        self.config = modelConfig
        self.modelPersistence = ModelPersistence()


    def load(self, modelFile, weightFile, learningRate=None):
        nn = self.modelPersistence.load(modelFile, weightFile)
        nn.compile(optimizer=self.config.optimizer, loss=self.config.lossFunction, metrics=['accuracy'])
        if learningRate != None:
            K.set_value(nn.optimizer.lr, learningRate)
        return nn


    def create(self, tokenMatrix, predMatrix):

        nn = None

        inputSentence = Input(shape=(None,), dtype='int32', name='InputSentence')
        inputAux = Input(shape=(None,), batch_shape=(None, None, 5), name='InputAux')
        inputPredicate = Input(shape=(None,), dtype='int32', name='InputPredicate')

        embedding = Embedding(tokenMatrix.shape[0], self.config.embeddingSize, weights=[tokenMatrix], trainable=self.config.trainableEmbeddings, name='Embedding')(inputSentence)
        embeddingPredicate = Embedding(predMatrix.shape[0], self.config.embeddingSize,  weights=[predMatrix], trainable=self.config.trainableEmbeddings, name='EmbeddingPred')(inputPredicate)

        conc = Concatenate(axis=-1, name='concatenate')([embedding, embeddingPredicate, inputAux])

        bi = Bidirectional(LSTM(self.config.lstmCells, activation=self.config.activation, recurrent_activation=self.config.recurrentActivation, recurrent_dropout=self.config.recurrentDropout, dropout=self.config.dropout, return_sequences=True))(conc)
        bi = Dropout(self.config.dropout)(bi)

        bi = Bidirectional(LSTM(self.config.lstmCells, activation=self.config.activation, recurrent_activation=self.config.recurrentActivation, recurrent_dropout=self.config.recurrentDropout, dropout=self.config.dropout, return_sequences=True))(bi)
        bi = Dropout(self.config.dropout)(bi)

        bi = Bidirectional(LSTM(self.config.lstmCells, activation=self.config.activation, recurrent_activation=self.config.recurrentActivation, recurrent_dropout=self.config.recurrentDropout, dropout=self.config.dropout, return_sequences=True))(bi)
        bi = Dropout(self.config.dropout)(bi)

        bi = Bidirectional(LSTM(self.config.lstmCells, activation=self.config.activation, recurrent_activation=self.config.recurrentActivation, recurrent_dropout=self.config.recurrentDropout, dropout=self.config.dropout, return_sequences=True))(bi)
        bi = Dropout(self.config.dropout)(bi)

        output = TimeDistributed(Dense(units=self.config.classes, activation='softmax'), name='output')(bi)

        nn = Model(inputs=[inputSentence, inputPredicate, inputAux], outputs=[output])

        nn.compile(optimizer=self.config.optimizer, loss=self.config.lossFunction, metrics=['accuracy'])
        return nn

