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


from keras import backend as K
from keras.callbacks import *
import numpy as np


class LrReducer(object):

    def __init__(self, trainingEpochs):
        self.trainingEpochs = trainingEpochs
        self.bestF1 = 0
        self.reductions = 0
        self.bestEpoch = 0
        self.currentEpoch = 0

    def setNetwork(self, nn):
        self.nn = nn

    def registerScore(self, newF1, epoch):
        self.currentEpoch = epoch
        if newF1 > self.bestF1:
            self.bestF1 = newF1
            print 'NEW BEST F1 : {}'.format(self.bestF1)
            return True
        return False


    def onEpochEnd(self, f1, epoch):
        pass

    def setParameters(self, options):
        pass

    def getLearningRate(self):
        return K.get_value(self.nn.optimizer.lr)

    def calculateNewLr(self):
        pass

    def setLearningRate(self, new_lr):
        print 'NEW LEARNING RATE : {}'.format(new_lr)
        K.set_value(self.nn.optimizer.lr, new_lr)


class RateBasedLrReducer(LrReducer):

    def __init__(self, trainingEpochs):
        super(RateBasedLrReducer, self).__init__(trainingEpochs)

    def onEpochEnd(self, f1, epoch):
        self.registerScore(f1, epoch)
        self.calculateNewLr()

    def calculateNewLr(self):
        lr = self.getLearningRate()

        decay = lr * float(self.currentEpoch) / (self.trainingEpochs)
        new_lr = lr * 1/(1 + decay * self.currentEpoch)

        self.setLearningRate(new_lr)

class PatienceBaseLrReducer(LrReducer):

    def __init__(self, trainingEpochs):
        super(PatienceBaseLrReducer, self).__init__(trainingEpochs)
        self.roundsAwaiting = 0
        self.reduceRate = 0.7
        self.patience = 3
        self.maxReductions = 15
        self.reductions = 0


    def onEpochEnd(self, f1, epoch):
        if self.registerScore(f1, epoch):
            self.roundsAwaiting = 0
        else:
            if self.roundsAwaiting > self.patience and self.reductions < self.maxReductions:
                self.calculateNewLr()
                self.roundsAwaiting = 0
            else:
                self.roundsAwaiting += 1
                print 'rounds awaiting : {}'.format(self.roundsAwaiting)

    def calculateNewLr(self):
        lr = self.getLearningRate()
        new_lr = lr * self.reduceRate
        self.setLearningRate(new_lr)
        self.reductions +=1



class CyclicLearningRate(Callback):

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular2', gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLearningRate, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}
        self._reset()


    def _reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            print 'learning rate : {}'.format(self.base_lr)
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


if __name__ == '__main__':
    trainingEpochs = 300
    lr = 0.001

    #decay = 0.001

    temp = []
    for epoch in xrange(1, trainingEpochs):
        decay = lr * (float(epoch) / trainingEpochs)
        lr = lr * 1/(1 + decay * epoch)
        print lr
        temp.append(lr)

    from pylab import *

    plot(temp)

    xlabel('time')
    ylabel('learning rate')
    title('Learning rate decay')
    grid(True)
    show()




