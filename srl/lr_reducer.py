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
        self.patience = 4
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




