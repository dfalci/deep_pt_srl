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

class LrReducer(object):

    def __init__(self, patience=2, reduceRate=0.5, maxReductions=20):
        self.patience =0
        self.roundsAwaiting = 0
        self.bestF1 = 0
        self.reduceRate =0
        self.maxReductions = maxReductions
        self.reductions = 0


    def onEpochEnd(self, nn, f1Score):
        if f1Score > self.bestF1:
            self.bestF1 = f1Score
            self.roundsAwaiting = 0
            print 'current best f1 : {}'.format(self.bestF1)
        else:
            if self.roundsAwaiting >= self.patience:
                if (self.reductions > self.maxReductions):
                    lr = nn.optimizer.lr_get_value()
                    print 'current learning rate : {}'.format(lr)
                    nn.optimizer.lr.set_value(lr * self.reduceRate)
                self.reductions+=1
            else:
                self.roundsAwaiting+=1
                print 'incremented rounds awaiting {}'.format(self.roundsAwaiting)




