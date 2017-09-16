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

import json

from utils import Singleton


@Singleton
class Config(object):

    def __init__(self):
        self.config = None

    def __getNestedReference(self, key):
        idx = self.config[key].find('${')
        if idx!=-1:
            idxE = self.config[key].find('}', idx)
            return self.config[key][idx+2:idxE]
        return None


    def prepare(self, baseDir):
        self.baseDir = baseDir
        configFile = self.baseDir+'/config/path.json'
        with open(configFile, 'r') as f:
            self.config = json.loads(f.read())
        f.close()
        for k, v in enumerate(self.config):
            self.__dict__[v] = self.config[v]
        self.__fill()

    def __fill(self):
        for k, v in enumerate(self.config):
            temp = self.__getNestedReference(v)
            if temp!=None:
                self.__dict__[v] = self.__dict__[v].replace('${'+temp+'}', self.__dict__[temp])



if __name__ == '__main__':
    config = Config.Instance()
    config.prepare('../.')
    print Config.Instance().resourceDir
