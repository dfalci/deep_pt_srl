# -*- coding: utf-8; -*-
from distutils.core import setup

setup(
        name='deep_pt_srl',
        version='0.1',
        packages=['srl', 'srl.model', 'srl.parser', 'srl.batcher', 'srl.inference', 'srl.evaluation', 'utils',
                  'embeddings'],
        url='www.danielfalci.com.br',
        license='BSD',
        author='Daniel Henrique Mourão Falci',
        author_email='danielfalci@gmail.com',
        description=''
)
