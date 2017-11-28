
import numpy as np
from utils.config_loader import readConfig
from corpus import CorpusCreator
from utils import Utils, extractAllTokens
from embeddings import W2VModel, EmbeddingLoader

"""
This file is responsible for creating the expanded corpus from wikipedia sentences.
"""


seed = 27
np.random.seed(seed)


print 'loading configuration'
config, modelConfig = readConfig()
print 'configuration loaded'


predicates = extractAllTokens(config.convertedCorpusDir+'/propbank_full.csv', 'predicate')

w2vFiles = {
    "npzFile":config.embeddingsDir+"/wordEmbeddings.npy",
    "npzModel":config.embeddingsDir+"/wordEmbeddings",
    "vecFile":config.embeddingsDir+"/model.vec",
    "w2idxFile":config.embeddingsDir+"/vocabulary.json"
}

w2v = W2VModel()
w2v.setResources(w2vFiles)
loader = EmbeddingLoader(w2v)
word2idx, idx2word, weights = loader.process()

c = CorpusCreator(80000, predicates, word2idx)
c.scanWikipediaCorpus(config.convertedCorpusDir+'/wiki.csv')


