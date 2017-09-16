
import numpy as np
from model.configuration import Config
from model.configuration import ModelConfig
from corpus import CorpusCreator
from utils import Utils, extractAllTokens
from embeddings import W2VModel, EmbeddingLoader


seed = 27
np.random.seed(seed)


print 'loading configuration'
config = Config.Instance()
config.prepare(Utils.getWorkingDirectory())
print 'base directory : {}'.format(config.baseDir)
modelConfig = ModelConfig.Instance()
modelConfig.prepare(config.srlConfig+'/srl-config.json')
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
c.scanWikipediaCorpus(config.corpusDir+'/wiki.csv')


