
from model.configuration import Config
from model.configuration.model_config import ModelConfig
from utils.function_utils import Utils
from utils import extractFeaturesFromSentence, toNNFormat
from embeddings import getEmbeddings
import pandas as pd

print 'loading configuration'
config = Config.Instance()
config.prepare(Utils.getWorkingDirectory())

modelConfig = ModelConfig.Instance()
modelConfig.prepare(config.srlConfig+'/srl-config.json')
print 'configuration loaded'

sentenceLoader, predicateLoader = getEmbeddings(config, modelConfig.embeddingType)

wikiFile = pd.read_csv(config.convertedCorpusDir+'/wiki.csv')

for i in xrange(0, len(wikiFile)):
    predicate = wikiFile['predicate'][i]
    sentence = wikiFile['sentence'][i]
    convertedSentence, convertedPredicate, allCaps, firstCaps, noCaps, context, distance = extractFeaturesFromSentence(sentence, predicate, sentenceLoader.word2idx, predicateLoader.word2idx)
    inputSentence, inputPredicate, inputAux = toNNFormat(convertedSentence, convertedPredicate, allCaps, firstCaps, noCaps, context, distance)

    print inputSentence.shape, inputPredicate.shape, inputAux.shape
    break
