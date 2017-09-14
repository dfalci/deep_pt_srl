

from nn_corpus_loader import CorpusConverter

from config import Config
from model_config import ModelConfig
from nn_utils import NNUtils
from function_utils import Utils
from prepare_hybrid_embeddings import getEmbeddings

def contabilizarUnkPredicates(predicates, predicateLoader):
    idx2word = predicateLoader.idx2word
    for i in xrange(0, len(predicates)):

        print idx2word[predicates[i][0][0]]




print 'loading configuration'
config = Config.Instance()
config.prepare(Utils.getWorkingDirectory())

modelConfig = ModelConfig.Instance()
modelConfig.prepare(config.srlConfig+'/srl-config.json')
print 'configuration loaded'



print 'loading word embeddings {}'.format(modelConfig.embeddingType)
sentenceLoader, predicateLoader = getEmbeddings(config, modelConfig.embeddingType)
nnUtils = NNUtils.Instance()
nnUtils.setWordUtils(sentenceLoader.word2idx, sentenceLoader.idx2word)
print 'loaded'


print 'loading corpus'
csvFiles = [config.convertedCorpusDir+'/propbank_training.csv', config.convertedCorpusDir+'/propbank_test.csv']
converter = CorpusConverter(csvFiles, sentenceLoader, predicateLoader)
data = converter.load(config.resourceDir+'/feature_file.npy')
tagMap = converter.tagMap
tagList = converter.tagList
nnUtils.setTagList(tagMap, tagList)
print 'loaded'

print 'preparing data for training'
trainingData = data[0]
testData = data[1]

print testData[1]

contabilizarUnkPredicates(testData[1], predicateLoader)
