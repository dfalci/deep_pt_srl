from .emb_loader import W2VModel, EmbeddingLoader, HybridModel
from utils.token_regex import extractAllTokens
from model.configuration import Config
from model.configuration.model_config import ModelConfig

def prepareEmbeddings(useWiki=False):
    config = Config.Instance()
    modelConfig = ModelConfig.Instance()

    tokens = extractAllTokens(config.convertedCorpusDir+'/propbank_full.csv')
    if useWiki:
        tokens.update(extractAllTokens(config.convertedCorpusDir+'/wiki.csv'))

    print '{} tokens found'.format(len(tokens))

    predicates = extractAllTokens(config.convertedCorpusDir+'/propbank_full.csv', 'predicate')


    w2vFiles = {
        "npzFile":config.embeddingsDir+"/wordEmbeddings.npy",
        "npzModel":config.embeddingsDir+"/wordEmbeddings",
        "vecFile":__getVecFile(config.embeddingsDir, modelConfig.embeddingSize),
        "w2idxFile":config.embeddingsDir+"/vocabulary.json"
    }

    w2v = W2VModel()
    w2v.setResources(w2vFiles)
    loader = EmbeddingLoader(w2v)
    word2idx, idx2word, weights = loader.process()

    if modelConfig.embeddingType == 'w2v':
        return loader, loader

    sentHybridFiles = {
        "npzFile":config.embeddingsDir+"/sent_hybrid.npy",
        "npzModel":config.embeddingsDir+"/sent_hybrid",
        "w2idxFile":config.embeddingsDir+"/sent_hybrid.json"
    }

    sentHybrid = HybridModel()
    sentHybrid.setResources(sentHybridFiles)
    print 'creating sentence corpus'
    sentHybrid.generateCorpus(tokens, weights, word2idx)
    Hloader = EmbeddingLoader(sentHybrid)
    Hword2idx, Hidx2word, Hweights = Hloader.process()


    predHybridFiles = {
        "npzFile":config.embeddingsDir+"/pred_hybrid.npy",
        "npzModel":config.embeddingsDir+"/pred_hybrid",
        "w2idxFile":config.embeddingsDir+"/pred_hybrid.json"
    }

    predHybrid = HybridModel()
    predHybrid.setResources(predHybridFiles)
    print 'creating predicate corpus'
    predHybrid.generateCorpus(predicates, weights, word2idx)
    Ploader = EmbeddingLoader(predHybrid)
    Pword2idx, Pidx2word, Pweights = Ploader.process()


    return Hloader, Ploader


def __getVecFile(embeddingsDir, embeddingSize):
    if embeddingSize == 150:
        return embeddingsDir+"/model.vec"
    elif embeddingSize == 100:
        return embeddingsDir+"/model100.vec"
    else:
        return embeddingsDir+"/model50.vec"


def getEmbeddings():
    config = Config.Instance()
    modelConfig = ModelConfig.Instance()

    if modelConfig.embeddingType == 'w2v':
        w2vFiles = {
            "npzFile":config.embeddingsDir+"/wordEmbeddings.npy",
            "npzModel":config.embeddingsDir+"/wordEmbeddings",
            "vecFile":__getVecFile(config.embeddingsDir, modelConfig.embeddingSize),
            "w2idxFile":config.embeddingsDir+"/vocabulary.json"
        }

        w2v = W2VModel()
        w2v.setResources(w2vFiles)
        loader = EmbeddingLoader(w2v)
        word2idx, idx2word, weights = loader.process()
        return loader, loader
    else:
        sentHybridFiles = {
            "npzFile":config.embeddingsDir+"/sent_hybrid.npy",
            "npzModel":config.embeddingsDir+"/sent_hybrid",
            "w2idxFile":config.embeddingsDir+"/sent_hybrid.json"
        }

        sentHybrid = HybridModel()
        sentHybrid.setResources(sentHybridFiles)
        Hloader = EmbeddingLoader(sentHybrid)
        Hword2idx, Hidx2word, Hweights = Hloader.process()


        predHybridFiles = {
            "npzFile":config.embeddingsDir+"/pred_hybrid.npy",
            "npzModel":config.embeddingsDir+"/pred_hybrid",
            "w2idxFile":config.embeddingsDir+"/pred_hybrid.json"
        }

        predHybrid = HybridModel()
        predHybrid.setResources(predHybridFiles)
        Ploader = EmbeddingLoader(predHybrid)
        Pword2idx, Pidx2word, Pweights = Ploader.process()


        return Hloader, Ploader