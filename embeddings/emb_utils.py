from .emb_loader import W2VModel, EmbeddingLoader, HybridModel
from utils.token_regex import extractAllTokens

def prepareEmbeddings(config, type='w2v'):

    tokens = extractAllTokens(config.convertedCorpusDir+'/propbank_full.csv')
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

    if type == 'w2v':
        return loader, loader

    sentHybridFiles = {
        "npzFile":config.embeddingsDir+"/sent_hybrid.npy",
        "npzModel":config.embeddingsDir+"/sent_hybrid",
        "w2idxFile":config.embeddingsDir+"/sent_hybrid.json"
    }

    sentHybrid = HybridModel()
    sentHybrid.setResources(sentHybridFiles)
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
    predHybrid.generateCorpus(predicates, weights, word2idx)
    Ploader = EmbeddingLoader(predHybrid)
    Pword2idx, Pidx2word, Pweights = Ploader.process()


    return Hloader, Ploader


def getEmbeddings(config, type='w2v'):
    if type == 'w2v':
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