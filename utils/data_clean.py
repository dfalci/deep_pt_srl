
import os
from model.configuration import Config

def __delete(file):
    try:
        os.remove(file)
    except:
        print 'error removing file {}'.format(file)

def deleteTrainingResources():
    __delete(Config.Instance().resourceDir+'/feature_file.npy')
    __delete(Config.Instance().resourceDir+'/embeddings/vocabulary.json')
    __delete(Config.Instance().resourceDir+'/embeddings/pred_hybrid.json')
    __delete(Config.Instance().resourceDir+'/embeddings/pred_hybrid.npy')
    __delete(Config.Instance().resourceDir+'/embeddings/sent_hybrid.json')
    __delete(Config.Instance().resourceDir+'/embeddings/sent_hybrid.npy')
    __delete(Config.Instance().resourceDir+'/embeddings/wordEmbeddings.npy')
