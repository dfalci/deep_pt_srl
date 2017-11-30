
import os
from model.configuration import Config

def __delete(file):
    try:
        os.remove(file)
    except:
        print 'error removing file {}'.format(file)

def deleteTrainingResources(k_folds=20):
    __delete(Config.Instance().resourceDir+'/feature_file.npy')
    for i in xrange(1, k_folds+1):
        __delete(Config.Instance().resourceDir+'/feature_file_'+str(i)+'.npy')
    __delete(Config.Instance().resourceDir+'/embeddings/vocabulary.json')
    __delete(Config.Instance().resourceDir+'/embeddings/pred_hybrid.json')
    __delete(Config.Instance().resourceDir+'/embeddings/pred_hybrid.npy')
    __delete(Config.Instance().resourceDir+'/embeddings/sent_hybrid.json')
    __delete(Config.Instance().resourceDir+'/embeddings/sent_hybrid.npy')
    __delete(Config.Instance().resourceDir+'/embeddings/wordEmbeddings.npy')
