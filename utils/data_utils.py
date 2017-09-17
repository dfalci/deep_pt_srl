
import os

import urllib2
import urlparse

def getFile(filename, origin):
    """
    Download the file with a given filename from an origin
    """
    u = urllib2.urlopen(origin)

    scheme, netloc, path, query, fragment = urlparse.urlsplit(origin)

    if os.path.exists(filename):
        print 'file already exists {}'.format(filename)
        return False

    with open(filename, 'wb') as f:

        print "Downloading file from : {0} to : {1}".format(origin, filename)

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)
    f.close()
    print 'Dowload completed'

    return True


def downloadData(config):
    """
    Download all data needed
    :param config:
    :return:
    """
    files = (
        (config.embeddingsDir+'/model.vec', 'https://dl.dropboxusercontent.com/s/825rx66x2hftiil/model.vec?dl=0'),
        (config.embeddingsDir+'/model.vec.syn1neg.npy', 'https://dl.dropboxusercontent.com/s/s7gx06xxcu2ewgv/model.vec.syn1neg.npy?dl=0'),
        (config.embeddingsDir+'/model.vec.wv.syn0.npy', 'https://dl.dropboxusercontent.com/s/dwu790boj8s870w/model.vec.wv.syn0.npy?dl=0'),
        (config.embeddingsDir+'/vocabulary.json', 'https://dl.dropboxusercontent.com/s/sy8r04v57yz38yq/vocabulary.json?dl=0'),
        (config.embeddingsDir+'/wordEmbeddings.npy', 'https://dl.dropboxusercontent.com/s/qy3t832mcem202q/wordEmbeddings.npy?dl=0'),
        (config.embeddingsDir+'/sent_hybrid.json', 'https://dl.dropboxusercontent.com/s/7oevekbmikmav41/sent_hybrid.json?dl=0'),
        (config.embeddingsDir+'/sent_hybrid.npy', 'https://dl.dropboxusercontent.com/s/3bm1e7wp3klunqb/sent_hybrid.npy?dl=0'),
        (config.embeddingsDir+'/pred_hybrid.json', 'https://dl.dropboxusercontent.com/s/s505ab48vufjpo2/pred_hybrid.json?dl=0'),
        (config.embeddingsDir+'/pred_hybrid.npy', 'https://dl.dropboxusercontent.com/s/1zbm8k47lqrjkxi/pred_hybrid.npy?dl=0'),

        (config.convertedCorpusDir+'/propbank_full.csv', 'https://dl.dropboxusercontent.com/s/ac8og8tef9zal9x/propbank_full.csv?dl=0'),
        (config.convertedCorpusDir+'/propbank_test.csv', 'https://dl.dropboxusercontent.com/s/e4gzfofarfeggwp/propbank_test.csv?dl=0'),
        (config.convertedCorpusDir+'/propbank_train.csv', 'https://dl.dropboxusercontent.com/s/0g0kela0ggiwmwr/propbank_training.csv?dl=0'),
        (config.convertedCorpusDir+'/wiki.csv', 'https://dl.dropboxusercontent.com/s/813n7v6fzjh700j/wiki.csv?dl=0'),
        (config.resourceDir+'/feature_file.npy', 'https://dl.dropboxusercontent.com/s/h7755you89nl34x/feature_file.npy?dl=0'),
        (config.resourceDir+'/wiki_00.bz2', 'https://dl.dropboxusercontent.com/s/poxln5en8nz6wj9/wiki_00.bz2?dl=0')


    )
    for item in files:
        getFile(item[0], item[1])

if __name__ == '__main__':
    getFile('./model.vec', 'https://www.dropbox.com/s/rjz42q1gjqbjyau/model.vec?dl=0')




