
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
        (config.embeddingsDir+'/model.vec', 'https://www.dropbox.com/s/rjz42q1gjqbjyau/model.vec?dl=0'),
        (config.embeddingsDir+'/model.vec.syn1neg.npy', 'https://www.dropbox.com/s/hjr6otaeshnozxj/model.vec.syn1neg.npy?dl=0'),
        (config.embeddingsDir+'/model.vec.wv.syn0.npy', 'https://www.dropbox.com/s/pmu7m82o5soref5/model.vec.wv.syn0.npy?dl=0'),
        (config.convertedCorpusDir+'/propbank_full.csv'),
        (config.convertedCorpusDir+'/propbank_test.csv'),
        (config.convertedCorpusDir+'/propbank_train.csv'),
        (config.resourceDir+'feature_file.npy')
    )
    for item in files:
        getFile(item[0], item[1])

if __name__ == '__main__':
    getFile('./model.vec', 'https://www.dropbox.com/s/rjz42q1gjqbjyau/model.vec?dl=0')




