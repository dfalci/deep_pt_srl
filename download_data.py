from utils import Utils, downloadData
from model.configuration import Config

print 'loading configuration'
config = Config.Instance()
config.prepare(Utils.getWorkingDirectory())

print 'downloading files'
downloadData(config)
print 'done'