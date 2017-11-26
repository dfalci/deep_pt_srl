from utils import downloadData
from utils.config_loader import readConfig

print 'loading configuration'
config, modelConfig = readConfig()

print 'downloading files'
downloadData(config)
print 'done'