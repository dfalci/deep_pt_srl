
import pandas as pd
from model.configuration import Config
from model.configuration.model_config import ModelConfig
from utils.function_utils import Utils


print 'loading configuration'
config = Config.Instance()
config.prepare(Utils.getWorkingDirectory())

modelConfig = ModelConfig.Instance()
modelConfig.prepare(config.srlConfig+'/srl-config.json')
print 'configuration loaded'

wikiFile = pd.read_csv(config.convertedCorpusDir+'/semi_sup_wiki.csv')
regularTraining = pd.read_csv(config.convertedCorpusDir+'/propbank_training.csv')


print len(wikiFile)


temp = wikiFile[wikiFile['roles'].str.match('B-AM|B-A2|B-A3|B-A4', na=False)]

temp = temp.append(regularTraining, ignore_index=True)

print len(temp)

temp.to_csv(config.convertedCorpusDir+'/semi_sup_wiki_auxiliary_only.csv', index=False)



