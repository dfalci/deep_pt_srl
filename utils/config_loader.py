from function_utils import Utils
from model.configuration import Config
from model.configuration.model_config import ModelConfig

def readConfig(file='/srl-config.json'):
    config = Config.Instance()
    config.prepare(Utils.getWorkingDirectory())

    modelConfig = ModelConfig.Instance()
    modelConfig.prepare(config.srlConfig+file)
    return config, modelConfig

