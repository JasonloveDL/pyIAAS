import json
import os.path

from .object_dict import ObjectDict
from ..model.module import DenseModule, RNNModule, LSTMModule, ConvModule

# add default modules
_defaultCls = {
    'dense': DenseModule,
    'rnn': RNNModule,
    'lstm': LSTMModule,
    'conv': ConvModule
}


class Config:
    """
    Configuration file of the whole training process
    """

    def __init__(self, nasConfig_file):
        super().__init__()
        with open(nasConfig_file) as f:
            self.NASConfig: ObjectDict = ObjectDict(json.load(f))
        self.modulesConfig: dict = self.NASConfig['Modules']
        self.NASConfig['editable'] = []
        self.modulesCls: dict = {}
        for k in self.modulesConfig.keys():
            if self.modulesConfig[k]['editable']:
                self.NASConfig['editable'].append(k)
            if k in _defaultCls:
                self.modulesCls[k] = _defaultCls[k]
        self.modulesList: list = list(self.modulesConfig.keys())

    @property
    def LOG_FILE(self):
        """
        :return: log file path
        """
        return os.path.join(self.NASConfig.OUT_DIR, 'log.txt')

    @property
    def SQL_FILE(self):
        """
        :return: log file path
        """
        return os.path.join(self.NASConfig.OUT_DIR, 'model.db')

    def register_module(self, name, module_cls):
        """
        register module to global configuration
        :param name: name of this module
        :param module_cls: module class
        :return:
        """
        self.modulesCls[name] = module_cls
