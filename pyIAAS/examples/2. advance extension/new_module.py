from pyIAAS.model.module import NasModule
# this is a sample subclassing of NasModule  to
# illustrate how to customize a new module in the pyIAAS package
class NewModule(NasModule):
    @property
    def is_max_level(self):
        # return: True if this module reaches the max width level, False otherwise
        raise NotImplementedError()

    @property
    def next_level(self):
        # return: width of next level
        raise NotImplementedError()

    def init_param(self, input_shape):
        # initialize the parameters of this module
        self.on_param_end(input_shape)
        raise NotImplementedError()

    @staticmethod
    def identity_module(cfg, name, input_shape: tuple):
        # generate an identity mapping module
        raise NotImplementedError()

    def get_module_instance(self):
        # generate a model instance once and use it for the rest procedures
        raise NotImplementedError()

    @property
    def token(self):
        # return: string type token of this module
        raise NotImplementedError()

    def perform_wider_transformation_current(self):
        # generate a new wider module by the wider function-preserving transformation
        # this function is called by layer i and returns the realized random mapping to the IAAS framework for the next layer's wider transformation.
        raise NotImplementedError()

    def perform_wider_transformation_next(self, mapping_g: list, scale_g: list):
        # generate a new wider module by the wider function-preserving transformation
        # this function is called by the layer i + 1
        raise NotImplementedError()