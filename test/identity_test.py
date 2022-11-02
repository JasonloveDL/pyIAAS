import unittest

from pyIAAS import *

set_seed(0)
config_file = 'NASConfig.json'

# load configuration file
cfg = Config(config_file)


class MyTestCase(unittest.TestCase):
    def test_identity_dense(self):
        # test dense identity
        for _ in range(10):
            for input_shape in [(4, 32), (16, 4), (32, 128)]:
                dense_identity = DenseModule.identity_module(cfg, 'dense', input_shape)
                input_data = torch.randn((2, *input_shape))
                identity_out = dense_identity(input_data)

                diff = (input_data - identity_out).sum().item()
                self.assertTrue(abs(diff) < 1e-5)  # add assertion here

    def test_identity_conv(self):
        # test dense identity
        for input_shape in [(4, 32), (16, 4), (32, 128)]:
            conv_identity = ConvModule.identity_module(cfg, 'conv', input_shape)
            input_data = torch.randn((2, *input_shape))
            identity_out = conv_identity(input_data)

            diff = (input_data - identity_out).sum().item()

            self.assertTrue(abs(diff) < 1e-5)  # add assertion here

    def test_identity_rnn(self):
        # test dense identity
        for input_shape in [(4, 32), (16, 4), (32, 128)]:
            rnn_identity = RNNModule.identity_module(cfg, 'rnn', input_shape)
            input_data = torch.randn((2, *input_shape))
            input_data = input_data - input_data.min()
            identity_out = rnn_identity(input_data)

            diff = (input_data - identity_out).sum().item()

            self.assertTrue(abs(diff) < 1e-5)  # add assertion here

    def test_identity_lstm(self):
        # test dense identity
        for input_shape in [(4, 32), (16, 4), (32, 128)]:
            lstm_identity = LSTMModule.identity_module(cfg, 'lstm', input_shape)
            input_data = torch.randn((2, *input_shape))
            input_data = input_data - input_data.min()
            identity_out = lstm_identity(input_data)

            diff = (input_data - identity_out).sum().item()

            self.assertTrue(abs(diff) < 1e-5)  # add assertion here


if __name__ == '__main__':
    unittest.main()
