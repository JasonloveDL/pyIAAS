import itertools
import unittest

from pyIAAS import *

set_seed(0)
config_file = 'NASConfig.json'

# load configuration file
cfg = Config(config_file)


class MyTestCase(unittest.TestCase):
    def test_widen(self):
        X_test = torch.rand(1, 168, 3)
        type_list = ['dense', 'rnn', 'conv', 'lstm']
        skeleton_list = list(itertools.product(type_list, type_list))
        skeleton_list = [list(i) for i in skeleton_list]
        feature_shape = X_test.shape[1:]
        for skeleton in skeleton_list:
            for _ in range(1):
                current_model = generate_new_model_config(cfg, feature_shape, 1, skeleton).generate_model()
                wider_index = 0
                wider_model = current_model.perform_wider_transformation(wider_index)

                current_out = current_model(X_test)
                wider_out = wider_model(X_test)
                diff = (((current_out - wider_out) / current_out).sum() / X_test.shape[0]).item()
                percent_diff = diff / (current_out.mean().item())
                print(f'wider action :{wider_index}')
                print(f'skeleton :{skeleton}')
                print(f'current_model :{current_model}')
                print(f'wider_model :{wider_model}')
                print(f'mean difference :{diff}')
                print(f'mean percent_diff :{percent_diff}')
                self.assertTrue(abs(diff) < 1e-3)  # add assertion here
