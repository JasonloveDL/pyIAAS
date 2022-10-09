import unittest
import torch
from pyIAAS import *

set_seed(0)
config_file = 'NASConfig.json'
input_file = 'VT_summer.csv'
target_name = 'RT_Demand'
test_ratio = 0.2  # the proportion of the test dataset in the whole dataset. It can be adjusted by users themself for specific tasks

# load configuration file
cfg = Config(config_file)


class MyTestCase(unittest.TestCase):
    def test_add_mask(self):
        # create cache file dir and output file dir
        cache_dir = 'cache'
        os.makedirs(cfg.NASConfig['OUT_DIR'], exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        # process data and store middle file in cache dir
        x, y = get_data(cache_dir, input_file, target_name,
                        cfg.NASConfig.timeLength, cfg.NASConfig.predictLength)

        # preprocess data by splitting train test datasets then convert to torch.Tensor object
        data = train_test_split(x, y, test_ratio)
        data = [torch.tensor(i, dtype=torch.float) for i in data]
        skeleton = ['dense', 'dense', 'dense', 'dense']  # specify network structure
        feature_shape = data[0].shape[1:]
        X_train, y_train, X_test, y_test = data
        model_raw = generate_new_model_config(cfg, feature_shape, 1, skeleton).generate_model()
        model_raw.train(X_train, y_train)




