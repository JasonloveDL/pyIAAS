import unittest

from pyIAAS import *

set_seed(0)
config_file = 'NASConfig.json'
input_file = 'VT_summer.csv'
target_name = 'RT_Demand'
test_ratio = 0.2

# load configuration file
cfg = Config(config_file)
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

devices = []
# devices.append(torch.device('cpu'))
if torch.cuda.is_available():
    devices.append(torch.device('cuda'))


class MyTestCase(unittest.TestCase):
    def test_prune(self):
        skeleton_list = [  # prune structure to test
            ['lstm', 'lstm'],
        ]
        # type_list = ['dense', 'rnn', 'conv','lstm']
        # skeleton_list = list(itertools.product(type_list, type_list))
        # skeleton_list = [list(i) for i in skeleton_list]

        prune_loop_times = 3
        for skeleton in skeleton_list:
            feature_shape = data[0].shape[1:]
            X_train, y_train, X_test, y_test = data
            for device in devices:
                # test prune in different devices
                set_seed(0)
                X_train, y_train, X_test, y_test = \
                    X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)
                model_raw = generate_new_model_config(cfg, feature_shape, 1, skeleton).generate_model()
                model_raw.model_instance.to(device)
                for i in range(prune_loop_times):
                    model_raw.train(X_train, y_train)
                    loss_prev = model_raw.test(X_test, y_test)
                    print('\n')
                    print(f'before pruning: {model_raw}')
                    model_raw = model_raw.prune()
                    loss_after = model_raw.test(X_test, y_test)
                    print(f'after pruning: {model_raw}')
                    print(f'{device} before prune:{loss_prev}\t after prune:{loss_after}')


if __name__ == '__main__':
    MyTestCase().test_prune()
