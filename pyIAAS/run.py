import pickle

from pyIAAS import *


def set_seed(seed):
    """
    set random seed.
    :param seed: seed number, int type
    """
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        print('Set seed for torch fail', e)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


def save_rng_states(cfg, logger, path=None):
    cpu_state = torch.get_rng_state()
    gpu_states = torch.cuda.get_rng_state_all()
    path = os.path.join(cfg.NASConfig['OUT_DIR'], 'rng_state.pkl') if path is None else path
    with open(path, 'wb') as f:
        pickle.dump([cpu_state, gpu_states], f)
        # logger.critical(f'save cpu and gpu rng state')


def try_load_rng_state(cfg, logger, path=None):
    path = os.path.join(cfg.NASConfig['OUT_DIR'], 'rng_state.pkl') if path is None else path
    if not os.path.exists(path):
        return None
    with open(path,'rb') as f:
        cpu_state, gpu_states = pickle.load(f)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(gpu_states)
        logger.critical(f'load cpu and gpu rng state')



def run_search(config, input_file, target_name, test_ratio):
    """
    Start point of pyIAAS framework, this function will process and read data
    create output directory and trigger the RL loop to search in the network space
    :param test_ratio: test data ration, float value in range (0,1)
    :param config: configuration file path(can be either absolute or relative to working directory) or Config object
    :param input_file: input csv file, all the column should be features and target name should be included in one feature. All data should be float values
    :param target_name: target value to predict
    """

    # check file existence
    if not os.path.exists(input_file):
        raise RuntimeError(f'input data file {input_file} does not exist')

    # load configuration file
    if isinstance(config, str):
        if not os.path.exists(config):
            raise RuntimeError(f'configuration file {config} does not exist')
        cfg = Config(config)
    else:
        cfg = config

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
    logger_ = get_logger(f'main loop', cfg.LOG_FILE)

    search_net(cfg, data, logger_)


def search_net(cfg, data, logger_):
    # start RL search loop
    env_ = NasEnv.try_load(cfg, logger_)
    if env_ is None:
        env_ = NasEnv(cfg, cfg.NASConfig['NetPoolSize'], data)
        states = env_.reset()
    else:
        states = env_.get_state()
    agent_ = Agent.try_load(cfg, logger_)
    if agent_ is None:
        agent_ = Agent(cfg, 16, 50, cfg.NASConfig['MaxLayers'])
    replay_memory = ReplayMemory()
    replay_memory.load_memories(cfg.NASConfig['OUT_DIR'])
    try_load_rng_state(cfg, logger_)
    st = time.time()
    for i in range(cfg.NASConfig['EPISODE']):
        # try:
        agent_.save()
        env_.save()
        save_rng_states(cfg, logger_)
        action = agent_.get_action(states)
        states, in_pool_trajectory, finished_trajectory = env_.step(action)
        replay_memory.record_trajectory(in_pool_trajectory, finished_trajectory)
        replay_memory.save_memories(cfg.NASConfig['OUT_DIR'])
        agent_.update(replay_memory)
        logger_.critical(
            f'episode {i} finish,\tpool {len(env_.net_pool)},\tperformance:{env_.performance()}\ttop performance:{env_.top_performance()}')
        # except Exception as e:
        #     logger_.fatal(f'error {e}')
    logger_.critical(
        f'Search episode: {cfg.NASConfig["EPISODE"]}\t Best performance: {env_.top_performance()}\t Search time :{time.time() - st:.2f} seconds')


def run_predict(config_file, input_file, target_name, output_dir, prediction_file):
    """
    do the prediction task on given input file and save result to prediction_file
    :param config_file: configuration file path, can be either absolute or relative to working directory
    :param input_file: input csv file, all the column should be features and target name should be included in one feature. All data should be float values
    :param target_name: target value to predict
    :param output_dir: output directory of previous search result
    :param prediction_file: file to save prediction result
    """
    # check file existence
    if not os.path.exists(config_file):
        raise RuntimeError(f'configuration file {config_file} does not exist')
    if not os.path.exists(input_file):
        raise RuntimeError(f'configuration file {input_file} does not exist')

    # load configuration file
    cfg = Config(config_file)

    model_path = os.path.join(output_dir, 'best', 'NasModel.pth')

    # previous searched model should exist
    if not os.path.exists(model_path):
        raise RuntimeError(f'output dir {cfg.NASConfig["OUT_DIR"]} do not contain previous '
                           f'search result: {model_path}')

    # create cache file dir and output file dir
    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)

    # process data and store middle file in cache dir
    x = get_predict_data(input_file, target_name,
                         cfg.NASConfig.timeLength)

    # preprocess data by splitting train test datasets then convert to torch.Tensor object
    x = torch.tensor(x, dtype=torch.float)
    logger_ = get_logger(f'predict', cfg.LOG_FILE)

    # load model
    model_ = torch.load(model_path)

    # transfer to same device
    if cfg.NASConfig.GPU:
        x = x.cuda()
        model_ = model_.cuda()
    else:
        model_ = model_.cpu()

    with torch.no_grad():
        output = model_(x)
        output = pd.DataFrame({target_name: output.view(-1).cpu().numpy()})
        output.to_csv(prediction_file)
        print('prediction result'.center(50, '='))
        print(output)
        print(''.center(50, '='))
        logger_.info(f'save result to file: {prediction_file}')
