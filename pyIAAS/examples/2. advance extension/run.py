import os.path

from data_process import get_data_wind_power
from pyIAAS import *
from new_module import NewModule


def train_wind_power(data, season, place, cfg):
    origin_dir = cfg.NASConfig['OUT_DIR']
    NASConfig = cfg.NASConfig
    EPISODE = NASConfig['EPISODE']
    IterationEachTime = NASConfig['IterationEachTime']

    # name the output dir by the training details
    cfg.NASConfig['OUT_DIR'] = os.path.join(NASConfig['OUT_DIR'], f'{place}_episode-{EPISODE}-{IterationEachTime}-{season}')
    if not os.path.exists(NASConfig['OUT_DIR']):
        os.makedirs(NASConfig['OUT_DIR'])
    logger_ = get_logger(f'main loop', cfg.LOG_FILE)

    # create training suite
    env_ = NasEnv(cfg, cfg.NASConfig['NetPoolSize'], data)
    agent_ = Agent(cfg, 16, 50, NASConfig['MaxLayers'])
    net_pool = env_.reset()

    # start RL loop
    for i in range(NASConfig['EPISODE']):
        action = agent_.get_action(net_pool)
        net_pool, reward, done, info = env_.step(action)
        agent_.update(reward, action, net_pool)
        env_.render()
        logger_.fatal(f'episode {i} finish,\tpool {len(env_.net_pool)},\tperformance:{env_.performance()}\ttop performance:{env_.top_performance()}')
    cfg.NASConfig['OUT_DIR'] = origin_dir


def main():
    set_seed(10086)
    place = 'WF1'
    cfg = Config('NASConfig.json')

    # register a new module to the global configuration
    cfg.register_module('new_module', NewModule)
    data_root = 'data'
    spring, summer, autumn, winter = get_data_wind_power(data_root, place, cfg.NASConfig.timeLength)

    # train on four season of this dataset
    train_wind_power(spring, 'spring', place, cfg)
    train_wind_power(summer, 'summer', place, cfg)
    train_wind_power(autumn, 'autumn', place, cfg)
    train_wind_power(winter, 'winter', place, cfg)


if __name__ == '__main__':
    main()
