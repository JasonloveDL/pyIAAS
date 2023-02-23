import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import seaborn as sns
import torch
from torchinfo import torchinfo

from structures import lstm_model, cnn_model
import pyIAAS.model

font_size = 18


def setup_plt():
    sns.set_theme()
    sns.set_style("white")
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.rcParams['font.size'] = font_size
    plt.rcParams['xtick.labelsize'] = font_size
    plt.rcParams['ytick.labelsize'] = font_size
    plt.rcParams["font.family"] = "Times New Roman"


def load_data():
    data = pd.DataFrame()
    set_real = True
    for k in names:
        d = pd.read_csv(names[k])
        if set_real:
            data.loc[:, 'Real'] = d.loc[:, 'truth']
            set_real = False
        data.loc[:, k] = d.loc[:, 'pred']
    return data


def evaluate(data):
    result = pd.DataFrame()
    for k in names:
        result.loc['RMSE', k] = RMSE(data.loc[:, k], data.loc[:, 'Real'])
        result.loc['MAE', k] = MAE(data.loc[:, k], data.loc[:, 'Real'])
        result.loc['MAPE', k] = MAEP(data.loc[:, k], data.loc[:, 'Real'])
    # get parameter count
    lstm = lstm_model()
    cnn = cnn_model()
    ours = torch.load(r'../out_dir_summer/best/NasModel.pth', 'cpu')
    result.loc['Parameter', 'LSTM'] = torchinfo.summary(lstm).total_params
    result.loc['Parameter', 'CNN-LSTM'] = torchinfo.summary(cnn).total_params
    result.loc['Parameter', 'pyIAAS'] = torchinfo.summary(ours).total_params
    return result


def RMSE(pred, truth):
    return np.sqrt(np.mean(np.square(pred - truth)))


def MAE(pred, truth):
    return np.mean(np.abs(pred - truth))


def MAEP(pred, truth):
    index = truth != 0
    return np.mean(np.abs((pred[index] - truth[index]) / truth[index]))


def plot_data(data, eval):
    keys = list(names.keys())
    for i in range(4):
        plt.subplot(4, 2, i * 2 + 1)
        sns.lineplot(data=data.loc[:, ['Real', keys[i]]])
        plt.ylabel('Load (MW)', fontsize=font_size)
    plt.xlabel('Time periods', fontsize=font_size)

    keys = eval.index
    length = 3
    for i in range(length):
        plt.subplot(length, 2, i * 2 + 2)
        sns.barplot(x=eval.loc[keys[i], :].index, y=eval.loc[keys[i], :].tolist())
        plt.title(keys[i], fontsize=font_size + 2)
    plt.subplots_adjust(top=0.95,
                        bottom=0.1,
                        left=0.05,
                        right=0.95,
                        hspace=0.345,
                        wspace=0.14)
    plt.show()


if __name__ == '__main__':
    names = {
        'pyIAAS': 'best_pred.csv',
        'CNN-LSTM': 'CNN-LSTM_VT_summer.csv',
        'LSTM': 'LSTM_VT_summer.csv',
        'RF': 'RF_VT_summer.csv',
    }
    setup_plt()
    data = load_data()
    eval = evaluate(data)
    eval.to_csv('eval.csv')
    plot_data(data, eval)
