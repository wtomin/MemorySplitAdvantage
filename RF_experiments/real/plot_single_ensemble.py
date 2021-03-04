import matplotlib
import matplotlib.pyplot as plt
font = {
        'size'   : 25}
matplotlib.rc('font', **font)
figsize = (32, 6)
import numpy as np
import os
from scipy.interpolate import interp1d
import pandas as pd

marker_dict = {10: 'd', 100: 'x', 784:'+'}
def plot_single_vs_ensemble(dfs_list, Ks_list, N_D, feature_dim, 
    ymin=0, ymax=1.0):
    assert len(dfs_list) == len(Ks_list)
    fig1, axes1 = plt.subplots(1, 3, figsize=figsize)
    dfs_list = [df[df['train_size']/feature_dim==N_D] for df in dfs_list]
    for cur_df, K in zip(dfs_list, Ks_list):
        if 'batch_size' in cur_df.keys():
            batch_size = np.unique(cur_df['batch_size'].astype("int32"))
            labels = ["K={}(Bs={})".format(K, bs) for bs in batch_size]
            cur_df = [cur_df[cur_df['batch_size']==bs] for bs in batch_size]
            markers = [marker_dict[bs] for bs in batch_size]
        else:
            markers = ['*']
            labels = ['K={}'.format(K)]
            cur_df = [cur_df]
        for df, marker, label in zip(cur_df, markers, labels):
            test_loss = df['test_loss']
            bias2 = df['bias2']
            var = df['variance']
            P_N = df['hidden_size']/df['train_size']
            P_N = np.log10(P_N)
            color = 'tab:blue' if K==1 else 'tab:orange'

            axes1[0].scatter(P_N, test_loss, label=label, marker=marker, color=color)
            axes1[1].scatter(P_N, bias2, label=label, marker=marker, color=color)
            axes1[2].scatter(P_N, var, label=label, marker=marker, color=color)
        
    axes1[0].set_xlabel(r'$Log_{10}[\frac{P}{N}]$')
    axes1[0].set_xticks([-2, -1, 0, 1])
    axes1[0].set_title("Test Loss")
    axes1[0].set_ylim(ymin, ymax)
    
    axes1[1].set_xlabel(r'$Log_{10}[\frac{P}{N}]$')
    axes1[1].set_title(r"$Bias^{2}$")
    axes1[1].set_xticks([-2, -1, 0, 1])
    axes1[1].set_ylim(ymin, ymax)
    
    axes1[2].set_xlabel(r'$Log_{10}[\frac{P}{N}]$')
    axes1[2].set_title("Variance")
    axes1[2].set_xticks([-2, -1, 0, 1])
    axes1[2].set_ylim(ymin, ymax)
    #fig1.suptitle("Bias-Variance Decomposition (N/D={:.2f})".format(N_D))
    plt.legend(fontsize='x-small')
    plt.show()

if __name__ == '__main__':
    df1 = pd.read_csv("mnist_coef_0.01/singleNN_output.csv")
    df2 = pd.read_csv("mnist_coef_0.01/ensembleNNK=2_output.csv")
    df3 = pd.read_csv("mnist_SGD_legacy_NN/num_iters_5000/singleNN_output.csv")
    plot_single_vs_ensemble([df1, df2, df3], [1, 2, 1], N_D=1, feature_dim=784,)