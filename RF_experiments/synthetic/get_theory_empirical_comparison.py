import matplotlib
import matplotlib.pyplot as plt
font = {
        'size'   : 25}
matplotlib.rc('font', **font)
figsize = (8, 8)
import numpy as np
import os
from scipy.interpolate import interp1d
import pandas as pd

def read_dat(dat_file):
    with open(dat_file, 'r') as f:
        content = [line.strip().split() for line in f.readlines()]
        content = [[float(x) for x in line] for line in content]
    return np.array(content)
def read_files(filename, save_dir, index=[1,2]):
    data = []
    for i, id in enumerate(index):
        filepath = os.path.join(save_dir, str(id), filename)
        assert os.path.exists(filepath), filepath
        data.append(read_dat(filepath))
    return data
def get_new_y(x, y, new_x):
    f = interp1d(x, y)
    return f(new_x)
def plot_test_bias_var_comparison(theory_dir, empirical_dir, N_D, feature_dim,
    test_error_file = 'totK.dat', bias_file='biasK.dat', var_files = ['varDataK.dat', 'varInK.dat', 'varNoiseK.dat'],
    xmin=-2.0, xmax=1.0, ymin=0, ymax=2, figsize = (36, 8), index = [1,2]):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    df1 = pd.read_csv(os.path.join(empirical_dir, 'singleNN_output.csv'))
    df2 = pd.read_csv(os.path.join(empirical_dir, 'ensembleNNK=2_output.csv'))
    df1 = df1[df1['train_size']/feature_dim==N_D]
    df2 = df2[df2['train_size']/feature_dim==N_D]
    dfs_list, Ks_list = [df1, df2], [1, 2]
    for cur_df, K in zip(dfs_list, Ks_list):
        test_loss = cur_df['test_loss']
        bias2 = cur_df['bias2']
        var = cur_df['variance']
        P_N = cur_df['hidden_size']/cur_df['train_size']
        P_N = np.log10(P_N)
        axes[0].scatter(P_N, test_loss, label='K={} (empirical)'.format(K), marker='*')
        axes[1].scatter(P_N, bias2, label='K={} (empirical)'.format(K), marker='*')
        axes[2].scatter(P_N, var, label='K={} (empirical)'.format(K), marker='*')

    # plot test error
    xs = np.linspace(xmin, xmax, 100)
    data = read_files(test_error_file,theory_dir, index=index)
    for i in index[1:]:
        new_y = get_new_y(data[0][:,0], data[0][:,1], xs)
        axes[0].plot(xs, new_y, '-', label='K=1 (theoretical)')
        new_y = get_new_y(data[i-1][:,0], data[i-1][:,1], xs)
        axes[0].plot(xs, new_y, '-', label='K={} (theoretical)'.format(i))

        axes[0].set_xlim([xmin,xmax])
        axes[0].set_ylim([ymin,ymax])
        #axes[0].set_ylabel("Generalization Error")
        axes[0].set_xlabel(r'$Log_{10}[\frac{P}{N}]$')
        
    data = read_files(bias_file,theory_dir, index=index)
    for i in index[1:]:
        new_y = get_new_y(data[0][:,0], data[0][:,1], xs)
        axes[1].plot(xs, new_y, '-', label='K=1 (theoretical)')
        new_y = get_new_y(data[i-1][:,0], data[i-1][:,1], xs)
        axes[1].plot(xs, new_y, '-', label='K={} (theoretical)'.format(i))
        axes[1].set_xlim([xmin,xmax])
        axes[1].set_ylim([ymin,ymax])
        #axes[1].set_ylabel("Bias Square")
        axes[1].set_xlabel(r'$Log_{10}[\frac{P}{N}]$')
    
    all_data = [read_files(filename, theory_dir, index=index) for filename in var_files]

    for i in index[1:]:
        x1, y1 = [np.array(data[0][:,0]) for data in all_data], [np.array(data[0][:,1]) for data in all_data]
        x1, y1 = np.mean(np.array(x1), axis=0), np.sum(np.array(y1), axis=0)
        new_y = get_new_y(x1, y1, xs)
        axes[2].plot(xs, new_y, '-', label='K=1 (theoretical)')
        x2, y2 = [np.array(data[i-1][:,0]) for data in all_data], [np.array(data[i-1][:,1]) for data in all_data]
        x2, y2 = np.mean(np.array(x2), axis=0), np.sum(np.array(y2), axis=0)
        new_y = get_new_y(x2, y2, xs)
        axes[2].plot(xs, new_y, '-', label='K={} (theoretical)'.format(i))

        axes[2].set_xlim([xmin,xmax])
        axes[2].set_ylim([ymin,ymax])
        #axes[2].set_ylabel("Variance")
        axes[2].set_xlabel(r'$Log_{10}[\frac{P}{N}]$')

    plt.legend(fontsize= 'x-small')
    plt.tight_layout()
    plt.subplots_adjust(left = 0.06, wspace = 0.25)
    axes[0].set_title("Test Error")
    axes[1].set_title(r"$Bias^{2}$")
    axes[2].set_title("Variance")
    plt.show()
def main():
    # plot theoretical results and synthetic dataset results when lambda=0.01, N/D=1
    # 
    theory_dir = '/media/Samsung/MemorySplitAdvantage/RF_theory/ResultsSingle/lambda=0.01/ND=1'
    empirical_dir = 'synthetic_coef_0.01'
    N_D = 1 # 10 is too large for empirical experiments
    feature_dim = 400 # for empirical results
    ymin, ymax= 0, 4
    figsize = (20, 6)
    plot_test_bias_var_comparison(theory_dir, empirical_dir, 
        N_D=N_D, feature_dim=feature_dim, figsize=figsize,
        ymin = ymin, ymax=ymax)

if __name__ == '__main__':
    main()