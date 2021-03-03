import matplotlib
import matplotlib.pyplot as plt
font = {
        'size'   : 25}
matplotlib.rc('font', **font)
figsize = (12,8)
import numpy as np
import os
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

def plot_test_bias_var(test_error_file, bias_file, var_files, 
	xmin=-2.0, xmax=2.0, ymin=0, ymax=2, figsize = (24, 8), 
	index = [1,2], fill=True, save_dir = '',
	title=True, xlabel=True):
	fig, axes = plt.subplots(1, 3, figsize=figsize)
	# plot test error
	data = read_files(test_error_file,save_dir, index=index)
	for i in index[1:]:
		axes[0].plot(data[0][:,0], data[0][:,1], '-', label='K=1')
		axes[0].plot(data[i-1][:,0], data[i-1][:,1], '-', label='K={}'.format(i))
		min_len = min(len(data[i-1][:,0]), len(data[0][:,0]))
		if fill:
			axes[0].fill_between(data[0][:,0][:min_len], data[0][:,1][:min_len], data[i-1][:,1][:min_len], color = 'mistyrose')
		axes[0].set_xlim([xmin,xmax])
		axes[0].set_ylim([ymin,ymax])
		if title:
			axes[0].set_title("Test Error")
		if xlabel:
			axes[0].set_xlabel(r'$Log_{10}[P/N]$')
		
	data = read_files(bias_file,save_dir, index=index)
	for i in index[1:]:
		axes[1].plot(data[0][:,0], data[0][:,1], '-', label='K=1')
		axes[1].plot(data[i-1][:,0], data[i-1][:,1], '-', label='K={}'.format(i))
		min_len = min(len(data[i-1][:,0]), len(data[0][:,0]))
		if fill:
			axes[1].fill_between(data[0][:,0][:min_len], data[0][:,1][:min_len], data[i-1][:,1][:min_len], color = 'lightyellow')
		axes[1].set_xlim([xmin,xmax])
		axes[1].set_ylim([ymin,ymax])
		if title:
			axes[1].set_title(r"$Bias^{2}$")
		if xlabel:
			axes[1].set_xlabel(r'$Log_{10}[P/N]$')
	
	all_data = [read_files(filename, save_dir, index=index) for filename in var_files]

	for i in index[1:]:

		x1, y1 = [np.array(data[0][:,0]) for data in all_data], [np.array(data[0][:,1]) for data in all_data]
		x1, y1 = np.mean(np.array(x1), axis=0), np.sum(np.array(y1), axis=0)
		axes[2].plot(x1, y1, '-', label='K=1')
		x2, y2 = [np.array(data[i-1][:,0]) for data in all_data], [np.array(data[i-1][:,1]) for data in all_data]
		x2, y2 = np.mean(np.array(x2), axis=0), np.sum(np.array(y2), axis=0)
		axes[2].plot(x2, y2, '-', label='K={}'.format(i))
		min_len = min(len(x1), len(x2))
		if fill:
			axes[2].fill_between(x1[:min_len], y1[:min_len], y2[:min_len], color = 'lightcyan')

		axes[2].set_xlim([xmin,xmax])
		axes[2].set_ylim([ymin,ymax])
		if title:
			axes[2].set_title("Variance")
		if xlabel:
			axes[2].set_xlabel(r'$Log_{10}[\frac{P}{N}]$')

	plt.legend()
	plt.tight_layout()
	plt.subplots_adjust(left = 0.06, wspace = 0.25)
	plt.show()

def main():
	index = [1, 2]
	(xmin, xmax) = (-2.0, 1.0)
	(ymin, ymax) = (0, 2)
	save_dir = 'ResultsSingle/lambda=0.01/ND=10'
	plot_test_bias_var('totK.dat', 'biasK.dat', ['varDataK.dat', 'varInK.dat', 'varNoiseK.dat'],
		xmin, xmax, ymin, ymax, (36, 6),index, 
		fill=False, save_dir = save_dir,
		xlabel=True, title=True)


if __name__ == '__main__':
	main()
	