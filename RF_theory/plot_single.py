import matplotlib
import matplotlib.pyplot as plt
font = {
		'size'   : 25}
matplotlib.rc('font', **font)
figsize = (12,8)
import numpy as np
import os
from scipy.interpolate import interp1d
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
	title=True, xlabel=True, legend=True,
	xaxis_visible=True):
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
		axes[0].yaxis.set_major_locator(plt.MaxNLocator(2))
		if not xaxis_visible:
			axes[0].xaxis.set_visible(False)
		else:
			x = xmin
			xticks = [x]
			while x<xmax:
				x+=1
				xticks.append(x)
			axes[0].set_xticks(xticks)
			axes[0].set_xticklabels([r"$10^{%s}$"%(str(int(x))) for x in xticks])
		if title:
			axes[0].set_title("Test Error")
		if xlabel:
			axes[0].set_xlabel(r'$\frac{P}{N}$')

	data = read_files(bias_file,save_dir, index=index)
	for i in index[1:]:
		axes[1].plot(data[0][:,0], data[0][:,1], '-', label='K=1')
		axes[1].plot(data[i-1][:,0], data[i-1][:,1], '-', label='K={}'.format(i))
		min_len = min(len(data[i-1][:,0]), len(data[0][:,0]))
		if fill:
			axes[1].fill_between(data[0][:,0][:min_len], data[0][:,1][:min_len], data[i-1][:,1][:min_len], color = 'lightyellow')
		axes[1].set_xlim([xmin,xmax])
		axes[1].set_ylim([ymin,ymax])
		axes[1].yaxis.set_major_locator(plt.MaxNLocator(2))
		if not xaxis_visible:
			axes[1].xaxis.set_visible(False)
		else:
			x = xmin
			xticks = [x]
			while x<xmax:
				x+=1
				xticks.append(x)
			axes[1].set_xticks(xticks)
			axes[1].set_xticklabels([r"$10^{%s}$"%(str(int(x))) for x in xticks])

		if title:
			axes[1].set_title(r"$Bias^{2}$")
		if xlabel:
			axes[1].set_xlabel(r'$\frac{P}{N}$')

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
		axes[2].yaxis.set_major_locator(plt.MaxNLocator(2))
		if not xaxis_visible:
			axes[2].xaxis.set_visible(False)
		else:
			x = xmin
			xticks = [x]
			while x<xmax:
				x+=1
				xticks.append(x)
			axes[2].set_xticks(xticks)
			axes[2].set_xticklabels([r"$10^{%s}$"%(str(int(x))) for x in xticks])

		if title:
			axes[2].set_title("Variance")
		if xlabel:
			axes[2].set_xlabel(r'$\frac{P}{N}$')

	if legend:
		plt.legend()

	plt.tight_layout()
	plt.subplots_adjust(left = 0.06, wspace = 0.25)
	plt.show()
def plot_three_phases(data_dict, file_name='totK.dat', 
	index=[1, 2], xmin=-2, xmax=2):
	num_phases = len(data_dict)
	fig,axes = plt.subplots(num_phases, 1, figsize=(8,10))
	keys = sorted(data_dict.keys())
	for i, ND in enumerate(keys):
		save_dir = data_dict[ND]
		data = read_files(file_name,save_dir, index=index)
		x1, y1 = data[0][:, 0], data[0][:, 1]
		x2, y2 = data[1][:, 0], data[1][:, 1]
		f1, f2 = interp1d(x1, y1), interp1d(x2, y2)
		xs = np.linspace(xmin, xmax, 300)
		axes[i].plot(xs, f1(xs) - f2(xs), 'k-')
		axes[i].plot(xs, [0]*len(xs), color ='grey', linestyle='--')
		axes[i].set_title(r"$N/D=10^{%s}$"%(str(ND)))
		if i !=num_phases-1:
			axes[i].xaxis.set_visible(False)
		else:
			x = xmin
			xticks = [x]
			while x< xmax:
				x+=1
				xticks.append(x)
			axes[i].set_xticks(xticks)
			axes[i].set_xticklabels([r"$10^{%s}$"%(str(x)) for x in xticks])
			axes[i].set_xlabel(r"P/N")
	plt.subplots_adjust(hspace = 0.5)
	plt.show()
def main():
	index = [1, 2]
	(xmin, xmax) = (-2.0, 1.0)
	(ymin, ymax) = (0, 4)
	save_dir = 'ResultsSingle/lambda=0.01/ND=0.1'
	plot_test_bias_var('totK.dat', 'biasK.dat', ['varDataK.dat', 'varInK.dat', 'varNoiseK.dat'],
		xmin, xmax, ymin, ymax, (32, 5.5),index, 
		fill=False, save_dir = save_dir,
		xlabel=False, title=True, legend=True,
		xaxis_visible=False)

	(xmin, xmax) = (-2.0, 1.0)
	(ymin, ymax) = (0, 2)
	save_dir = 'ResultsSingle/lambda=0.01/ND=3.1622'
	plot_test_bias_var('totK.dat', 'biasK.dat', ['varDataK.dat', 'varInK.dat', 'varNoiseK.dat'],
		xmin, xmax, ymin, ymax, (32, 5),index, 
		fill=True, save_dir = save_dir,
		xlabel=False, title=False, legend=False,
		xaxis_visible=False)

	(xmin, xmax) = (-2.0, 1.0)
	(ymin, ymax) = (0, 1)
	save_dir = 'ResultsSingle/lambda=0.01/ND=10'
	plot_test_bias_var('totK.dat', 'biasK.dat', ['varDataK.dat', 'varInK.dat', 'varNoiseK.dat'],
		xmin, xmax, ymin, ymax, (32, 6),index, 
		fill=False, save_dir = save_dir,
		xlabel=True, title=False, legend=False)

	plot_three_phases({-1: 'ResultsSingle/lambda=0.01/ND=0.1',
		0.5: 'ResultsSingle/lambda=0.01/ND=3.1622',
		1: 'ResultsSingle/lambda=0.01/ND=10'})
if __name__ == '__main__':
	main()
	