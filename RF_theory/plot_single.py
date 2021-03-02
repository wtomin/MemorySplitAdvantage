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
def plot_files(filename, N_D, xmin=-2.0, xmax=2.0, ymin=0, ymax=2, index=[1, 2], N_D_v = 0.5,
	ylabel = "Generalization Error", fill_between_c = None):
	save_dir = './ND={}/'.format(N_D)
	data = read_files(filename,save_dir, index=index)
	for i in index[1:]:
		fig, ax = plt.subplots(figsize=figsize)
		plt.plot(data[0][:,0], data[0][:,1], '--', label='K=1')
		plt.plot(data[i-1][:,0], data[i-1][:,1], '-', label='K={}'.format(i))
		if fill_between_c is not None:
			plt.fill_between(data[0][:,0], data[0][:,1], data[i-1][:,1], color = fill_between_c)
		axes = plt.gca()
		axes.set_xlim([xmin,xmax])
		axes.set_ylim([ymin,ymax])
		plt.ylabel(ylabel)
		plt.xlabel(r'$Log_{10}[P/N]$')
		plt.legend(title='Ensemble Size')
		#plt.text(1.5, 0.5, r'N/D={}'.format(N_D_v))
		plt.show()
def plot_variances(filenames, N_D, xmin=-2.0, xmax=2.0, ymin=0, ymax=2, index=[1, 2], N_D_v = 0.5,
	ylabel = "Generalization Error", fill_between_c = None):
	save_dir = './ND={}/'.format(N_D)
	all_data = [read_files(filename, save_dir, index=index) for filename in filenames]

	for i in index[1:]:
		fig, ax = plt.subplots(figsize=figsize)
		x1, y1 = [np.array(data[0][:,0]) for data in all_data], [np.array(data[0][:,1]) for data in all_data]
		x1, y1 = np.mean(np.array(x1), axis=0), np.sum(np.array(y1), axis=0)
		plt.plot(x1, y1, '--', label='K=1')
		x2, y2 = [np.array(data[i-1][:,0]) for data in all_data], [np.array(data[i-1][:,1]) for data in all_data]
		x2, y2 = np.mean(np.array(x2), axis=0), np.sum(np.array(y2), axis=0)
		plt.plot(x2, y2, '-', label='K={}'.format(i))
		if fill_between_c is not None:
			plt.fill_between(x1, y1, y2, color = fill_between_c)
		axes = plt.gca()
		axes.set_xlim([xmin,xmax])
		axes.set_ylim([ymin,ymax])
		plt.ylabel(ylabel)
		plt.xlabel(r'$Log_{10}[P/N]$')
		plt.legend(title='Ensemble Size')
		#plt.text(1.5, 0.5, r'N/D={}'.format(N_D_v))
		plt.show()
def plot_byTerm(xs, ys, labels, xlabel=r'$Log_{10}[P/N]$', ylabel = '', N_D_v = 0.5, 
	 xmin=-2.0, xmax=2.0, ymin=0, ymax=2, 
	title = ''):
	fig, ax = plt.subplots(figsize=figsize)
	for x, y, label in zip(xs, ys, labels):
		plt.plot(x, y, label=label)
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	axes = plt.gca()
	axes.set_xlim([xmin,xmax])
	axes.set_ylim([ymin,ymax])
	plt.legend()
	#plt.text(1.5, 0.5, r'N/D={}'.format(N_D_v))
	plt.title(title)
	plt.show()

def plot_test_bias_var(test_error_file, bias_file, var_files, N_D, 
	xmin=-2.0, xmax=2.0, ymin=0, ymax=2, figsize = (36, 8), index = [1,2], fill=True, save_dir = ''):
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
		axes[0].set_ylabel("Generalization Error")
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
		axes[1].set_ylabel("Bias Square")
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
		axes[2].set_ylabel("Variance")
		axes[2].set_xlabel(r'$Log_{10}[\frac{P}{N}]$')

	plt.legend(title='Ensemble Size')
	plt.tight_layout()
	plt.subplots_adjust(left = 0.06, wspace = 0.25)
	plt.show()

def main():
	index = [1, 2]
	# filename = 'totK.dat'
	N_D = '1'
	(xmin, xmax) = (-2.0, 1.0)
	(ymin, ymax) = (0, 2)
	save_dir = 'lambda=001//ND=10'
	# total = read_files(filename,save_dir, index=index)
	# bias = read_files('biasK.dat' , save_dir,index=index)
	# varData = read_files('varDataK.dat', save_dir,index=index)
	# varInt = read_files('varInK.dat',save_dir, index=index)
	# varNoise = read_files('varNoiseK.dat',save_dir, index=index)

	# plot_byTerm(xs = [total[id][:, 0], bias[id][:, 0], varData[id][:, 0], varInt[id][:, 0], varNoise[id][:, 0]],
	# 	ys = [total[id][:, 1], bias[id][:, 1], varData[id][:, 1], varInt[id][:, 1],varNoise[id][:, 1]],
	# 	labels = ['Total', 'Bias', 'VarData', 'varInt', 'varNoise'],
	# 	title = 'K=1',N_D_v = 1.,
	# 	)

	# plot_byTerm(xs = [total[id][:, 0], bias[id][:, 0], varData[id][:, 0], varInt[id][:, 0], varNoise[id][:, 0]],
	# 	ys = [total[id][:, 1], bias[id][:, 1], varData[id][:, 1], varInt[id][:, 1],varNoise[id][:, 1]],
	# 	labels = ['Total', 'Bias', 'VarData', 'varInt', 'varNoise'],
	# 	title = 'K=2', N_D_v = 1.,
	# 	)
	# plot_files(filename, N_D, xmin, xmax, ymin, ymax, index, N_D_v=1., 
	# 	ylabel = "Generalization Error", fill_between_c = 'mistyrose')
	# filename = 'biasK.dat' 
	# plot_files(filename, N_D, xmin, xmax, ymin, ymax, index, N_D_v=1., 
	# 	ylabel = "Bias Square", fill_between_c = 'lightyellow')
	# plot_variances(['varDataK.dat', 'varInK.dat', 'varNoiseK.dat'], N_D, xmin, xmax, ymin, ymax, index, N_D_v=1., 
	# 	ylabel = "Variance", fill_between_c = 'lightcyan')
	# draw theoretical results
	plot_test_bias_var('totK.dat', 'biasK.dat', ['varDataK.dat', 'varInK.dat', 'varNoiseK.dat'],
		N_D, xmin, xmax, ymin, ymax, (36, 6),index, fill=False, save_dir = save_dir)


	# compare theoretical with synthetica dataset results

if __name__ == '__main__':
	main()
	