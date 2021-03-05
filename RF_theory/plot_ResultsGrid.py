import matplotlib
import matplotlib.pyplot as plt
font = {
        'size'   : 25}
matplotlib.rc('font', **font)
figsize = (8, 8)
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

def plot_total(template, xlabel, ylabel, save_dir, four_phases = {}, xmin=-1.0, xmax=2.0, xpoints=50, 
	ymin=-1.9, ymax=2, ypoints=50, index=[1, 2] ,vmin=-0.1, vmax=0.1, 
	interpolation='nearest', title=''):
	fig, axes = plt.subplots(1, 1, figsize=figsize)
	x_list = []
	subdir_list = []
	for subdir in sorted(os.listdir(save_dir)):
		x = float(subdir.split(template)[-1])
		x_list.append(x)
		subdir_list.append(subdir)
	x_list = np.array(x_list)
	ids = np.argsort(x_list)
	x_list, subdir_list = x_list[ids], [subdir_list[i] for i in ids]
	i = 0
	filename = 'totK.dat'
	outputs = []
	ys = np.linspace(ymin, ymax, ypoints)
	mask = np.multiply(x_list >=(10**xmin), x_list<=(10**xmax))
	x_list = x_list[mask]
	subdir_list = [subdir_list[i] for i,m in enumerate(mask) if m]
	for x, subdir in zip(x_list, subdir_list):
		cur_dir = os.path.join(save_dir, subdir)
		data = read_files(filename, cur_dir, index=index)
		#assert np.isclose(data[0][:,0], data[1][:,0]), "the inputs x of K=1 and K=2 are not close"
		y1, t1 = data[0][:,0], data[0][:,1]
		y2, t2 = data[1][:,0], data[1][:,1]
		f1, f2 = interp1d(y1, t1), interp1d(y2, t2) 
		chi = f1(ys) - f2(ys)
		outputs.append(chi)

	outputs = np.array(outputs).T
	# flip the image and ys 
	outputs = outputs[::-1, :]
	ys = ys[::-1]
	ax = axes.imshow(outputs, cmap=plt.get_cmap('coolwarm'),vmax=vmax, vmin=vmin,
		interpolation=interpolation)
	cbar = fig.colorbar(ax, ax=axes, ticks=[vmin, (vmin+vmax)/2, vmax])
	cbar.ax.set_yticklabels(['< {}'.format(vmin), '0', '> {}'.format(vmax)]) 
	axes.set_xlabel(xlabel)
	axes.set_ylabel(ylabel)
	# ticks 
	xticks = [xmin]
	x = xmin
	while x<xmax:
		x += 1
		xticks.append(x)
	yticks = [ymin]
	y = ymin
	while y<ymax:
		y += 1
		yticks.append(y)
	xids = [np.argmin([np.abs(np.log10(x) - xt) for x in x_list]) for xt in xticks]
	yids = [np.argmin([np.abs(y - yt) for y in ys]) for yt in yticks]
	axes.set_xticks(xids)
	axes.set_yticks(yids)
	axes.set_xticklabels(xticks)
	axes.set_yticklabels(yticks)
	#plt.tight_layout()
	axes.set_title(title)
	plt.show()
	# draw four phases
	if len(four_phases) !=0:
		fig, axes = plt.subplots(len(four_phases), 1 , figsize=(8, 10))
		for i, phase in enumerate(four_phases.keys()):
			x = four_phases[phase]
			xid =np.argmin([np.abs(np.log10(rx) - x) for rx in x_list])
			chi = outputs[:, xid]
			axes[i].plot(ys, chi, 'k-')
			axes[i].plot(ys, [0]*len(ys), color = 'grey', linestyle='--')
			#axes[i].set_ylim([vmin, vmax])
			# axes[i].set_yticks([0, max(chi)])
			# axes[i].set_yticklabels([0, np.round(max(chi), 1)])
			axes[i].set_title(phase)
			if i!= len(four_phases)-1:
				axes[i].xaxis.set_visible(False)
			else:
				axes[i].set_xticks(ys[yids])
				axes[i].set_xticklabels(yticks)
				axes[i].set_xlabel(ylabel)
		#plt.ylabel(r"$\chi$")
		plt.subplots_adjust(hspace = 0.5)
		fig.text(-0.05, 0.5, r"$\chi$", va='center', rotation='vertical')
		plt.show()

if __name__ == '__main__':
	# plot_total("ND=", xlabel= r'$Log_{10}[\frac{N}{D}]$', ylabel = r'$Log_{10}[\frac{P}{N}]$',
	# 	save_dir='ResultsGrid\\lambda=0.01', xmin = -1, xmax=2, ymin=-1, ymax=2, 
	# 	index=[1,2], interpolation='bilinear', title=r'$\chi$', 
	# 	four_phases={r"$Log_{10}[\frac{N}{D}]=-1$": -1, 
	# 	r"$Log_{10}[\frac{N}{D}]=0.5$": 0.5, r"$Log_{10}[\frac{N}{D}]=1$": 1})
	plot_total("lambda=", xlabel= r'$Log_{10}[\lambda]$', ylabel = r'$Log_{10}[\frac{P}{N}]$',
		save_dir='ResultsGrid/ND=10', xmin = -4, xmax=0, ymin=-2, ymax=2, 
		index=[1,2], interpolation='bilinear', title=r'$\chi$', 
		four_phases={r"$Log_{10}[\lambda]=-4$": -4, 
		r"$Log_{10}[\lambda]=0.5$": 0.5,})# given lambda




