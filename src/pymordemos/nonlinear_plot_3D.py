import numpy as np
from pymor.core.pickle import load
import matplotlib.pyplot as plt
from matplotlib import cm
from os.path import isfile
from mpl_toolkits.mplot3d import axes3d

# Make data
X2, Y2, Z2 = axes3d.get_test_data(0.05)

M=25
plot_batches = range(1,31)
plot_Ms = [20, 25, 30, 35, 40, 45, 50]

Y = np.reshape(np.repeat(plot_batches, len(plot_Ms)),(len(plot_batches), len(plot_Ms)))
X = np.reshape(np.repeat(plot_Ms, len(plot_batches)), (len(plot_batches), len(plot_Ms)),order='F')

calc_times = np.empty((len(plot_batches), len(plot_Ms))) * np.nan
val_times = np.empty((len(plot_batches), len(plot_Ms))) * np.nan
num_ext = np.empty((len(plot_batches), len(plot_Ms))) * np.nan
num_iter = np.empty((len(plot_batches), len(plot_Ms))) * np.nan

fig = plt.figure()
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222, projection='3d')
ax3 = fig.add_subplot(223, projection='3d')
ax4 = fig.add_subplot(224, projection='3d')

for i in range(len(plot_batches)):

    bs = plot_batches[i]

    for j in range(len(plot_Ms)):

        M = plot_Ms[j]

        file_string = f'bm_nonlin_reac_M{M}_BS{bs}.pkl'

        if isfile(file_string):
            with open(file_string, 'rb') as f:
                results = load(f)

            calc_times[i,j] = results['calc_time']
            val_times[i,j] = results['val_time']
            num_ext[i,j] = results['num_extensions']
            num_iter[i,j] = results['num_iterations']


ax1.plot_surface(X, Y, calc_times, cmap=cm.winter, edgecolor='k')
ax1.set_xlim(np.min(plot_Ms), np.max(plot_Ms))
ax1.set_ylim(np.min(plot_batches), np.max(plot_batches))
ax1.set_xlabel('M')
ax1.set_ylabel('b')
ax1.set_zlabel('Offline greedy time')

ax2.plot_surface(X, Y, val_times, cmap=cm.winter, edgecolor='k')
ax2.set_xlim(np.min(plot_Ms), np.max(plot_Ms))
ax2.set_ylim(np.min(plot_batches), np.max(plot_batches))
ax2.set_xlabel('M')
ax2.set_ylabel('b')
ax2.set_zlabel('Validation time')

ax3.plot_surface(X, Y, num_ext, cmap=cm.winter, edgecolor='k')
ax3.set_xlim(np.min(plot_Ms), np.max(plot_Ms))
ax3.set_ylim(np.min(plot_batches), np.max(plot_batches))
ax3.set_xlabel('M')
ax3.set_ylabel('b')
ax3.set_zlabel('Final basis size')

ax4.plot_surface(X, Y, num_iter, cmap=cm.winter, edgecolor='k')
ax4.set_xlim(np.min(plot_Ms), np.max(plot_Ms))
ax4.set_ylim(np.min(plot_batches), np.max(plot_batches))
ax4.set_xlabel('M')
ax4.set_ylabel('b')
ax4.set_zlabel('# greedy iterations')

plt.show()
# plt.xlabel('Batch size $b$')
# plt.ylabel('Offline greedy time in [$s$]')
# plt.grid()

# plt.subplot(223)
# plt.plot(batchsizes, num_ext, 'o:', label='Final basis size $N$')
# plt.plot(batchsizes, num_iter, 'o:', label='# greedy iterations')
# plt.xlabel('Batch size $b$')
# plt.legend(loc=0)
# plt.grid()

# plt.subplot(224)
# plt.plot(batchsizes, val_times, 'o:')
# plt.xlabel('Batch size $b$')
# plt.ylabel('Validation time in [$s$]')
# plt.grid()

# plt.suptitle(f'Results for M={M}.')
# plt.subplot(221)
# plt.xlabel('Reduced basis size $N$')
# plt.ylabel('Max rel. error in $H^1_0$ semi norm')
# plt.legend(loc =1)
# plt.grid()

# plt.show()





