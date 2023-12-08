import numpy as np
from pymor.core.pickle import load
import matplotlib.pyplot as plt
from os.path import isfile

# with open('benchmark_batch_nonlinear_reaction_M10_BS1.pkl', 'rb') as f:
#     results_bs1 = load(f)
# with open('benchmark_batch_nonlinear_reaction_M10_BS2.pkl', 'rb') as f:
#     results_bs2 = load(f)
# with open('benchmark_batch_nonlinear_reaction_M10_BS3.pkl', 'rb') as f:
#     results_bs3 = load(f)

# rel_err_bs1 = results_bs1['max_rel_errors'][0]
# rel_err_bs2 = results_bs2['max_rel_errors'][0]
# rel_err_bs3 = results_bs3['max_rel_errors'][0]

# times = [results_bs1['time'], results_bs2['time'], results_bs3['time']]

# plt.subplot(121)
# plt.semilogy(rel_err_bs1,'x:')
# plt.semilogy(rel_err_bs2,'x:')
# plt.semilogy(rel_err_bs3,'x:')

# plt.subplot(122)
# plt.plot([1, 2, 3], times,'x:')

# plt.show()

# plt.switch_backend('QtaAgg')

M=20
plot_batches = [2, 4, 8, 12]
max_batchsize = 50

calc_times = []
val_times = []
num_ext = []
num_iter = []
batchsizes = []

plt.subplot(121)
if len(plot_batches)==0:
    plot_this_batch = np.ones(max_batchsize+1)
else:
    plot_this_batch = np.zeros(max_batchsize+1)
    plot_this_batch[plot_batches] = 1

for bs in range(1, max_batchsize+1):

    file_string = f'bm_nonlin_reac_N1_M{M}_BS{bs}.pkl'

    if isfile(file_string):
        with open(file_string, 'rb') as f:
            results = load(f)
        
        if plot_this_batch[bs]:
            pp = results['max_errs_pp']
            pp = pp[::-1]
            plt.subplot(121)
            plt.semilogy(results['max_rel_errors'][0],'x:',label=f'$bs={bs}$')
            plt.subplot(122)
            plt.semilogy(pp,'x:',label=f'$bs={bs}$')

        calc_times.append(results['calc_time'])
        val_times.append(results['val_time'])
        num_ext.append(results['num_extensions'])
        num_iter.append(results['num_iterations'])
        batchsizes.append(bs)


# plt.subplot(222)
# plt.plot(batchsizes, calc_times, 'o:')
# plt.xlabel('Batch size $b$')
# plt.ylabel('Offline greedy time in [$s$]')
# plt.grid()

# plt.subplot(223)
# plt.plot(batchsizes, num_ext, 'o:', label='Final basis size $N$')
# plt.plot(batchsizes, num_iter, 'o:', label='# greedy iterations')
# plt.xlabel('Batch size $b$')
# plt.legend(loc=0)
# plt.grid()

# plt.subplot(221)
# plt.plot(batchsizes, val_times, 'o:')
# plt.xlabel('Batch size $b$')
# plt.ylabel('Validation time in [$s$]')
# plt.grid()

plt.subplot(121)
plt.xlabel('Reduced basis size $N$')
plt.ylabel('Quotient')
plt.legend(loc =1)
plt.grid()

plt.suptitle(f'Results for M={M}.')
plt.subplot(122)
plt.xlabel('Reduced basis size $N$')
plt.ylabel('postprocessing error')
plt.legend(loc =1)
plt.grid()

plt.show()





