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
plot_batches = [1, 2, 4, 8, 12]
max_batchsize = 30
max_P = 16

calc_times = []
val_times = []
num_ext = []
num_iter = []
num_pp = []
batchsizes = []

calc_times_p = []
val_times_p = []
num_ext_p = []
num_iter_p = []
num_pp_p = []
batchsizes_p = []

plt.subplot(221)
if len(plot_batches)==0:
    plot_this_batch = np.ones(max_batchsize+1)
else:
    plot_this_batch = np.zeros(max_batchsize+1)
    plot_this_batch[plot_batches] = 1

for p in range(1, max_P+1):

    for bs in range(1, max_batchsize+1):

        file_string = f'thermalblock_3x2_N{p}_BS{bs}.pkl'

        if isfile(file_string):
            with open(file_string, 'rb') as f:
                results = load(f)
            
            if plot_this_batch[bs] and p==1:
                plt.subplot(221)
                plt.semilogy(results['max_rel_errors'][0],'x:',label=f'$b={bs}$')
                # plt.subplot(224)
                # plt.semilogy(results['max_rel_errors'][0][1:]/results['max_rel_errors'][0][:-1],'x:',label=f'$bs={bs}$')

            if p==1:
                calc_times.append(results['calc_time'])
                val_times.append(results['val_time'])
                num_ext.append(results['num_extensions'])
                num_iter.append(results['num_iterations'])
                num_pp.append(results['num_extensions'] - len(results['max_errs_pp']) + 1)
                batchsizes.append(bs)
            elif p==bs:
                calc_times_p.append(results['calc_time'])
                val_times_p.append(results['val_time'])
                num_ext_p.append(results['num_extensions'])
                num_iter_p.append(results['num_iterations'])
                num_pp_p.append(results['num_extensions'] - len(results['max_errs_pp']) + 1)
                batchsizes_p.append(bs)


plt.subplot(222)
plt.plot(batchsizes, calc_times, 'o:', label='sequential')
plt.plot(batchsizes_p, calc_times_p, 'o:', label='mpi parallel')
plt.xlabel('Batch size $b$')
plt.ylabel('Offline greedy time in [$s$]')
plt.legend(loc=0)
plt.grid()

plt.subplot(223)
plt.plot(batchsizes, num_ext, 'o:', label='Basis size $N$ after greedy')
plt.plot(batchsizes, num_pp, 'o:', label='Final basis size $N$ after postprocessing')
plt.plot(batchsizes, num_iter, 'o:', label='greedy iterations')
# plt.plot(batchsizes_p, num_ext_p, 'o:', label='Final basis size $N$ for $p=b$')
# plt.plot(batchsizes_p, num_iter_p, 'o:', label='# greedy iterations for $p=b$')
# plt.plot(batchsizes_p, num_pp_p, 'o:', label='Final basis size $N$ after PP for $p=1$')
plt.xlabel('Batch size $b$')
plt.legend(loc=0)
plt.grid()

plt.subplot(224)
plt.plot(batchsizes, val_times, 'o:')
#plt.plot(batchsizes_p, val_times_p, 'o:', label='mpi parallel')
plt.xlabel('Batch size $b$')
plt.ylabel('online time per $\mu$ in [$s$]')
plt.legend(loc=0)
plt.grid()

# plt.subplot(224)
# plt.xlabel('Reduced basis size $N$')
# plt.ylabel('Quotient')
# plt.legend(loc =1)
# plt.grid()

plt.suptitle(f'3x2 Thermalblock: 9 snapshots per dim, 2665 dofs')
plt.subplot(221)
plt.xlabel('Reduced basis size $N$')
plt.ylabel('Max rel. error in $H^1_0$ semi norm')
plt.legend(loc =1)
plt.grid()
plt.show()
