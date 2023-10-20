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

M=10
max_batchsize = 5

calctimes = []
batchsizes = []

for bs in range(1, max_batchsize+1):

    file_string = f'benchmark_batch_nonlinear_reaction_M{M}_BS{bs}.pkl'

    if isfile(file_string):
        with open(file_string, 'rb') as f:
            results = load(f)
        
        plt.subplot(121)
        plt.semilogy(results['max_rel_errors'][0],'x:',label=f'$bs={bs}$')

        calctimes.append(results['time'])
        batchsizes.append(bs)


plt.subplot(122)
plt.plot(batchsizes, calctimes, 'x:')
plt.xlabel('batch size $b$')
plt.ylabel('Calculation time in [$s$]')

plt.suptitle(f'Results for M={M}.')
plt.subplot(121)
plt.xlabel('Reduced basis size $N$')
plt.ylabel('Max rel. error in $H^1_0$ semi norm')
plt.legend(loc =1)

plt.show()





