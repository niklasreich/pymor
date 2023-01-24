#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import sys
import os
import pathlib
from glob import glob
import pandas as pd
import numpy as np

from typer import Argument, Option, run

s = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(s + '/../../src/')

from pymor.algorithms.error import reduction_error_analysis
from pymor.core.pickle import load
#from pymor.parallel.default import new_parallel_pool
#from pymor.tools.typer import Choices

# python -m thermalblock 3 2 5 100 5 --alg batch_greedy --plot-batch-comparison

def main(
    res_dir: str = Argument(..., help='Directory of pickled benchmark results.'),
    export_error_data: bool = Option(False, help='Create files to plot errors in Matlab.'),
    stoch_test_size: int = Option(10, help='Use COUNT snapshots for stochastic error estimation.'),
    export_mu_choice: bool = Option(False, help='Create files to analyze the snapshot choices.'),
    export_calc_time: bool = Option(False, help='Create files to analyze the overall computation times.'),
):  
    path = 'results/'+ res_dir

    if not os.path.exists(path):
        print('ERROR: the directory:')
        print('  ' + res_dir)
        print("is not an extisting subdirectory of 'results/'")
        return

    if export_error_data:

        # Load detailed model
        print('\n\n## Export Error Data ##\n')
        print('Loading detailed model ...')
        detailed_path = glob('*_detailed',root_dir=path + '/')
        if len(detailed_path)==0:
            print('ERROR: the directory:')
            print('  ' + res_dir)
            print("has no pickle-file for a detailed model")
            return
        if len(detailed_path)>1:
            print('ERROR: the directory:')
            print('  ' + res_dir)
            print("has multiple pickle-files for a detailed model")
            return
        detailed_path = detailed_path[0]
        with open(path + '/' + detailed_path, 'rb') as f:
            fom = load(f)
        print('\tDetailed model loaded.')

        # Identify reduced models
        print('Identifying reduced models ...')
        reduced_files = glob('*_reduced_bs*',root_dir=path + '/')
        print('\t' + str(len(reduced_files)) + ' pickle-file(s) of reduced models found.')

        # Calculate errors for every reduced model
        print('Calculation of errors:')

        for i in range(len(reduced_files)):

            # load reduced model
            reduced_path = reduced_files[i]
            with open(path + '/' + reduced_path, 'rb') as f:
                (batch_greedy_data, parameter_space, reductor) = load(f)

            # calculate errors
            results = reduction_error_analysis(batch_greedy_data['rom'],
                                        fom=fom,
                                        reductor=reductor,
                                        error_estimator=True,
                                        error_norms=(fom.h1_0_semi_norm, fom.l2_norm),
                                        condition=True,
                                        test_mus=parameter_space.sample_randomly(stoch_test_size),
                                        basis_sizes=0,
                                        pool=None
                                        )

            # retrace batchsize and dtermine name string
            ind = len(reduced_path)-1
            while  reduced_path[ind-1:].isnumeric():
                ind -= 1
            batchsize = int(reduced_path[ind:])
            if batchsize<10:
                size_str = '0' + str(batchsize)
            else:
                size_str = str(batchsize)
            filename = path + '/err_over_basissize_bs' + size_str + '_'

            # save in results
            error_norms = 'norms' in results
            error_estimator = 'error_estimates' in results

            basis_sizes = results['basis_sizes']
            if error_norms:
                error_norm_names = results['error_norm_names']
                max_errors = results['max_errors']
                errors = results['errors']
            if error_estimator:
                max_estimates = results['max_error_estimates']

            if error_norms or error_estimator:
                if error_norms:
                    for name, errors in zip(error_norm_names, max_errors):
                        df = pd.DataFrame(errors)
                        df.to_csv(filename+name+'.csv')
                if error_estimator:
                    df = pd.DataFrame(max_estimates)
                    df.to_csv(filename+'err_est.csv')      
    
    if export_mu_choice:

        # Load detailed model
        print('\n\n## Export Data for choices of mu ##\n')
        print('Loading detailed model ...')
        detailed_path = glob('*_detailed',root_dir=path + '/')
        if len(detailed_path)==0:
            print('ERROR: the directory:')
            print('  ' + res_dir)
            print("has no pickle-file for a detailed model")
            return
        if len(detailed_path)>1:
            print('ERROR: the directory:')
            print('  ' + res_dir)
            print("has multiple pickle-files for a detailed model")
            return
        detailed_path = detailed_path[0]
        with open(path + '/' + detailed_path, 'rb') as f:
            fom = load(f)
        print('\tDetailed model loaded.')

        # Identify reduced models
        print('Identifying reduced models ...')
        reduced_files = glob('*_reduced_bs*',root_dir=path + '/')
        print('\t' + str(len(reduced_files)) + ' pickle-file(s) of reduced models found.')

        #Identifying the number of snapshots per parameter dimension
        config_path = glob('config.txt',root_dir=path + '/')
        if len(config_path)==0:
            print('ERROR: the directory:')
            print('  ' + res_dir)
            print("has no config file")
            return
        if len(config_path)>1:
            print('ERROR: the directory:')
            print('  ' + res_dir)
            print("has multiple config files")
            return
        config_path = config_path[0]
        with open(path + '/' + config_path, 'r') as f:
            config_str = f.read()
        config_key = 'parameter vals per dim: '
        ind = str.find(config_str,config_key)
        if ind < 0:
            print('ERROR: the config file does not include the number of snapshots')
        ind += len(config_key)
        ind_end = ind+1
        while ind_end < len(config_str):
            if not config_str[ind:ind_end+1].isnumeric():
                break
            ind_end += 1
        snapshots_per_dim = int(config_str[ind:ind_end])

        # Retracing the mu choices for every reduced model
        print('Retracing the mu choices...')

        for i in range(len(reduced_files)):

            # load reduced model
            reduced_path = reduced_files[i]
            with open(path + '/' + reduced_path, 'rb') as f:
                (batch_greedy_data, parameter_space, reductor) = load(f)

            #Retracing training space
            training_space = parameter_space.sample_uniformly(snapshots_per_dim)

            #find the indices of the snapshot in the training space
            mus = batch_greedy_data['max_err_mus_ext']
            mu_sequence = np.zeros(len(mus))
            mu_matrix = np.zeros((len(mus),len(mus[0]['diffusion'])))
            for i in range(len(mus)):
                mu_sequence[i] = training_space.index(mus[i])
                mu_matrix[i,:] = mus[i]['diffusion']

            # retrace batchsize and determine name string
            ind = len(reduced_path)-1
            while  reduced_path[ind-1:].isnumeric():
                ind -= 1
            batchsize = int(reduced_path[ind:])
            if batchsize<10:
                size_str = '0' + str(batchsize)
            else:
                size_str = str(batchsize)
            filename = path + '/mu_sequence_bs' + size_str + '_'

            # save in results
            df = pd.DataFrame(mu_sequence)
            df.to_csv(path + '/mu_sequence_bs' + size_str + '.csv')
            df = pd.DataFrame(mu_matrix)
            df.to_csv(path + '/mu_matrix_bs' + size_str + '.csv')
            print('\tbatch size ' + size_str + ' done.')

    
    if export_calc_time:

        # Load detailed model
        print('\n\n## Export Calculation Times ##\n')

        # Identify reduced models
        print('Identifying reduced models ...')
        reduced_files = glob('*_reduced_bs*',root_dir=path + '/')
        print('\t' + str(len(reduced_files)) + ' pickle-file(s) of reduced models found.')

        data = {}
        data['batchsize']=[]
        data['calc_time']=[]

        # Retracing the mu choices for every reduced model
        print('Retracing the mu choices...')

        for i in range(len(reduced_files)):

            # load reduced model
            reduced_path = reduced_files[i]
            with open(path + '/' + reduced_path, 'rb') as f:
                (batch_greedy_data, parameter_space, reductor) = load(f)


            # retrace batchsize and determine name string
            ind = len(reduced_path)-1
            while  reduced_path[ind-1:].isnumeric():
                ind -= 1
            batchsize = int(reduced_path[ind:])

            data['batchsize'].append(batchsize)
            data['calc_time'].append(batch_greedy_data['time'])

        # save in results
        df = pd.DataFrame(data)
        df.to_csv(path + '/calc_times.csv')
        print('\tDone.')

if __name__ == '__main__':
    run(main)
