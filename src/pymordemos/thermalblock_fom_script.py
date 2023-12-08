#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import sys
import time
import pickle
from datetime import datetime

from typer import Argument, Option, run

from pymor.algorithms.error import plot_reduction_error_analysis, reduction_error_analysis, plot_batch_reduction
from pymor.core.pickle import dump
from pymor.parallel.default import new_parallel_pool, dummy_pool
from pymor.tools.typer import Choices
from pymor.algorithms.batchgreedy import rb_batch_greedy

# python -m thermalblock 3 2 5 100 5 --alg batch_greedy --plot-batch-comparison


def main(
    xblocks: int = Argument(..., help='Number of blocks in x direction.'),
    yblocks: int = Argument(..., help='Number of blocks in y direction.'),
    snapshots: int = Argument(
        ...,
        help='naive: ignored\n\n'
             'greedy/pod: Number of training_set parameters per block '
             '(in total SNAPSHOTS^(XBLOCKS * YBLOCKS) parameters).\n\n'
             'adaptive_greedy: size of validation set.\n\n'
    ),
    batchsize: int = Argument(..., help='Size of the (parallel) batch in each greedy iteration.')
):
    """Thermalblock demo."""


    pool = new_parallel_pool(allow_mpi=True)
    if pool is not dummy_pool:
        print(f'Using pool of {len(pool)} workers for parallelization.')
    else:
        print(f'No functional pool. Only dummy_pool is used.')

    assert batchsize>=0, 'Batch size must be nonnegative.'
    if batchsize==0: batchsize = len(pool)

    grid = 36
    rb_size = 100
    rtol = 1e-5
    test_snapshots = 100

    tic = time.perf_counter()

    fom, fom_summary = discretize_pymor(xblocks, yblocks, grid, use_list_vector_array=False)

    parameter_space = fom.parameters.space(0.1, 1.)

    n_test = 500
    test_mus = parameter_space.sample_randomly(n_test)

    tic = time.perf_counter()
    U = fom.solution_space.empty()
    for mu in test_mus:
        U.append(fom.solve(mu))
    toc = time.perf_counter()

    calctime = (toc - tic)/n_test

    print(f'Time: {calctime} [s]')

    # fom.enable_caching('memory')

    # print('')
    # print('')
    # print('RB generation for batch size ' + str(batchsize) + ' ...')

    # # define estimator for coercivity constant
    # from pymor.parameters.functionals import ExpressionParameterFunctional
    # coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', fom.parameters)

    # # inner product for computation of Riesz representatives
    # product = fom.h1_0_semi_product

    # reductor_str = 'traditional'
    # if reductor_str == 'residual_basis':
    #     from pymor.reductors.coercive import CoerciveRBReductor
    #     reductor = CoerciveRBReductor(fom, product=product, coercivity_estimator=coercivity_estimator,
    #                                     check_orthonormality=False)
    # elif reductor_str == 'traditional':
    #     from pymor.reductors.coercive import SimpleCoerciveRBReductor
    #     reductor = SimpleCoerciveRBReductor(fom, product=product, coercivity_estimator=coercivity_estimator,
    #                                         check_orthonormality=False)
    # else:
    #     assert False  # this should never happen
    
    # training_set = parameter_space.sample_uniformly(snapshots)
    # greedy_data = rb_batch_greedy(fom, reductor, training_set,
    #                               use_error_estimator=True,
    #                               error_norm=fom.h1_0_semi_norm,
    #                               max_extensions=rb_size,
    #                               pool=pool,
    #                               batchsize=batchsize,
    #                               rtol=rtol,
    #                               postprocessing=True
    #                               )

    # toc = time.perf_counter()
    # offline_time = toc - tic

    # rom = greedy_data['rom']
    
    # test_sample = parameter_space.sample_randomly(test_snapshots)
    # results = reduction_error_analysis(rom,
    #                                    fom=fom,
    #                                    reductor=reductor,
    #                                    error_estimator=True,
    #                                    error_norms=(fom.h1_0_semi_norm, fom.l2_norm),
    #                                    error_estimator_norm_index=0,
    #                                    test_mus=test_sample,
    #                                    basis_sizes=0,
    #                                    pool=pool
    #                                    )

    # # Online time
    # n_online = 500
    # tic = time.perf_counter()
    # for mu in parameter_space.sample_randomly(n_online):
    #     reductor.reconstruct(rom.solve(mu))
    # toc = time.perf_counter()
    # online_time = (toc - tic)/n_online
    
    # results['num_extensions'] = greedy_data['extensions']
    # results['num_iterations'] = greedy_data['iterations']
    # results['max_errs_pp'] = greedy_data['max_errs_pp']

    # results['val_time'] = online_time  # Specify what time is saved
    # results.pop('time', None)  # Delete old key
    # results['calc_time'] = offline_time # Also save offline time

    # with open(f'thermalblock_{xblocks}x{yblocks}_N{len(pool)}_BS{batchsize}.pkl', 'wb') as fp:
    #         pickle.dump(results, fp)

    # global test_results
    # test_results = results


def discretize_pymor(xblocks, yblocks, grid_num_intervals, use_list_vector_array):
    from pymor.analyticalproblems.thermalblock import thermal_block_problem
    from pymor.discretizers.builtin import discretize_stationary_cg
    from pymor.discretizers.builtin.list import convert_to_numpy_list_vector_array

    print('Discretize ...')
    # setup analytical problem
    problem = thermal_block_problem(num_blocks=(xblocks, yblocks))

    # discretize using continuous finite elements
    fom, _ = discretize_stationary_cg(problem, diameter=1. / grid_num_intervals)

    if use_list_vector_array:
        fom = convert_to_numpy_list_vector_array(fom)

    summary = f'''pyMOR model:
   number of blocks: {xblocks}x{yblocks}
   grid intervals:   {grid_num_intervals}
   ListVectorArray:  {use_list_vector_array}
'''

    return fom, summary


def discretize_fenics(xblocks, yblocks, grid_num_intervals, element_order):
    from pymor.tools import mpi

    if mpi.parallel:
        from pymor.models.mpi import mpi_wrap_model
        fom = mpi_wrap_model(lambda: _discretize_fenics(xblocks, yblocks, grid_num_intervals, element_order),
                             use_with=True, pickle_local_spaces=False)
    else:
        fom = _discretize_fenics(xblocks, yblocks, grid_num_intervals, element_order)

    summary = f'''FEniCS model:
   number of blocks:      {xblocks}x{yblocks}
   grid intervals:        {grid_num_intervals}
   finite element order:  {element_order}
'''

    return fom, summary


def _discretize_fenics(xblocks, yblocks, grid_num_intervals, element_order):

    from pymor.analyticalproblems.thermalblock import thermal_block_problem
    from pymor.discretizers.fenics import discretize_stationary_cg

    print('Discretize ...')
    # setup analytical problem
    problem = thermal_block_problem(num_blocks=(xblocks, yblocks))

    # discretize using continuous finite elements
    fom, _ = discretize_stationary_cg(problem, diameter=1. / grid_num_intervals, degree=element_order)

    return fom


def reduce_naive(fom, reductor, parameter_space, basis_size):

    tic = time.perf_counter()

    training_set = parameter_space.sample_randomly(basis_size)

    for mu in training_set:
        reductor.extend_basis(fom.solve(mu), method='trivial')

    rom = reductor.reduce()

    elapsed_time = time.perf_counter() - tic

    summary = f'''Naive basis generation:
   basis size set: {basis_size}
   elapsed time:   {elapsed_time}
'''

    return rom, summary


def reduce_greedy(fom, reductor, parameter_space, snapshots_per_block,
                  extension_alg_name, max_extensions, use_error_estimator, pool):

    from pymor.algorithms.greedy import rb_greedy

    # run greedy
    training_set = parameter_space.sample_uniformly(snapshots_per_block)
    greedy_data = rb_greedy(fom, reductor, training_set,
                            use_error_estimator=use_error_estimator, error_norm=fom.h1_0_semi_norm,
                            extension_params={'method': extension_alg_name}, max_extensions=max_extensions,
                            pool=pool)
    rom = greedy_data['rom']

    # generate summary
    real_rb_size = rom.solution_space.dim
    training_set_size = len(training_set)
    summary = f'''Greedy basis generation:
   size of training set:   {training_set_size}
   error estimator used:   {use_error_estimator}
   extension method:       {extension_alg_name}
   prescribed basis size:  {max_extensions}
   actual basis size:      {real_rb_size}
   elapsed time:           {greedy_data["time"]}
'''

    return rom, summary


def reduce_batch_greedy(fom, reductor, parameter_space, snapshots_per_block,
                        extension_alg_name, max_extensions, use_error_estimator, pool,
                        batchsize, greedy_start, atol, parallel_batch):

    from pymor.algorithms.batchgreedy import rb_batch_greedy

    # run greedy
    training_set = parameter_space.sample_uniformly(snapshots_per_block)
    greedy_data = rb_batch_greedy(fom, reductor, training_set,
                                  use_error_estimator=use_error_estimator, error_norm=fom.h1_0_semi_norm,
                                  extension_params={'method': extension_alg_name}, max_extensions=max_extensions,
                                  pool=pool, batchsize=batchsize, greedy_start=greedy_start, atol=atol, parallel_batch=parallel_batch)
    rom = greedy_data['rom']

    # generate summary
    real_rb_size = rom.solution_space.dim
    training_set_size = len(training_set)
    summary = f'''Greedy basis generation:
   size of training set:   {training_set_size}
   error estimator used:   {use_error_estimator}
   extension method:       {extension_alg_name}
   prescribed basis size:  {max_extensions}
   actual basis size:      {real_rb_size}
   elapsed time:           {greedy_data["time"]}
'''

    return rom, summary, greedy_data


def reduce_adaptive_greedy(fom, reductor, parameter_space, validation_mus,
                           extension_alg_name, max_extensions, use_error_estimator,
                           rho, gamma, theta, pool):

    from pymor.algorithms.adaptivegreedy import rb_adaptive_greedy

    # run greedy
    greedy_data = rb_adaptive_greedy(fom, reductor, parameter_space, validation_mus=-validation_mus,
                                     use_error_estimator=use_error_estimator, error_norm=fom.h1_0_semi_norm,
                                     extension_params={'method': extension_alg_name}, max_extensions=max_extensions,
                                     rho=rho, gamma=gamma, theta=theta, pool=pool)
    rom = greedy_data['rom']

    # generate summary
    real_rb_size = rom.solution_space.dim
    # the validation set consists of `validation_mus` random parameters plus the centers of the
    # adaptive sample set cells
    validation_mus += 1
    summary = f'''Adaptive greedy basis generation:
   initial size of validation set:  {validation_mus}
   error estimator used:            {use_error_estimator}
   extension method:                {extension_alg_name}
   prescribed basis size:           {max_extensions}
   actual basis size:               {real_rb_size}
   elapsed time:                    {greedy_data["time"]}
'''

    return rom, summary


def reduce_pod(fom, reductor, parameter_space, snapshots_per_block, basis_size):
    from pymor.algorithms.pod import pod

    tic = time.perf_counter()

    training_set = parameter_space.sample_uniformly(snapshots_per_block)

    print('Solving on training set ...')
    snapshots = fom.operator.source.empty(reserve=len(training_set))
    for mu in training_set:
        snapshots.append(fom.solve(mu))

    print('Performing POD ...')
    basis, singular_values = pod(snapshots, modes=basis_size, product=reductor.products['RB'])

    print('Reducing ...')
    reductor.extend_basis(basis, method='trivial')
    rom = reductor.reduce()

    elapsed_time = time.perf_counter() - tic

    # generate summary
    real_rb_size = rom.solution_space.dim
    training_set_size = len(training_set)
    summary = f'''POD basis generation:
   size of training set:   {training_set_size}
   prescribed basis size:  {basis_size}
   actual basis size:      {real_rb_size}
   elapsed time:           {elapsed_time}
'''

    return rom, summary


if __name__ == '__main__':
    run(main)
