# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import time

import numpy as np
import multiprocessing as mp

from pymor.algorithms.greedy import RBSurrogate
from pymor.core.base import BasicObject, abstractmethod
from pymor.core.exceptions import ExtensionError
from pymor.core.logger import getLogger
from pymor.parallel.dummy import dummy_pool
from pymor.parallel.interface import RemoteObject
from pymor.parallel.manager import RemoteObjectManager
from pymor.parallel.mpi import MPIPool
from mpi4py import MPI


def weak_batch_greedy(surrogate, training_set, atol=None, rtol=None, max_extensions=None, pool=None,
                      batchsize=None, greedy_start=None, postprocessing=False):
    """Weak greedy basis generation algorithm :cite:`BCDDPW11`.

    This algorithm generates an approximation basis for a given set of vectors
    associated with a training set of parameters by iteratively evaluating a
    :class:`surrogate <WeakGreedySurrogate>` for the approximation error on
    the training set and adding the worst approximated vector (according to
    the surrogate) to the basis.

    The constructed basis is extracted from the surrogate after termination
    of the algorithm.

    Parameters
    ----------
    surrogate
        An instance of :class:`WeakGreedySurrogate` representing the surrogate
        for the approximation error.
    training_set
        The set of parameter samples on which to perform the greedy search.
    atol
        If not `None`, stop the algorithm if the maximum (estimated) error
        on the training set drops below this value.
    rtol
        If not `None`, stop the algorithm if the maximum (estimated)
        relative error on the training set drops below this value.
    max_extensions
        If not `None`, stop the algorithm after `max_extensions` extension
        steps.
    pool
        If not `None`, a |WorkerPool| to use for parallelization. Parallelization
        needs to be supported by `surrogate`.

    Returns
    -------
    Dict with the following fields:

        :max_errs:               Sequence of maximum estimated errors during the greedy run.
        :max_err_mus:            The parameters corresponding to `max_errs`.
        :extensions:             Number of performed basis extensions.
        :time:                   Total runtime of the algorithm.
    """

    if batchsize is None:
        batchsize = 1

    if greedy_start is None:
        greedy_start = 'standard'

    logger = getLogger('pymor.algorithms.greedy.weak_greedy')
    training_set = list(training_set)
    logger.info(f'Started batch greedy search on training set of size {len(training_set)}.')

    tic = time.perf_counter()
    if not training_set:
        logger.info('There is nothing else to do for an empty training set.')
        return {'max_errs': [], 'max_err_mus': [], 'extensions': 0,
                'time': time.perf_counter() - tic}

    # parallel_batch = False
    if pool is None:
        pool = dummy_pool
    # Always use parallel extension of basis by batch
    parallel_batch = True

    # Distribute the training set evenly among the workers.
    training_set_rank = pool.scatter_list(training_set)

    # if surrogate.extension_params['method'] == 'gram_schmidt_batch':
    #     surrogate.extension_params['orthogonalize'] = False

    extensions = 0
    iterations = 0
    max_errs_ext = []
    max_err_mus_ext = []
    max_errs_iter = []
    max_err_mus_iter = []
    appended_mus = []

    stopped = False
    while not stopped:
        with logger.block('Estimating errors ...'):
            # max_err, max_err_mu = surrogate.evaluate(training_set)
            this_i_errs = surrogate.evaluate(training_set_rank, return_all_values=True)
            this_i_mus = []
            if (extensions == 0) and (iterations == 0):
                if greedy_start == 'minmax':
                    # for the first batch prefer snapshots that only contain the minimal
                    # or maximal value for each parameter
                    logger.info('First batch computation: minmax.')
                    max_err = np.max(this_i_errs)
                    min_val = float('inf')
                    max_val = float('-inf')
                    for i in range(len(training_set)):
                        this_min = np.min(training_set[i]['diffusion'])
                        this_max = np.max(training_set[i]['diffusion'])
                        if this_min < min_val:
                            min_val = this_min
                        if this_max > max_val:
                            max_val = this_max
                    for i in range(len(training_set)):
                        stopped = False
                        mu = training_set[i]['diffusion']
                        for j in range(len(training_set[0]['diffusion'])):
                            if not (mu[j] == min_val or mu[j] == max_val):
                                stopped = True
                                break
                        if not stopped:
                            this_i_errs[i] = 2*max_err
                elif greedy_start == 'random':
                    # start with random config in the first batch
                    # this achieved by artificially increasing certain error values
                    logger.info('First batch computation: random.')
                    rand_ind = np.random.randint(0, len(training_set), size=batchsize)
                    while len(np.unique(rand_ind)) < len(rand_ind):
                        rand_ind = np.random.randint(0, len(training_set), size=batchsize)
                    max_err = np.max(this_i_errs)
                    for i in range(batchsize):
                        this_i_errs[rand_ind[i]] = 2*max_err
                elif greedy_start == 'standard':
                    # for 'standard' start with the chronological frist snapshots
                    # in the first batch
                    logger.info('First batch computation: standard.')
                else:
                    logger.info('Unknwon starting method for th greedy alogrithm. Aborting now.')
                    return
            for i in range(batchsize):
                max_ind = np.argmax(this_i_errs)
                if i == 0:  # only once per batch -> once every greedy iteration
                    max_err = this_i_errs[max_ind]
                    max_err_mu = training_set[max_ind]
                    max_errs_iter.append(max_err)
                    max_err_mus_iter.append(max_err_mu)

                # for every mu of the batch -> once every basis extension
                max_errs_ext.append(this_i_errs[max_ind])
                max_err_mus_ext.append(training_set[max_ind])

                this_i_mus.append(training_set[max_ind])
                this_i_errs[max_ind] = 0

                appended_mus.append(training_set[max_ind])

            # max_errs.append(max_err)
            # max_err_mus.append(max_err_mu)

        logger.info(f'Maximum error after {iterations} iterations ({extensions} extensions): {max_err} (mu = {max_err_mu})')

        if atol is not None and max_err <= atol:
            logger.info(f'Absolute error tolerance ({atol}) reached! Stopping extension loop.')
            stopped = True
            break

        if rtol is not None and max_err / max_errs_iter[0] <= rtol:
            logger.info(f'Relative error tolerance ({rtol}) reached! Stopping extension loop.')
            stopped = True
            break

        stopped = True
        if parallel_batch:
            with logger.block(f'Extending in parallel...'):
                try:
                    extensions += surrogate.extend(this_i_mus)
                    stopped = False
                except ExtensionError:
                    pass
        else:
            for i in range(batchsize):
                with logger.block(f'Extending surrogate for mu = {this_i_mus[i]} ...'):
                    try:
                        # if i==batchsize-1:
                        #     surrogate.extension_params['orthogonalize'] = True
                        # else:
                        #     surrogate.extension_params['orthogonalize'] = False
                        surrogate.extend(this_i_mus[i])
                        appended_mus.append(this_i_mus[i])
                        stopped = False
                    except ExtensionError:
                        logger.info('This extension failed. Still trying other extensions from the batch.')
                        # stopped = True
                        break
                    extensions += 1
        iterations += 1

        logger.info('')

        if max_extensions is not None and extensions >= max_extensions:
            logger.info(f'Maximum number of {max_extensions} extensions reached.')
            stopped = True
            break

    max_errs_pp = []
    if postprocessing:
        logger.info(f'Postprocessing...')
        if rtol is None: rtol = 1e-5
        N_start = surrogate.rom.solution_space.dim
        ref_sols = surrogate.fom.solution_space.empty()
        ref_norms = []
        for mu in appended_mus:
            res_rom = surrogate.rom.compute(solution=True, mu=mu)
            u_rom = surrogate.reductor.reconstruct(res_rom['solution'])
            ref_sols.append(u_rom)
            ref_norms.append(surrogate.error_norm(u_rom))
        N_pp = N_start - 1
        for N_pp in range(N_start-1, np.max(N_start-batchsize-1,0), -1):
            rom_pp = surrogate.reductor.reduce(N_pp)
            max_err = 0
            for i in range(len(appended_mus)):
                mu = appended_mus[i]
                res_pp = rom_pp.compute(solution=True, mu=mu)
                u_pp = surrogate.reductor.reconstruct(res_pp['solution'])
                err = surrogate.error_norm(u_pp - ref_sols[i])/ref_norms[i]
                if err>max_err: max_err = err
            max_err = max_err.item()
            max_errs_pp.append(max_err)
            if max_err>0.1*rtol: break

        surrogate.rom = surrogate.reductor.reduce(N_pp+1)
        logger.info(f'Size of reduced basis cut from {N_start} to {N_pp+1}')



    tictoc = time.perf_counter() - tic
    logger.info(f'Greedy search took {tictoc} seconds')
    return {'max_errs_iter': max_errs_iter, 'max_err_mus_iter': max_err_mus_iter,
            'max_errs_ext': max_errs_ext, 'max_err_mus_ext': max_err_mus_ext,
            'extensions': extensions, 'iterations': iterations,
            'time': tictoc, 'max_errs_pp': max_errs_pp}


def rb_batch_greedy(fom, reductor, training_set, use_error_estimator=True, error_norm=None,
                    atol=None, rtol=None, max_extensions=None, extension_params=None, pool=None,
                    batchsize=None, greedy_start=None, postprocessing=False):
    """Weak Greedy basis generation using the RB approximation error as surrogate.

    This algorithm generates a reduced basis using the :func:`weak greedy <weak_greedy>`
    algorithm :cite:`BCDDPW11`, where the approximation error is estimated from computing
    solutions of the reduced order model for the current reduced basis and then estimating
    the model reduction error.

    Parameters
    ----------
    fom
        The |Model| to reduce.
    reductor
        Reductor for reducing the given |Model|. This has to be
        an object with a `reduce` method, such that `reductor.reduce()`
        yields the reduced model, and an `exted_basis` method,
        such that `reductor.extend_basis(U, copy_U=False, **extension_params)`
        extends the current reduced basis by the vectors contained in `U`.
        For an example see :class:`~pymor.reductors.coercive.CoerciveRBReductor`.
    training_set
        The training set of |Parameters| on which to perform the greedy search.
    use_error_estimator
        If `False`, exactly compute the model reduction error by also computing
        the solution of `fom` for all |parameter values| of the training set.
        This is mainly useful when no estimator for the model reduction error
        is available.
    error_norm
        If `use_error_estimator` is `False`, use this function to calculate the
        norm of the error. If `None`, the Euclidean norm is used.
    atol
        See :func:`weak_greedy`.
    rtol
        See :func:`weak_greedy`.
    max_extensions
        See :func:`weak_greedy`.
    extension_params
        `dict` of parameters passed to the `reductor.extend_basis` method.
        If `None`, `'gram_schmidt'` basis extension will be used as a default
        for stationary problems (`fom.solve` returns `VectorArrays` of length 1)
        and `'pod'` basis extension (adding a single POD mode) for instationary
        problems.
    pool
        See :func:`weak_greedy`.

    Returns
    -------
    Dict with the following fields:

        :rom:                    The reduced |Model| obtained for the
                                 computed basis.
        :max_errs:               Sequence of maximum errors during the greedy run.
        :max_err_mus:            The parameters corresponding to `max_errs`.
        :extensions:             Number of performed basis extensions.
        :time:                   Total runtime of the algorithm.
    """
    surrogate = RBSurrogate(fom, reductor, use_error_estimator, error_norm, extension_params, pool or dummy_pool)

    result = weak_batch_greedy(surrogate, training_set, atol=atol, rtol=rtol, max_extensions=max_extensions, pool=pool,
                               batchsize=batchsize, greedy_start=greedy_start, postprocessing=postprocessing)
    result['rom'] = surrogate.rom

    return result
