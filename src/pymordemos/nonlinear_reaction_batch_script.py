import time
import pickle
from typer import Option, run
from pymor.basic import *
from pymor.discretizers.builtin.cg import discretize_stationary_cg as discretizer
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.algorithms.batchgreedy import rb_batch_greedy
from pymor.parallel.default import new_parallel_pool, dummy_pool
from pymor.parallel.mpi import MPIPool
import numpy as np
from mpi4py import MPI


def main(
    batchsize: int = Option(0, help='Size of batch. If 0, the size of WorkerPool is used.')
    ):
    # mpi_comm = MPI.COMM_WORLD
    # mpi_rank = mpi_comm.Get_rank()

    set_log_levels({'pymor': 'INFO'})

    domain = RectDomain(([0,0], [1,1]))
    l = ExpressionFunction('100 * sin(2 * pi * x[0]) * sin(2 * pi * x[1])', dim_domain = 2)
    parameters = Parameters({'reaction': 2})
    diffusion = ConstantFunction(1,2)

    pool = new_parallel_pool(allow_mpi=True)
    if pool is not dummy_pool:
        print(f'Using pool of {len(pool)} workers for parallelization.')
    else:
        print(f'No functional pool. Only dummy_pool is used.')

    assert batchsize>=0, 'Batch size must be nonnegative.'
    if batchsize==0: batchsize = len(pool)

    diameter = 1/36  # comparable to original paper 
    ei_snapshots = 12  # same as paper (creates 12x12 grid)
    ei_size = 20  # maximum number of bases in EIM
    rb_size = 45  # maximum number of bases in RBM
    test_snapshots = 15 # same as paper (creates 15x15 grid)

    tic = time.perf_counter()

    nonlinear_reaction_coefficient = ConstantFunction(1,2)
    test_nonlinearreaction = ExpressionFunction('reaction[0] * (exp(reaction[1] * u[0]) - 1) / reaction[1]', dim_domain = 1, parameters = parameters, variable = 'u')
    test_nonlinearreaction_derivative = ExpressionFunction('reaction[0] * exp(reaction[1] * u[0])', dim_domain = 1, parameters = parameters, variable = 'u')
    problem = StationaryProblem(domain = domain, rhs = l, diffusion = diffusion, nonlinear_reaction_coefficient = nonlinear_reaction_coefficient,
                                nonlinear_reaction = test_nonlinearreaction, nonlinear_reaction_derivative = test_nonlinearreaction_derivative)
    grid, boundary_info = discretize_domain_default(problem.domain, diameter=diameter)
    print('Anzahl Element', grid.size(0))
    print('Anzahl DoFs', grid.size(2))
    fom, data = discretizer(problem, diameter = diameter)

    # cache_id = (f'pymordemos.nonlinear_reaction {ei_snapshots} {test_snapshots}')
    fom.enable_caching('memory')

    parameter_space = fom.parameters.space((0.01, 10))
    parameter_sample = parameter_space.sample_uniformly(ei_snapshots)
    nonlin_op = fom.operator.operators[2]
    #evaluations = pool.map(_eval_nonlin_op, parameter_sample, fom=fom)

    if pool is not dummy_pool:
        with RemoteObjectManager() as reobma:
            evaluations = reobma.manage(pool.push(nonlin_op.range.empty()))
            pool.map(_interpolate_operator_build_evaluations, parameter_sample,
                    fom=fom, operator=nonlin_op, evaluations=evaluations)

            # Test set
            test_sample = parameter_space.sample_uniformly(test_snapshots)
            test_norms = list(zip(*pool.map(_test_set_norm, test_sample, fom=fom)))
            u_max_norm = np.max(test_norms)
            u_max_norm = u_max_norm.item()

            dofs, basis, data = ei_greedy(evaluations, copy=False,
                                        error_norm=fom.l2_norm,
                                        max_interpolation_dofs=ei_size,
                                        pool=pool)
    else:
        evaluations = nonlin_op.range.empty()
        for mu in parameter_sample:
            U = fom.solve(mu)
            evaluations.append(nonlin_op.apply(U, mu=mu))

        # Test set
        test_sample = parameter_space.sample_uniformly(test_snapshots)
        test_norms = []
        for mu in test_sample:
            U = fom.solve(mu)
            test_norms.append(U.norm(fom.h1_0_semi_product))
        u_max_norm = np.max(test_norms)
        u_max_norm = u_max_norm.item()

        dofs, basis, data = ei_greedy(evaluations, copy=False,
                                    error_norm=fom.l2_norm,
                                    max_interpolation_dofs=ei_size,
                                    pool=pool)
    ei_op = EmpiricalInterpolatedOperator(nonlin_op, dofs, basis, triangular=True)  #False for DEIM
    new_ops = [ei_op if i == 2 else op for i, op in enumerate(fom.operator.operators)]
    fom_ei = fom.with_(operator=fom.operator.with_(operators=new_ops))

    print('RB generation ...')

    reductor = StationaryRBReductor(fom_ei)

    greedy_data = rb_batch_greedy(fom, reductor, parameter_sample,
                                use_error_estimator=False,
                                error_norm=fom.h1_0_semi_norm,
                                max_extensions=rb_size,
                                rtol=1e-5,
                                batchsize=batchsize,
                                postprocessing=True,
                                pool=pool)
    
    toc = time.perf_counter()
    offline_time = toc - tic

    rom = greedy_data['rom']
    
    results = reduction_error_analysis(rom,
                                       fom=fom,
                                       reductor=reductor,
                                       error_estimator=False,
                                       error_norms=(fom.h1_0_semi_norm, fom.l2_norm),
                                       test_mus=test_sample,
                                       basis_sizes=0,
                                       pool=pool
                                       )

    # Online time
    n_online = 500
    tic = time.perf_counter()
    for mu in parameter_space.sample_randomly(n_online):
        reductor.reconstruct(rom.solve(mu))
    toc = time.perf_counter()
    online_time = (toc - tic)/n_online
    
    results['num_extensions'] = greedy_data['extensions']
    results['num_iterations'] = greedy_data['iterations']
    results['max_errs_pp'] = greedy_data['max_errs_pp']

    results['val_time'] = online_time  # Specify what time is saved
    results.pop('time', None)  # Delete old key
    results['calc_time'] = offline_time # Also save offline time

    with open(f'bm_nonlin_reac_N{len(pool)}_M{ei_size}_BS{batchsize}.pkl', 'wb') as fp:
            pickle.dump(results, fp)

def _interpolate_operator_build_evaluations(mu, fom=None, operator=None, evaluations=None):
    U = fom.solve(mu)
    evaluations.append(operator.apply(U, mu=mu))

def _test_set_norm(mu, fom=None):
    U = fom.solve(mu)
    return U.norm(fom.h1_0_semi_product)

if __name__ == '__main__':
    run(main)