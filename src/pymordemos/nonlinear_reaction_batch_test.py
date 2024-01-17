from pymor.basic import *
from pymor.discretizers.builtin.cg import discretize_stationary_cg as discretizer
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.algorithms.batchgreedy import rb_batch_greedy
from pymor.parallel.default import new_parallel_pool, dummy_pool
from pymor.parallel.mpi import MPIPool
import numpy as np
from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()


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

diameter = 1/36  # comparable to original paper 
ei_snapshots = 12  # same as paper (creates 12x12 grid)
ei_size = 20  # maximum number of bases in EIM
rb_size = 45  # maximum number of bases in RBM
test_snapshots = 15 # same as paper (creates 15x15 grid)
batchsize = len(pool)

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

# Training set
def _interpolate_operator_build_evaluations(mu, fom=None, operator=None, evaluations=None):
    U = fom.solve(mu)
    evaluations.append(operator.apply(U, mu=mu))

def _test_set_norm(mu, fom=fom):
    U = fom.solve(mu)
    return U.norm(fom.h1_0_semi_product)


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
                            batchsize=batchsize,
                            postprocessing=False,
                            pool=pool)

rom = greedy_data['rom']

# print('Testing ROM...')

# max_err = -1
# for mu in test_sample:
#     u_fom = fom.solve(mu)
#     u_rom = rom.solve(mu)
#     this_diff = u_fom - reductor.reconstruct(u_rom)
#     this_err = this_diff.norm(fom.h1_0_semi_product)[0]
#     if this_err > max_err: max_err = this_err

# rel_error = max_err.item()/u_max_norm
print(f'RB size N: {len(reductor.bases["RB"])}')
# print(f'max. rel. error: {rel_error:2.5e}')

# test_sample = parameter_space.sample_uniformly(test_snapshots)
# abs_errs = []
# rel_errs = []
# max_abs_err = -1
# max_rel_err = -1
# max_abs_diff = None
# max_rel_diff = None
# for mu in test_sample:
#     u_fom = fom.solve(mu)
#     u_rom = rom.solve(mu)
#     this_diff = u_fom - reductor.reconstruct(u_rom)
#     this_abs_err = this_diff.norm(fom.h1_0_semi_product)[0]
#     this_rel_err = this_diff.norm(fom.h1_0_semi_product)[0]/u_fom.norm(fom.h1_0_semi_product)[0]
#     abs_errs.append(this_abs_err)
#     rel_errs.append(this_rel_err)
#     if this_abs_err > max_abs_err:
#         max_abs_err = this_abs_err
#         max_abs_diff = this_diff
#     if this_rel_err > max_rel_err:
#         max_rel_err = this_rel_err
#         max_rel_diff = this_diff

# print(f'max. abs. err.: {max_abs_err:2.5e}')
# print(f'max. rel. err.: {max_rel_err:2.5e}')

# fom.visualize((max_abs_diff, max_rel_diff), legend = ('Maximum Absolute Test Error', 'Maximum Relative Test Error'), separate_colorbars=True)
# del pool