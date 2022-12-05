# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
import scipy.linalg as spla

from pymor.operators.block import BlockDiagonalOperator, BlockOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace

pytestmark = pytest.mark.builtin


def test_apply():
    A11 = np.random.randn(2, 3)
    A12 = np.random.randn(2, 4)
    A21 = np.zeros((5, 3))
    A22 = np.random.randn(5, 4)
    A = np.vstack((np.hstack((A11, A12)),
                   np.hstack((A21, A22))))
    A11op = NumpyMatrixOperator(A11)
    A12op = NumpyMatrixOperator(A12)
    A22op = NumpyMatrixOperator(A22)
    Aop = BlockOperator(np.array([[A11op, A12op], [None, A22op]]))

    v1 = np.random.randn(3)
    v2 = np.random.randn(4)
    v = np.hstack((v1, v2))
    v1va = NumpyVectorSpace.from_numpy(v1)
    v2va = NumpyVectorSpace.from_numpy(v2)
    vva = BlockVectorSpace.make_array((v1va, v2va))

    wva = Aop.apply(vva)
    w = np.hstack((wva.blocks[0].to_numpy(), wva.blocks[1].to_numpy()))
    assert np.allclose(A.dot(v), w)


def test_apply_adjoint():
    A11 = np.random.randn(2, 3)
    A12 = np.random.randn(2, 4)
    A21 = np.zeros((5, 3))
    A22 = np.random.randn(5, 4)
    A = np.vstack((np.hstack((A11, A12)),
                   np.hstack((A21, A22))))
    A11op = NumpyMatrixOperator(A11)
    A12op = NumpyMatrixOperator(A12)
    A22op = NumpyMatrixOperator(A22)
    Aop = BlockOperator(np.array([[A11op, A12op], [None, A22op]]))

    v1 = np.random.randn(2)
    v2 = np.random.randn(5)
    v = np.hstack((v1, v2))
    v1va = NumpyVectorSpace.from_numpy(v1)
    v2va = NumpyVectorSpace.from_numpy(v2)
    vva = BlockVectorSpace.make_array((v1va, v2va))

    wva = Aop.apply_adjoint(vva)
    w = np.hstack((wva.blocks[0].to_numpy(), wva.blocks[1].to_numpy()))
    assert np.allclose(A.T.dot(v), w)


def test_block_diagonal():
    A = np.random.randn(2, 3)
    B = np.random.randn(4, 5)
    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = BlockDiagonalOperator((Aop, Bop))
    assert Cop.source.dim == 8
    assert Cop.range.dim == 6


def test_blk_diag_apply_inverse():
    A = np.random.randn(2, 2)
    B = np.random.randn(3, 3)
    C = spla.block_diag(A, B)
    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = BlockDiagonalOperator((Aop, Bop))

    v1 = np.random.randn(2)
    v2 = np.random.randn(3)
    v = np.hstack((v1, v2))
    v1va = NumpyVectorSpace.from_numpy(v1)
    v2va = NumpyVectorSpace.from_numpy(v2)
    vva = BlockVectorSpace.make_array((v1va, v2va))

    wva = Cop.apply_inverse(vva)
    w = np.hstack((wva.blocks[0].to_numpy(), wva.blocks[1].to_numpy()))
    assert np.allclose(spla.solve(C, v), w)


def test_blk_diag_apply_inverse_adjoint():
    A = np.random.randn(2, 2)
    B = np.random.randn(3, 3)
    C = spla.block_diag(A, B)
    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = BlockDiagonalOperator((Aop, Bop))

    v1 = np.random.randn(2)
    v2 = np.random.randn(3)
    v = np.hstack((v1, v2))
    v1va = NumpyVectorSpace.from_numpy(v1)
    v2va = NumpyVectorSpace.from_numpy(v2)
    vva = BlockVectorSpace.make_array((v1va, v2va))

    wva = Cop.apply_inverse_adjoint(vva)
    w = np.hstack((wva.blocks[0].to_numpy(), wva.blocks[1].to_numpy()))
    assert np.allclose(spla.solve(C.T, v), w)


def test_block_jacobian():
    from pymor.operators.constructions import QuadraticFunctional

    A = np.random.randn(2, 2)
    B = np.random.randn(3, 3)
    C = np.random.randn(4, 4)
    Aop = QuadraticFunctional(NumpyMatrixOperator(A))
    Bop = QuadraticFunctional(NumpyMatrixOperator(B))
    Cop = NumpyMatrixOperator(C)
    Dop = BlockDiagonalOperator((Aop, Bop, Cop))
    Dop_single_block = BlockDiagonalOperator(np.array([[Aop]]))
    assert not Dop.linear and not Dop_single_block.linear

    v1 = np.random.randn(2)
    v2 = np.random.randn(3)
    v3 = np.random.randn(4)
    v1va = NumpyVectorSpace.from_numpy(v1)
    v2va = NumpyVectorSpace.from_numpy(v2)
    v3va = NumpyVectorSpace.from_numpy(v3)
    vva = BlockVectorSpace.make_array((v1va, v2va, v3va))
    vva_single_block = BlockVectorSpace.make_array(v1va)

    jac = Dop.jacobian(vva, mu=None)
    jac_single_block = Dop_single_block.jacobian(vva_single_block, mu=None)
    assert jac.linear and jac_single_block.linear
    assert np.all(jac.blocks[0, 0].vector.to_numpy()[0] == np.dot(A.T, v1) + np.dot(A, v1))
    assert np.all(jac.blocks[1, 1].vector.to_numpy()[0] == np.dot(B.T, v2) + np.dot(B, v2))
    assert np.all(jac.blocks[2, 2].matrix == C)


def test_sparse():
    # TODO: organize this test better
    op = np.empty((4, 4), dtype=object)
    for I in range(3):
        op[I, I] = NumpyMatrixOperator(np.eye(20))
    op[3, 3] = NumpyMatrixOperator(np.eye(10))
    from pymor.parameters.functionals import ProjectionParameterFunctional, ExpressionParameterFunctional
    op[0, 2] = ProjectionParameterFunctional('mu', 1, 0) * NumpyMatrixOperator(4 * np.eye(20))
    block_op = BlockOperator(op, make_sparse=True)
    assert block_op.parametric
    mu = block_op.parameters.parse([1])
    assembled_op = block_op.assemble(mu=mu)
    assert assembled_op != block_op
    op2 = op.copy()
    op2[1, 2] = ExpressionParameterFunctional('0*mu[0]', {'mu': 1}) * NumpyMatrixOperator(4 * np.eye(20))
    block_op_2 = BlockOperator(op2, make_sparse=True)
    (0.5 * block_op + 0.5 * block_op_2).assemble(mu)
    block_op_2_dense = BlockOperator(op2, make_sparse=False)
    (0.5 * block_op + 0.5 * block_op_2_dense).assemble(mu)
    ones = block_op.range.ones()
    block_op.apply(ones, mu=mu)
    block_op.apply_adjoint(ones, mu=mu)
    block_op.d_mu('something')
    dense_block_op = block_op.to_dense()
    dense_block_op.assemble(mu=mu)
    a = dense_block_op.as_source_array(mu=mu).to_numpy()
    b = block_op.as_source_array(mu=mu).to_numpy()
    assert np.allclose(a, b)
    a = dense_block_op.as_range_array(mu=mu).to_numpy()
    b = block_op.as_range_array(mu=mu).to_numpy()
    assert np.allclose(a, b)
    assert block_op.apply2(ones, ones, mu=mu) == dense_block_op.apply2(ones, ones, mu=mu)
    block_diag_1 = BlockDiagonalOperator([op[0, 0], op[0, 2]])
    block_diag_2 = BlockDiagonalOperator([op2[0, 0], op2[1, 2]])
    (block_diag_1 + block_diag_2).assemble(mu)
