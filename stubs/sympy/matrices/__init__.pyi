from .dense import (
	casoratian as casoratian, diag as diag, eye as eye, GramSchmidt as GramSchmidt, hessian as hessian,
	jordan_cell as jordan_cell, list2numpy as list2numpy, matrix2numpy as matrix2numpy,
	matrix_multiply_elementwise as matrix_multiply_elementwise, MutableDenseMatrix as MutableDenseMatrix, ones as ones,
	randMatrix as randMatrix, rot_axis1 as rot_axis1, rot_axis2 as rot_axis2, rot_axis3 as rot_axis3,
	rot_ccw_axis1 as rot_ccw_axis1, rot_ccw_axis2 as rot_ccw_axis2, rot_ccw_axis3 as rot_ccw_axis3,
	rot_givens as rot_givens, symarray as symarray, wronskian as wronskian, zeros as zeros)
from .exceptions import NonSquareMatrixError as NonSquareMatrixError, ShapeError as ShapeError
from .expressions import (
	Adjoint as Adjoint, block_collapse as block_collapse, blockcut as blockcut, BlockDiagMatrix as BlockDiagMatrix,
	BlockMatrix as BlockMatrix, det as det, Determinant as Determinant, DiagMatrix as DiagMatrix,
	diagonalize_vector as diagonalize_vector, DiagonalMatrix as DiagonalMatrix, DiagonalOf as DiagonalOf,
	DotProduct as DotProduct, FunctionMatrix as FunctionMatrix, hadamard_product as hadamard_product,
	HadamardPower as HadamardPower, HadamardProduct as HadamardProduct, Identity as Identity, Inverse as Inverse,
	kronecker_product as kronecker_product, KroneckerProduct as KroneckerProduct, MatAdd as MatAdd, MatMul as MatMul,
	MatPow as MatPow, matrix_symbols as matrix_symbols, MatrixExpr as MatrixExpr, MatrixPermute as MatrixPermute,
	MatrixSet as MatrixSet, MatrixSlice as MatrixSlice, MatrixSymbol as MatrixSymbol, OneMatrix as OneMatrix, per as per,
	Permanent as Permanent, PermutationMatrix as PermutationMatrix, Trace as Trace, trace as trace, Transpose as Transpose,
	ZeroMatrix as ZeroMatrix)
from .immutable import ImmutableDenseMatrix as ImmutableDenseMatrix, ImmutableSparseMatrix as ImmutableSparseMatrix
from .kind import MatrixKind as MatrixKind
from .matrixbase import DeferredVector as DeferredVector, MatrixBase as MatrixBase
from .sparse import MutableSparseMatrix as MutableSparseMatrix
from .sparsetools import banded as banded
from .utilities import dotprodsimp as dotprodsimp

__all__ = ['Adjoint', 'BlockDiagMatrix', 'BlockMatrix', 'DeferredVector', 'Determinant', 'DiagMatrix', 'DiagonalMatrix', 'DiagonalOf', 'DotProduct', 'FunctionMatrix', 'GramSchmidt', 'HadamardPower', 'HadamardProduct', 'Identity', 'ImmutableDenseMatrix', 'ImmutableMatrix', 'ImmutableSparseMatrix', 'Inverse', 'KroneckerProduct', 'MatAdd', 'MatMul', 'MatPow', 'Matrix', 'MatrixBase', 'MatrixExpr', 'MatrixKind', 'MatrixPermute', 'MatrixSet', 'MatrixSlice', 'MatrixSymbol', 'MutableDenseMatrix', 'MutableMatrix', 'MutableSparseMatrix', 'NonSquareMatrixError', 'OneMatrix', 'Permanent', 'PermutationMatrix', 'ShapeError', 'SparseMatrix', 'Trace', 'Transpose', 'ZeroMatrix', 'banded', 'block_collapse', 'blockcut', 'casoratian', 'det', 'diag', 'diagonalize_vector', 'dotprodsimp', 'eye', 'hadamard_product', 'hessian', 'jordan_cell', 'kronecker_product', 'list2numpy', 'matrix2numpy', 'matrix_multiply_elementwise', 'matrix_symbols', 'ones', 'per', 'randMatrix', 'rot_axis1', 'rot_axis2', 'rot_axis3', 'rot_ccw_axis1', 'rot_ccw_axis2', 'rot_ccw_axis3', 'rot_givens', 'symarray', 'trace', 'wronskian', 'zeros']

MutableMatrix = MutableDenseMatrix
Matrix = MutableMatrix
ImmutableMatrix = ImmutableDenseMatrix
SparseMatrix = MutableSparseMatrix
