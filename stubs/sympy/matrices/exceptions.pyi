class MatrixError(Exception): ...
class ShapeError(ValueError, MatrixError):
    """Wrong matrix shape"""
class NonSquareMatrixError(ShapeError): ...
class NonInvertibleMatrixError(ValueError, MatrixError):
    """The matrix in not invertible (division by multidimensional zero error)."""
class NonPositiveDefiniteMatrixError(ValueError, MatrixError):
    """The matrix is not a positive-definite matrix."""
