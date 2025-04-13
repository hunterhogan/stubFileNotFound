import numpy

def affine_transform(points: numpy.ndarray[numpy.float64], trans: trans_affine) -> object:
    """affine_transform(points: numpy.ndarray[numpy.float64], trans: trans_affine) -> object"""
def cleanup_path(path: PathIterator, trans: trans_affine, remove_nans: bool, clip_rect: rect_d, snap_mode: e_snap_mode, stroke_width: float, simplify: bool | None, return_curves: bool, sketch: SketchParams) -> tuple:
    """cleanup_path(path: PathIterator, trans: trans_affine, remove_nans: bool, clip_rect: rect_d, snap_mode: e_snap_mode, stroke_width: float, simplify: Optional[bool], return_curves: bool, sketch: SketchParams) -> tuple"""
def clip_path_to_rect(path: PathIterator, rect: rect_d, inside: bool) -> list:
    """clip_path_to_rect(path: PathIterator, rect: rect_d, inside: bool) -> list"""
def convert_path_to_polygons(path: PathIterator, trans: trans_affine, width: float = ..., height: float = ..., closed_only: bool = ...) -> list:
    """convert_path_to_polygons(path: PathIterator, trans: trans_affine, width: float = 0.0, height: float = 0.0, closed_only: bool = False) -> list"""
def convert_to_string(path: PathIterator, trans: trans_affine, clip_rect: rect_d, simplify: bool | None, sketch: SketchParams, precision: int, codes, postfix: bool) -> object:
    '''convert_to_string(path: PathIterator, trans: trans_affine, clip_rect: rect_d, simplify: Optional[bool], sketch: SketchParams, precision: int, codes: Annotated[list[str], FixedSize(5)], postfix: bool) -> object

    --

    Convert *path* to a bytestring.

    The first five parameters (up to *sketch*) are interpreted as in `.cleanup_path`. The
    following ones are detailed below.

    Parameters
    ----------
    path : Path
    trans : Transform or None
    clip_rect : sequence of 4 floats, or None
    simplify : bool
    sketch : tuple of 3 floats, or None
    precision : int
        The precision used to "%.*f"-format the values. Trailing zeros and decimal points
        are always removed. (precision=-1 is a special case used to implement
        ttconv-back-compatible conversion.)
    codes : sequence of 5 bytestrings
        The bytes representation of each opcode (MOVETO, LINETO, CURVE3, CURVE4, CLOSEPOLY),
        in that order. If the bytes for CURVE3 is empty, quad segments are automatically
        converted to cubic ones (this is used by backends such as pdf and ps, which do not
        support quads).
    postfix : bool
        Whether the opcode comes after the values (True) or before (False).

    '''
def count_bboxes_overlapping_bbox(bbox: rect_d, bboxes: numpy.ndarray[numpy.float64]) -> int:
    """count_bboxes_overlapping_bbox(bbox: rect_d, bboxes: numpy.ndarray[numpy.float64]) -> int"""
def get_path_collection_extents(master_transform: trans_affine, paths: PathGenerator, transforms: numpy.ndarray[numpy.float64], offsets: numpy.ndarray[numpy.float64], offset_transform: trans_affine) -> tuple:
    """get_path_collection_extents(master_transform: trans_affine, paths: PathGenerator, transforms: numpy.ndarray[numpy.float64], offsets: numpy.ndarray[numpy.float64], offset_transform: trans_affine) -> tuple"""
def is_sorted_and_has_non_nan(array: object) -> bool:
    """is_sorted_and_has_non_nan(array: object) -> bool

    --

    Return whether the 1D *array* is monotonically increasing, ignoring NaNs, and has at
    least one non-nan value.
    """
def path_in_path(path_a: PathIterator, trans_a: trans_affine, path_b: PathIterator, trans_b: trans_affine) -> bool:
    """path_in_path(path_a: PathIterator, trans_a: trans_affine, path_b: PathIterator, trans_b: trans_affine) -> bool"""
def path_intersects_path(path1: PathIterator, path2: PathIterator, filled: bool = ...) -> bool:
    """path_intersects_path(path1: PathIterator, path2: PathIterator, filled: bool = False) -> bool"""
def path_intersects_rectangle(path: PathIterator, rect_x1: float, rect_y1: float, rect_x2: float, rect_y2: float, filled: bool = ...) -> bool:
    """path_intersects_rectangle(path: PathIterator, rect_x1: float, rect_y1: float, rect_x2: float, rect_y2: float, filled: bool = False) -> bool"""
def point_in_path(x: float, y: float, radius: float, path: PathIterator, trans: trans_affine) -> bool:
    """point_in_path(x: float, y: float, radius: float, path: PathIterator, trans: trans_affine) -> bool"""
def point_in_path_collection(x: float, y: float, radius: float, master_transform: trans_affine, paths: PathGenerator, transforms: numpy.ndarray[numpy.float64], offsets: numpy.ndarray[numpy.float64], offset_trans: trans_affine, filled: bool) -> object:
    """point_in_path_collection(x: float, y: float, radius: float, master_transform: trans_affine, paths: PathGenerator, transforms: numpy.ndarray[numpy.float64], offsets: numpy.ndarray[numpy.float64], offset_trans: trans_affine, filled: bool) -> object"""
def points_in_path(points: numpy.ndarray[numpy.float64], radius: float, path: PathIterator, trans: trans_affine) -> numpy.ndarray[numpy.float64]:
    """points_in_path(points: numpy.ndarray[numpy.float64], radius: float, path: PathIterator, trans: trans_affine) -> numpy.ndarray[numpy.float64]"""
def update_path_extents(path: PathIterator, trans: trans_affine, rect: rect_d, minpos: numpy.ndarray[numpy.float64], ignore: bool) -> tuple:
    """update_path_extents(path: PathIterator, trans: trans_affine, rect: rect_d, minpos: numpy.ndarray[numpy.float64], ignore: bool) -> tuple"""
