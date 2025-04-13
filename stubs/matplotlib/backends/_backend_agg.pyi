import numpy
from typing import overload

class BufferRegion:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def get_extents(self) -> object:
        """get_extents(self: matplotlib.backends._backend_agg.BufferRegion) -> object"""
    def set_x(self, arg0: int) -> None:
        """set_x(self: matplotlib.backends._backend_agg.BufferRegion, arg0: int) -> None"""
    def set_y(self, arg0: int) -> None:
        """set_y(self: matplotlib.backends._backend_agg.BufferRegion, arg0: int) -> None"""
    def __buffer__(self, *args, **kwargs):
        """Return a buffer object that exposes the underlying memory of the object."""
    def __release_buffer__(self, *args, **kwargs):
        """Release the buffer object that exposes the underlying memory of the object."""

class RendererAgg:
    def __init__(self, width: int, height: int, dpi: float) -> None:
        """__init__(self: matplotlib.backends._backend_agg.RendererAgg, width: int, height: int, dpi: float) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def clear(self) -> None:
        """clear(self: matplotlib.backends._backend_agg.RendererAgg) -> None"""
    def copy_from_bbox(self, bbox: rect_d) -> BufferRegion:
        """copy_from_bbox(self: matplotlib.backends._backend_agg.RendererAgg, bbox: rect_d) -> BufferRegion"""
    def draw_gouraud_triangles(self, gc: GCAgg, points: numpy.ndarray[numpy.float64], colors: numpy.ndarray[numpy.float64], trans: trans_affine = ...) -> None:
        """draw_gouraud_triangles(self: matplotlib.backends._backend_agg.RendererAgg, gc: GCAgg, points: numpy.ndarray[numpy.float64], colors: numpy.ndarray[numpy.float64], trans: trans_affine = None) -> None"""
    def draw_image(self, gc: GCAgg, x: float, y: float, image: numpy.ndarray[numpy.uint8]) -> None:
        """draw_image(self: matplotlib.backends._backend_agg.RendererAgg, gc: GCAgg, x: float, y: float, image: numpy.ndarray[numpy.uint8]) -> None"""
    def draw_markers(self, gc: GCAgg, marker_path: PathIterator, marker_path_trans: trans_affine, path: PathIterator, trans: trans_affine, face: object = ...) -> None:
        """draw_markers(self: matplotlib.backends._backend_agg.RendererAgg, gc: GCAgg, marker_path: PathIterator, marker_path_trans: trans_affine, path: PathIterator, trans: trans_affine, face: object = None) -> None"""
    def draw_path(self, gc: GCAgg, path: PathIterator, trans: trans_affine, face: object = ...) -> None:
        """draw_path(self: matplotlib.backends._backend_agg.RendererAgg, gc: GCAgg, path: PathIterator, trans: trans_affine, face: object = None) -> None"""
    def draw_path_collection(self, gc: GCAgg, master_transform: trans_affine, paths: PathGenerator, transforms: numpy.ndarray[numpy.float64], offsets: numpy.ndarray[numpy.float64], offset_trans: trans_affine, facecolors: numpy.ndarray[numpy.float64], edgecolors: numpy.ndarray[numpy.float64], linewidths: numpy.ndarray[numpy.float64], dashes: list[Dashes], antialiaseds: numpy.ndarray[numpy.uint8], ignored: object, offset_position: object) -> None:
        """draw_path_collection(self: matplotlib.backends._backend_agg.RendererAgg, gc: GCAgg, master_transform: trans_affine, paths: PathGenerator, transforms: numpy.ndarray[numpy.float64], offsets: numpy.ndarray[numpy.float64], offset_trans: trans_affine, facecolors: numpy.ndarray[numpy.float64], edgecolors: numpy.ndarray[numpy.float64], linewidths: numpy.ndarray[numpy.float64], dashes: list[Dashes], antialiaseds: numpy.ndarray[numpy.uint8], ignored: object, offset_position: object) -> None"""
    def draw_quad_mesh(self, gc: GCAgg, master_transform: trans_affine, mesh_width: int, mesh_height: int, coordinates: numpy.ndarray[numpy.float64], offsets: numpy.ndarray[numpy.float64], offset_trans: trans_affine, facecolors: numpy.ndarray[numpy.float64], antialiased: bool, edgecolors: numpy.ndarray[numpy.float64]) -> None:
        """draw_quad_mesh(self: matplotlib.backends._backend_agg.RendererAgg, gc: GCAgg, master_transform: trans_affine, mesh_width: int, mesh_height: int, coordinates: numpy.ndarray[numpy.float64], offsets: numpy.ndarray[numpy.float64], offset_trans: trans_affine, facecolors: numpy.ndarray[numpy.float64], antialiased: bool, edgecolors: numpy.ndarray[numpy.float64]) -> None"""
    def draw_text_image(self, image: numpy.ndarray[numpy.uint8], x: float | int, y: float | int, angle: float, gc: GCAgg) -> None:
        """draw_text_image(self: matplotlib.backends._backend_agg.RendererAgg, image: numpy.ndarray[numpy.uint8], x: Union[float, int], y: Union[float, int], angle: float, gc: GCAgg) -> None"""
    @overload
    def restore_region(self, region: BufferRegion) -> None:
        """restore_region(*args, **kwargs)
        Overloaded function.

        1. restore_region(self: matplotlib.backends._backend_agg.RendererAgg, region: BufferRegion) -> None

        2. restore_region(self: matplotlib.backends._backend_agg.RendererAgg, region: BufferRegion, xx1: int, yy1: int, xx2: int, yy2: int, x: int, y: int) -> None
        """
    @overload
    def restore_region(self, region: BufferRegion, xx1: int, yy1: int, xx2: int, yy2: int, x: int, y: int) -> None:
        """restore_region(*args, **kwargs)
        Overloaded function.

        1. restore_region(self: matplotlib.backends._backend_agg.RendererAgg, region: BufferRegion) -> None

        2. restore_region(self: matplotlib.backends._backend_agg.RendererAgg, region: BufferRegion, xx1: int, yy1: int, xx2: int, yy2: int, x: int, y: int) -> None
        """
    def __buffer__(self, *args, **kwargs):
        """Return a buffer object that exposes the underlying memory of the object."""
    def __release_buffer__(self, *args, **kwargs):
        """Release the buffer object that exposes the underlying memory of the object."""
