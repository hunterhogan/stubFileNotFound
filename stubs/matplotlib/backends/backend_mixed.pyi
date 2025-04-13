from .backend_agg import RendererAgg as RendererAgg
from _typeshed import Incomplete
from matplotlib import cbook as cbook
from matplotlib._tight_bbox import process_figure_for_rasterizing as process_figure_for_rasterizing

class MixedModeRenderer:
    """
    A helper class to implement a renderer that switches between
    vector and raster drawing.  An example may be a PDF writer, where
    most things are drawn with PDF vector commands, but some very
    complex objects, such as quad meshes, are rasterised and then
    output as images.
    """
    _raster_renderer_class: Incomplete
    _width: Incomplete
    _height: Incomplete
    dpi: Incomplete
    _vector_renderer: Incomplete
    _raster_renderer: Incomplete
    figure: Incomplete
    _figdpi: Incomplete
    _bbox_inches_restore: Incomplete
    _renderer: Incomplete
    def __init__(self, figure, width, height, dpi, vector_renderer, raster_renderer_class: Incomplete | None = None, bbox_inches_restore: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        figure : `~matplotlib.figure.Figure`
            The figure instance.
        width : float
            The width of the canvas in logical units
        height : float
            The height of the canvas in logical units
        dpi : float
            The dpi of the canvas
        vector_renderer : `~matplotlib.backend_bases.RendererBase`
            An instance of a subclass of
            `~matplotlib.backend_bases.RendererBase` that will be used for the
            vector drawing.
        raster_renderer_class : `~matplotlib.backend_bases.RendererBase`
            The renderer class to use for the raster drawing.  If not provided,
            this will use the Agg backend (which is currently the only viable
            option anyway.)

        """
    def __getattr__(self, attr): ...
    def start_rasterizing(self) -> None:
        '''
        Enter "raster" mode.  All subsequent drawing commands (until
        `stop_rasterizing` is called) will be drawn with the raster backend.
        '''
    def stop_rasterizing(self) -> None:
        '''
        Exit "raster" mode.  All of the drawing that was done since
        the last `start_rasterizing` call will be copied to the
        vector backend by calling draw_image.
        '''
