from . import _api as _api, _docstring as _docstring, artist as artist, cbook as cbook
from .artist import Artist as Artist
from .patches import FancyArrowPatch as FancyArrowPatch, FancyBboxPatch as FancyBboxPatch, Rectangle as Rectangle
from .textpath import TextPath as TextPath, TextToPath as TextToPath
from .transforms import Affine2D as Affine2D, Bbox as Bbox, BboxBase as BboxBase, BboxTransformTo as BboxTransformTo, IdentityTransform as IdentityTransform, Transform as Transform
from _typeshed import Incomplete

_log: Incomplete

def _get_textbox(text, renderer):
    """
    Calculate the bounding box of the text.

    The bbox position takes text rotation into account, but the width and
    height are those of the unrotated box (unlike `.Text.get_window_extent`).
    """
def _get_text_metrics_with_cache(renderer, text, fontprop, ismath, dpi):
    """Call ``renderer.get_text_width_height_descent``, caching the results."""
def _get_text_metrics_with_cache_impl(renderer_ref, text, fontprop, ismath, dpi): ...

class Text(Artist):
    """Handle storing and drawing of text in window or data coordinates."""
    zorder: int
    _charsize_cache: Incomplete
    def __repr__(self) -> str: ...
    _text: str
    def __init__(self, x: int = 0, y: int = 0, text: str = '', *, color: Incomplete | None = None, verticalalignment: str = 'baseline', horizontalalignment: str = 'left', multialignment: Incomplete | None = None, fontproperties: Incomplete | None = None, rotation: Incomplete | None = None, linespacing: Incomplete | None = None, rotation_mode: Incomplete | None = None, usetex: Incomplete | None = None, wrap: bool = False, transform_rotates_text: bool = False, parse_math: Incomplete | None = None, antialiased: Incomplete | None = None, **kwargs) -> None:
        """
        Create a `.Text` instance at *x*, *y* with string *text*.

        The text is aligned relative to the anchor point (*x*, *y*) according
        to ``horizontalalignment`` (default: 'left') and ``verticalalignment``
        (default: 'baseline'). See also
        :doc:`/gallery/text_labels_and_annotations/text_alignment`.

        While Text accepts the 'label' keyword argument, by default it is not
        added to the handles of a legend.

        Valid keyword arguments are:

        %(Text:kwdoc)s
        """
    _multialignment: Incomplete
    _transform_rotates_text: Incomplete
    _bbox_patch: Incomplete
    _renderer: Incomplete
    def _reset_visual_defaults(self, text: str = '', color: Incomplete | None = None, fontproperties: Incomplete | None = None, usetex: Incomplete | None = None, parse_math: Incomplete | None = None, wrap: bool = False, verticalalignment: str = 'baseline', horizontalalignment: str = 'left', multialignment: Incomplete | None = None, rotation: Incomplete | None = None, transform_rotates_text: bool = False, linespacing: Incomplete | None = None, rotation_mode: Incomplete | None = None, antialiased: Incomplete | None = None) -> None: ...
    def update(self, kwargs): ...
    def __getstate__(self): ...
    def contains(self, mouseevent):
        """
        Return whether the mouse event occurred inside the axis-aligned
        bounding-box of the text.
        """
    def _get_xy_display(self):
        """
        Get the (possibly unit converted) transformed x, y in display coords.
        """
    def _get_multialignment(self): ...
    def _char_index_at(self, x):
        """
        Calculate the index closest to the coordinate x in display space.

        The position of text[index] is assumed to be the sum of the widths
        of all preceding characters text[:index].

        This works only on single line texts.
        """
    def get_rotation(self):
        """Return the text angle in degrees between 0 and 360."""
    def get_transform_rotates_text(self):
        """
        Return whether rotations of the transform affect the text direction.
        """
    _rotation_mode: Incomplete
    stale: bool
    def set_rotation_mode(self, m) -> None:
        '''
        Set text rotation mode.

        Parameters
        ----------
        m : {None, \'default\', \'anchor\'}
            If ``"default"``, the text will be first rotated, then aligned according
            to their horizontal and vertical alignments.  If ``"anchor"``, then
            alignment occurs before rotation. Passing ``None`` will set the rotation
            mode to ``"default"``.
        '''
    def get_rotation_mode(self):
        """Return the text rotation mode."""
    _antialiased: Incomplete
    def set_antialiased(self, antialiased) -> None:
        """
        Set whether to use antialiased rendering.

        Parameters
        ----------
        antialiased : bool

        Notes
        -----
        Antialiasing will be determined by :rc:`text.antialiased`
        and the parameter *antialiased* will have no effect if the text contains
        math expressions.
        """
    def get_antialiased(self):
        """Return whether antialiased rendering is used."""
    _color: Incomplete
    _verticalalignment: Incomplete
    _horizontalalignment: Incomplete
    _fontproperties: Incomplete
    _usetex: Incomplete
    _rotation: Incomplete
    _picker: Incomplete
    _linespacing: Incomplete
    def update_from(self, other) -> None: ...
    def _get_layout(self, renderer):
        """
        Return the extent (bbox) of the text together with
        multiple-alignment information. Note that it returns an extent
        of a rotated text when necessary.
        """
    def set_bbox(self, rectprops) -> None:
        """
        Draw a bounding box around self.

        Parameters
        ----------
        rectprops : dict with properties for `.patches.FancyBboxPatch`
             The default boxstyle is 'square'. The mutation
             scale of the `.patches.FancyBboxPatch` is set to the fontsize.

        Examples
        --------
        ::

            t.set_bbox(dict(facecolor='red', alpha=0.5))
        """
    def get_bbox_patch(self):
        """
        Return the bbox Patch, or None if the `.patches.FancyBboxPatch`
        is not made.
        """
    def update_bbox_position_size(self, renderer) -> None:
        """
        Update the location and the size of the bbox.

        This method should be used when the position and size of the bbox needs
        to be updated before actually drawing the bbox.
        """
    def _update_clip_properties(self) -> None: ...
    def set_clip_box(self, clipbox) -> None: ...
    def set_clip_path(self, path, transform: Incomplete | None = None) -> None: ...
    def set_clip_on(self, b) -> None: ...
    def get_wrap(self):
        """Return whether the text can be wrapped."""
    _wrap: Incomplete
    def set_wrap(self, wrap) -> None:
        """
        Set whether the text can be wrapped.

        Wrapping makes sure the text is confined to the (sub)figure box. It
        does not take into account any other artists.

        Parameters
        ----------
        wrap : bool

        Notes
        -----
        Wrapping does not work together with
        ``savefig(..., bbox_inches='tight')`` (which is also used internally
        by ``%matplotlib inline`` in IPython/Jupyter). The 'tight' setting
        rescales the canvas to accommodate all content and happens before
        wrapping.
        """
    def _get_wrap_line_width(self):
        """
        Return the maximum line width for wrapping text based on the current
        orientation.
        """
    def _get_dist_to_box(self, rotation, x0, y0, figure_box):
        """
        Return the distance from the given points to the boundaries of a
        rotated box, in pixels.
        """
    def _get_rendered_text_width(self, text):
        """
        Return the width of a given text string, in pixels.
        """
    def _get_wrapped_text(self):
        """
        Return a copy of the text string with new lines added so that the text
        is wrapped relative to the parent figure (if `get_wrap` is True).
        """
    def draw(self, renderer) -> None: ...
    def get_color(self):
        """Return the color of the text."""
    def get_fontproperties(self):
        """Return the `.font_manager.FontProperties`."""
    def get_fontfamily(self):
        """
        Return the list of font families used for font lookup.

        See Also
        --------
        .font_manager.FontProperties.get_family
        """
    def get_fontname(self):
        """
        Return the font name as a string.

        See Also
        --------
        .font_manager.FontProperties.get_name
        """
    def get_fontstyle(self):
        """
        Return the font style as a string.

        See Also
        --------
        .font_manager.FontProperties.get_style
        """
    def get_fontsize(self):
        """
        Return the font size as an integer.

        See Also
        --------
        .font_manager.FontProperties.get_size_in_points
        """
    def get_fontvariant(self):
        """
        Return the font variant as a string.

        See Also
        --------
        .font_manager.FontProperties.get_variant
        """
    def get_fontweight(self):
        """
        Return the font weight as a string or a number.

        See Also
        --------
        .font_manager.FontProperties.get_weight
        """
    def get_stretch(self):
        """
        Return the font stretch as a string or a number.

        See Also
        --------
        .font_manager.FontProperties.get_stretch
        """
    def get_horizontalalignment(self):
        """
        Return the horizontal alignment as a string.  Will be one of
        'left', 'center' or 'right'.
        """
    def get_unitless_position(self):
        """Return the (x, y) unitless position of the text."""
    def get_position(self):
        """Return the (x, y) position of the text."""
    def get_text(self):
        """Return the text string."""
    def get_verticalalignment(self):
        """
        Return the vertical alignment as a string.  Will be one of
        'top', 'center', 'bottom', 'baseline' or 'center_baseline'.
        """
    def get_window_extent(self, renderer: Incomplete | None = None, dpi: Incomplete | None = None):
        """
        Return the `.Bbox` bounding the text, in display units.

        In addition to being used internally, this is useful for specifying
        clickable regions in a png file on a web page.

        Parameters
        ----------
        renderer : Renderer, optional
            A renderer is needed to compute the bounding box.  If the artist
            has already been drawn, the renderer is cached; thus, it is only
            necessary to pass this argument when calling `get_window_extent`
            before the first draw.  In practice, it is usually easier to
            trigger a draw first, e.g. by calling
            `~.Figure.draw_without_rendering` or ``plt.show()``.

        dpi : float, optional
            The dpi value for computing the bbox, defaults to
            ``self.get_figure(root=True).dpi`` (*not* the renderer dpi); should be set
            e.g. if to match regions with a figure saved with a custom dpi value.
        """
    def set_backgroundcolor(self, color) -> None:
        """
        Set the background color of the text by updating the bbox.

        Parameters
        ----------
        color : :mpltype:`color`

        See Also
        --------
        .set_bbox : To change the position of the bounding box
        """
    def set_color(self, color) -> None:
        """
        Set the foreground color of the text

        Parameters
        ----------
        color : :mpltype:`color`
        """
    def set_horizontalalignment(self, align) -> None:
        """
        Set the horizontal alignment relative to the anchor point.

        See also :doc:`/gallery/text_labels_and_annotations/text_alignment`.

        Parameters
        ----------
        align : {'left', 'center', 'right'}
        """
    def set_multialignment(self, align) -> None:
        """
        Set the text alignment for multiline texts.

        The layout of the bounding box of all the lines is determined by the
        horizontalalignment and verticalalignment properties. This property
        controls the alignment of the text lines within that box.

        Parameters
        ----------
        align : {'left', 'right', 'center'}
        """
    def set_linespacing(self, spacing) -> None:
        """
        Set the line spacing as a multiple of the font size.

        The default line spacing is 1.2.

        Parameters
        ----------
        spacing : float (multiple of font size)
        """
    def set_fontfamily(self, fontname) -> None:
        """
        Set the font family.  Can be either a single string, or a list of
        strings in decreasing priority.  Each string may be either a real font
        name or a generic font class name.  If the latter, the specific font
        names will be looked up in the corresponding rcParams.

        If a `Text` instance is constructed with ``fontfamily=None``, then the
        font is set to :rc:`font.family`, and the
        same is done when `set_fontfamily()` is called on an existing
        `Text` instance.

        Parameters
        ----------
        fontname : {FONTNAME, 'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'}

        See Also
        --------
        .font_manager.FontProperties.set_family
        """
    def set_fontvariant(self, variant) -> None:
        """
        Set the font variant.

        Parameters
        ----------
        variant : {'normal', 'small-caps'}

        See Also
        --------
        .font_manager.FontProperties.set_variant
        """
    def set_fontstyle(self, fontstyle) -> None:
        """
        Set the font style.

        Parameters
        ----------
        fontstyle : {'normal', 'italic', 'oblique'}

        See Also
        --------
        .font_manager.FontProperties.set_style
        """
    def set_fontsize(self, fontsize) -> None:
        """
        Set the font size.

        Parameters
        ----------
        fontsize : float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
            If a float, the fontsize in points. The string values denote sizes
            relative to the default font size.

        See Also
        --------
        .font_manager.FontProperties.set_size
        """
    def get_math_fontfamily(self):
        """
        Return the font family name for math text rendered by Matplotlib.

        The default value is :rc:`mathtext.fontset`.

        See Also
        --------
        set_math_fontfamily
        """
    def set_math_fontfamily(self, fontfamily) -> None:
        """
        Set the font family for math text rendered by Matplotlib.

        This does only affect Matplotlib's own math renderer. It has no effect
        when rendering with TeX (``usetex=True``).

        Parameters
        ----------
        fontfamily : str
            The name of the font family.

            Available font families are defined in the
            :ref:`default matplotlibrc file
            <customizing-with-matplotlibrc-files>`.

        See Also
        --------
        get_math_fontfamily
        """
    def set_fontweight(self, weight) -> None:
        """
        Set the font weight.

        Parameters
        ----------
        weight : {a numeric value in range 0-1000, 'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'}

        See Also
        --------
        .font_manager.FontProperties.set_weight
        """
    def set_fontstretch(self, stretch) -> None:
        """
        Set the font stretch (horizontal condensation or expansion).

        Parameters
        ----------
        stretch : {a numeric value in range 0-1000, 'ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed', 'normal', 'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'}

        See Also
        --------
        .font_manager.FontProperties.set_stretch
        """
    def set_position(self, xy) -> None:
        """
        Set the (*x*, *y*) position of the text.

        Parameters
        ----------
        xy : (float, float)
        """
    _x: Incomplete
    def set_x(self, x) -> None:
        """
        Set the *x* position of the text.

        Parameters
        ----------
        x : float
        """
    _y: Incomplete
    def set_y(self, y) -> None:
        """
        Set the *y* position of the text.

        Parameters
        ----------
        y : float
        """
    def set_rotation(self, s) -> None:
        """
        Set the rotation of the text.

        Parameters
        ----------
        s : float or {'vertical', 'horizontal'}
            The rotation angle in degrees in mathematically positive direction
            (counterclockwise). 'horizontal' equals 0, 'vertical' equals 90.
        """
    def set_transform_rotates_text(self, t) -> None:
        """
        Whether rotations of the transform affect the text direction.

        Parameters
        ----------
        t : bool
        """
    def set_verticalalignment(self, align) -> None:
        """
        Set the vertical alignment relative to the anchor point.

        See also :doc:`/gallery/text_labels_and_annotations/text_alignment`.

        Parameters
        ----------
        align : {'baseline', 'bottom', 'center', 'center_baseline', 'top'}
        """
    def set_text(self, s) -> None:
        """
        Set the text string *s*.

        It may contain newlines (``\\n``) or math in LaTeX syntax.

        Parameters
        ----------
        s : object
            Any object gets converted to its `str` representation, except for
            ``None`` which is converted to an empty string.
        """
    def _preprocess_math(self, s):
        '''
        Return the string *s* after mathtext preprocessing, and the kind of
        mathtext support needed.

        - If *self* is configured to use TeX, return *s* unchanged except that
          a single space gets escaped, and the flag "TeX".
        - Otherwise, if *s* is mathtext (has an even number of unescaped dollar
          signs) and ``parse_math`` is not set to False, return *s* and the
          flag True.
        - Otherwise, return *s* with dollar signs unescaped, and the flag
          False.
        '''
    def set_fontproperties(self, fp) -> None:
        """
        Set the font properties that control the text.

        Parameters
        ----------
        fp : `.font_manager.FontProperties` or `str` or `pathlib.Path`
            If a `str`, it is interpreted as a fontconfig pattern parsed by
            `.FontProperties`.  If a `pathlib.Path`, it is interpreted as the
            absolute path to a font file.
        """
    def set_usetex(self, usetex) -> None:
        """
        Parameters
        ----------
        usetex : bool or None
            Whether to render using TeX, ``None`` means to use
            :rc:`text.usetex`.
        """
    def get_usetex(self):
        """Return whether this `Text` object uses TeX for rendering."""
    _parse_math: Incomplete
    def set_parse_math(self, parse_math) -> None:
        """
        Override switch to disable any mathtext parsing for this `Text`.

        Parameters
        ----------
        parse_math : bool
            If False, this `Text` will never use mathtext.  If True, mathtext
            will be used if there is an even number of unescaped dollar signs.
        """
    def get_parse_math(self):
        """Return whether mathtext parsing is considered for this `Text`."""
    def set_fontname(self, fontname) -> None:
        """
        Alias for `set_fontfamily`.

        One-way alias only: the getter differs.

        Parameters
        ----------
        fontname : {FONTNAME, 'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'}

        See Also
        --------
        .font_manager.FontProperties.set_family

        """

class OffsetFrom:
    """Callable helper class for working with `Annotation`."""
    _artist: Incomplete
    _ref_coord: Incomplete
    def __init__(self, artist, ref_coord, unit: str = 'points') -> None:
        """
        Parameters
        ----------
        artist : `~matplotlib.artist.Artist` or `.BboxBase` or `.Transform`
            The object to compute the offset from.

        ref_coord : (float, float)
            If *artist* is an `.Artist` or `.BboxBase`, this values is
            the location to of the offset origin in fractions of the
            *artist* bounding box.

            If *artist* is a transform, the offset origin is the
            transform applied to this value.

        unit : {'points, 'pixels'}, default: 'points'
            The screen units to use (pixels or points) for the offset input.
        """
    _unit: Incomplete
    def set_unit(self, unit) -> None:
        """
        Set the unit for input to the transform used by ``__call__``.

        Parameters
        ----------
        unit : {'points', 'pixels'}
        """
    def get_unit(self):
        """Return the unit for input to the transform used by ``__call__``."""
    def __call__(self, renderer):
        """
        Return the offset transform.

        Parameters
        ----------
        renderer : `RendererBase`
            The renderer to use to compute the offset

        Returns
        -------
        `Transform`
            Maps (x, y) in pixel or point units to screen units
            relative to the given artist.
        """

class _AnnotationBase:
    xy: Incomplete
    xycoords: Incomplete
    _draggable: Incomplete
    def __init__(self, xy, xycoords: str = 'data', annotation_clip: Incomplete | None = None) -> None: ...
    def _get_xy(self, renderer, xy, coords): ...
    def _get_xy_transform(self, renderer, coords): ...
    _annotation_clip: Incomplete
    def set_annotation_clip(self, b) -> None:
        '''
        Set the annotation\'s clipping behavior.

        Parameters
        ----------
        b : bool or None
            - True: The annotation will be clipped when ``self.xy`` is
              outside the Axes.
            - False: The annotation will always be drawn.
            - None: The annotation will be clipped when ``self.xy`` is
              outside the Axes and ``self.xycoords == "data"``.
        '''
    def get_annotation_clip(self):
        """
        Return the annotation's clipping behavior.

        See `set_annotation_clip` for the meaning of return values.
        """
    def _get_position_xy(self, renderer):
        """Return the pixel position of the annotated point."""
    def _check_xy(self, renderer: Incomplete | None = None):
        """Check whether the annotation at *xy_pixel* should be drawn."""
    def draggable(self, state: Incomplete | None = None, use_blit: bool = False):
        """
        Set whether the annotation is draggable with the mouse.

        Parameters
        ----------
        state : bool or None
            - True or False: set the draggability.
            - None: toggle the draggability.
        use_blit : bool, default: False
            Use blitting for faster image composition. For details see
            :ref:`func-animation`.

        Returns
        -------
        DraggableAnnotation or None
            If the annotation is draggable, the corresponding
            `.DraggableAnnotation` helper is returned.
        """

class Annotation(Text, _AnnotationBase):
    """
    An `.Annotation` is a `.Text` that can refer to a specific position *xy*.
    Optionally an arrow pointing from the text to *xy* can be drawn.

    Attributes
    ----------
    xy
        The annotated position.
    xycoords
        The coordinate system for *xy*.
    arrow_patch
        A `.FancyArrowPatch` to point from *xytext* to *xy*.
    """
    def __str__(self) -> str: ...
    _textcoords: Incomplete
    arrowprops: Incomplete
    _arrow_relpos: Incomplete
    arrow_patch: Incomplete
    def __init__(self, text, xy, xytext: Incomplete | None = None, xycoords: str = 'data', textcoords: Incomplete | None = None, arrowprops: Incomplete | None = None, annotation_clip: Incomplete | None = None, **kwargs) -> None:
        '''
        Annotate the point *xy* with text *text*.

        In the simplest form, the text is placed at *xy*.

        Optionally, the text can be displayed in another position *xytext*.
        An arrow pointing from the text to the annotated point *xy* can then
        be added by defining *arrowprops*.

        Parameters
        ----------
        text : str
            The text of the annotation.

        xy : (float, float)
            The point *(x, y)* to annotate. The coordinate system is determined
            by *xycoords*.

        xytext : (float, float), default: *xy*
            The position *(x, y)* to place the text at. The coordinate system
            is determined by *textcoords*.

        xycoords : single or two-tuple of str or `.Artist` or `.Transform` or callable, default: \'data\'

            The coordinate system that *xy* is given in. The following types
            of values are supported:

            - One of the following strings:

              ==================== ============================================
              Value                Description
              ==================== ============================================
              \'figure points\'      Points from the lower left of the figure
              \'figure pixels\'      Pixels from the lower left of the figure
              \'figure fraction\'    Fraction of figure from lower left
              \'subfigure points\'   Points from the lower left of the subfigure
              \'subfigure pixels\'   Pixels from the lower left of the subfigure
              \'subfigure fraction\' Fraction of subfigure from lower left
              \'axes points\'        Points from lower left corner of the Axes
              \'axes pixels\'        Pixels from lower left corner of the Axes
              \'axes fraction\'      Fraction of Axes from lower left
              \'data\'               Use the coordinate system of the object
                                   being annotated (default)
              \'polar\'              *(theta, r)* if not native \'data\'
                                   coordinates
              ==================== ============================================

              Note that \'subfigure pixels\' and \'figure pixels\' are the same
              for the parent figure, so users who want code that is usable in
              a subfigure can use \'subfigure pixels\'.

            - An `.Artist`: *xy* is interpreted as a fraction of the artist\'s
              `~matplotlib.transforms.Bbox`. E.g. *(0, 0)* would be the lower
              left corner of the bounding box and *(0.5, 1)* would be the
              center top of the bounding box.

            - A `.Transform` to transform *xy* to screen coordinates.

            - A function with one of the following signatures::

                def transform(renderer) -> Bbox
                def transform(renderer) -> Transform

              where *renderer* is a `.RendererBase` subclass.

              The result of the function is interpreted like the `.Artist` and
              `.Transform` cases above.

            - A tuple *(xcoords, ycoords)* specifying separate coordinate
              systems for *x* and *y*. *xcoords* and *ycoords* must each be
              of one of the above described types.

            See :ref:`plotting-guide-annotation` for more details.

        textcoords : single or two-tuple of str or `.Artist` or `.Transform` or callable, default: value of *xycoords*
            The coordinate system that *xytext* is given in.

            All *xycoords* values are valid as well as the following strings:

            =================   =================================================
            Value               Description
            =================   =================================================
            \'offset points\'     Offset, in points, from the *xy* value
            \'offset pixels\'     Offset, in pixels, from the *xy* value
            \'offset fontsize\'   Offset, relative to fontsize, from the *xy* value
            =================   =================================================

        arrowprops : dict, optional
            The properties used to draw a `.FancyArrowPatch` arrow between the
            positions *xy* and *xytext*.  Defaults to None, i.e. no arrow is
            drawn.

            For historical reasons there are two different ways to specify
            arrows, "simple" and "fancy":

            **Simple arrow:**

            If *arrowprops* does not contain the key \'arrowstyle\' the
            allowed keys are:

            ==========  =================================================
            Key         Description
            ==========  =================================================
            width       The width of the arrow in points
            headwidth   The width of the base of the arrow head in points
            headlength  The length of the arrow head in points
            shrink      Fraction of total length to shrink from both ends
            ?           Any `.FancyArrowPatch` property
            ==========  =================================================

            The arrow is attached to the edge of the text box, the exact
            position (corners or centers) depending on where it\'s pointing to.

            **Fancy arrow:**

            This is used if \'arrowstyle\' is provided in the *arrowprops*.

            Valid keys are the following `.FancyArrowPatch` parameters:

            ===============  ===================================
            Key              Description
            ===============  ===================================
            arrowstyle       The arrow style
            connectionstyle  The connection style
            relpos           See below; default is (0.5, 0.5)
            patchA           Default is bounding box of the text
            patchB           Default is None
            shrinkA          In points. Default is 2 points
            shrinkB          In points. Default is 2 points
            mutation_scale   Default is text size (in points)
            mutation_aspect  Default is 1
            ?                Any `.FancyArrowPatch` property
            ===============  ===================================

            The exact starting point position of the arrow is defined by
            *relpos*. It\'s a tuple of relative coordinates of the text box,
            where (0, 0) is the lower left corner and (1, 1) is the upper
            right corner. Values <0 and >1 are supported and specify points
            outside the text box. By default (0.5, 0.5), so the starting point
            is centered in the text box.

        annotation_clip : bool or None, default: None
            Whether to clip (i.e. not draw) the annotation when the annotation
            point *xy* is outside the Axes area.

            - If *True*, the annotation will be clipped when *xy* is outside
              the Axes.
            - If *False*, the annotation will always be drawn.
            - If *None*, the annotation will be clipped when *xy* is outside
              the Axes and *xycoords* is \'data\'.

        **kwargs
            Additional kwargs are passed to `.Text`.

        Returns
        -------
        `.Annotation`

        See Also
        --------
        :ref:`annotations`

        '''
    def contains(self, mouseevent): ...
    @property
    def xycoords(self): ...
    _xycoords: Incomplete
    @xycoords.setter
    def xycoords(self, xycoords): ...
    @property
    def xyann(self):
        """
        The text position.

        See also *xytext* in `.Annotation`.
        """
    @xyann.setter
    def xyann(self, xytext) -> None: ...
    def get_anncoords(self):
        """
        Return the coordinate system to use for `.Annotation.xyann`.

        See also *xycoords* in `.Annotation`.
        """
    def set_anncoords(self, coords) -> None:
        """
        Set the coordinate system to use for `.Annotation.xyann`.

        See also *xycoords* in `.Annotation`.
        """
    anncoords: Incomplete
    def set_figure(self, fig) -> None: ...
    def update_positions(self, renderer):
        """
        Update the pixel positions of the annotation text and the arrow patch.
        """
    _renderer: Incomplete
    def draw(self, renderer) -> None: ...
    def get_window_extent(self, renderer: Incomplete | None = None): ...
    def get_tightbbox(self, renderer: Incomplete | None = None): ...
