import abc
from _typeshed import Incomplete
from collections.abc import Generator
from matplotlib import _api as _api, cbook as cbook
from matplotlib._animation_data import DISPLAY_TEMPLATE as DISPLAY_TEMPLATE, INCLUDED_FRAMES as INCLUDED_FRAMES, JS_INCLUDE as JS_INCLUDE, STYLE_INCLUDE as STYLE_INCLUDE

_log: Incomplete
subprocess_creation_flags: Incomplete

def adjusted_figsize(w, h, dpi, n):
    """
    Compute figure size so that pixels are a multiple of n.

    Parameters
    ----------
    w, h : float
        Size in inches.

    dpi : float
        The dpi.

    n : int
        The target multiple.

    Returns
    -------
    wnew, hnew : float
        The new figure size in inches.
    """

class MovieWriterRegistry:
    """Registry of available writer classes by human readable name."""
    _registered: Incomplete
    def __init__(self) -> None: ...
    def register(self, name):
        """
        Decorator for registering a class under a name.

        Example use::

            @registry.register(name)
            class Foo:
                pass
        """
    def is_available(self, name):
        """
        Check if given writer is available by name.

        Parameters
        ----------
        name : str

        Returns
        -------
        bool
        """
    def __iter__(self):
        """Iterate over names of available writer class."""
    def list(self):
        """Get a list of available MovieWriters."""
    def __getitem__(self, name):
        """Get an available writer class from its name."""

writers: Incomplete

class AbstractMovieWriter(abc.ABC, metaclass=abc.ABCMeta):
    """
    Abstract base class for writing movies, providing a way to grab frames by
    calling `~AbstractMovieWriter.grab_frame`.

    `setup` is called to start the process and `finish` is called afterwards.
    `saving` is provided as a context manager to facilitate this process as ::

        with moviewriter.saving(fig, outfile='myfile.mp4', dpi=100):
            # Iterate over frames
            moviewriter.grab_frame(**savefig_kwargs)

    The use of the context manager ensures that `setup` and `finish` are
    performed as necessary.

    An instance of a concrete subclass of this class can be given as the
    ``writer`` argument of `Animation.save()`.
    """
    fps: Incomplete
    metadata: Incomplete
    codec: Incomplete
    bitrate: Incomplete
    def __init__(self, fps: int = 5, metadata: Incomplete | None = None, codec: Incomplete | None = None, bitrate: Incomplete | None = None) -> None: ...
    outfile: Incomplete
    fig: Incomplete
    dpi: Incomplete
    @abc.abstractmethod
    def setup(self, fig, outfile, dpi: Incomplete | None = None):
        """
        Setup for writing the movie file.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The figure object that contains the information for frames.
        outfile : str
            The filename of the resulting movie file.
        dpi : float, default: ``fig.dpi``
            The DPI (or resolution) for the file.  This controls the size
            in pixels of the resulting movie file.
        """
    @property
    def frame_size(self):
        """A tuple ``(width, height)`` in pixels of a movie frame."""
    def _supports_transparency(self):
        """
        Whether this writer supports transparency.

        Writers may consult output file type and codec to determine this at runtime.
        """
    @abc.abstractmethod
    def grab_frame(self, **savefig_kwargs):
        """
        Grab the image information from the figure and save as a movie frame.

        All keyword arguments in *savefig_kwargs* are passed on to the
        `~.Figure.savefig` call that saves the figure.  However, several
        keyword arguments that are supported by `~.Figure.savefig` may not be
        passed as they are controlled by the MovieWriter:

        - *dpi*, *bbox_inches*:  These may not be passed because each frame of the
           animation much be exactly the same size in pixels.
        - *format*: This is controlled by the MovieWriter.
        """
    @abc.abstractmethod
    def finish(self):
        """Finish any processing for writing the movie."""
    def saving(self, fig, outfile, dpi, *args, **kwargs) -> Generator[Incomplete]:
        """
        Context manager to facilitate writing the movie file.

        ``*args, **kw`` are any parameters that should be passed to `setup`.
        """

class MovieWriter(AbstractMovieWriter):
    """
    Base class for writing movies.

    This is a base class for MovieWriter subclasses that write a movie frame
    data to a pipe. You cannot instantiate this class directly.
    See examples for how to use its subclasses.

    Attributes
    ----------
    frame_format : str
        The format used in writing frame data, defaults to 'rgba'.
    fig : `~matplotlib.figure.Figure`
        The figure to capture data from.
        This must be provided by the subclasses.
    """
    supported_formats: Incomplete
    frame_format: Incomplete
    extra_args: Incomplete
    def __init__(self, fps: int = 5, codec: Incomplete | None = None, bitrate: Incomplete | None = None, extra_args: Incomplete | None = None, metadata: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        fps : int, default: 5
            Movie frame rate (per second).
        codec : str or None, default: :rc:`animation.codec`
            The codec to use.
        bitrate : int, default: :rc:`animation.bitrate`
            The bitrate of the movie, in kilobits per second.  Higher values
            means higher quality movies, but increase the file size.  A value
            of -1 lets the underlying movie encoder select the bitrate.
        extra_args : list of str or None, optional
            Extra command-line arguments passed to the underlying movie encoder. These
            arguments are passed last to the encoder, just before the filename. The
            default, None, means to use :rc:`animation.[name-of-encoder]_args` for the
            builtin writers.
        metadata : dict[str, str], default: {}
            A dictionary of keys and values for metadata to include in the
            output file. Some keys that may be of use include:
            title, artist, genre, subject, copyright, srcform, comment.
        """
    def _adjust_frame_size(self): ...
    def setup(self, fig, outfile, dpi: Incomplete | None = None) -> None: ...
    _proc: Incomplete
    def _run(self) -> None: ...
    def finish(self) -> None:
        """Finish any processing for writing the movie."""
    def grab_frame(self, **savefig_kwargs) -> None: ...
    def _args(self):
        """Assemble list of encoder-specific command-line arguments."""
    @classmethod
    def bin_path(cls):
        """
        Return the binary path to the commandline tool used by a specific
        subclass. This is a class method so that the tool can be looked for
        before making a particular MovieWriter subclass available.
        """
    @classmethod
    def isAvailable(cls):
        """Return whether a MovieWriter subclass is actually available."""

class FileMovieWriter(MovieWriter):
    """
    `MovieWriter` for writing to individual files and stitching at the end.

    This must be sub-classed to be useful.
    """
    def __init__(self, *args, **kwargs) -> None: ...
    fig: Incomplete
    outfile: Incomplete
    dpi: Incomplete
    _tmpdir: Incomplete
    temp_prefix: Incomplete
    _frame_counter: int
    _temp_paths: Incomplete
    fname_format_str: str
    def setup(self, fig, outfile, dpi: Incomplete | None = None, frame_prefix: Incomplete | None = None) -> None:
        """
        Setup for writing the movie file.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The figure to grab the rendered frames from.
        outfile : str
            The filename of the resulting movie file.
        dpi : float, default: ``fig.dpi``
            The dpi of the output file. This, with the figure size,
            controls the size in pixels of the resulting movie file.
        frame_prefix : str, optional
            The filename prefix to use for temporary files.  If *None* (the
            default), files are written to a temporary directory which is
            deleted by `finish`; if not *None*, no temporary files are
            deleted.
        """
    def __del__(self) -> None: ...
    @property
    def frame_format(self):
        """
        Format (png, jpeg, etc.) to use for saving the frames, which can be
        decided by the individual subclasses.
        """
    _frame_format: Incomplete
    @frame_format.setter
    def frame_format(self, frame_format) -> None: ...
    def _base_temp_name(self): ...
    def grab_frame(self, **savefig_kwargs) -> None: ...
    def finish(self) -> None: ...

class PillowWriter(AbstractMovieWriter):
    def _supports_transparency(self): ...
    @classmethod
    def isAvailable(cls): ...
    _frames: Incomplete
    def setup(self, fig, outfile, dpi: Incomplete | None = None) -> None: ...
    def grab_frame(self, **savefig_kwargs) -> None: ...
    def finish(self) -> None: ...

class FFMpegBase:
    """
    Mixin class for FFMpeg output.

    This is a base class for the concrete `FFMpegWriter` and `FFMpegFileWriter`
    classes.
    """
    _exec_key: str
    _args_key: str
    def _supports_transparency(self): ...
    codec: Incomplete
    @property
    def output_args(self): ...

class FFMpegWriter(FFMpegBase, MovieWriter):
    """
    Pipe-based ffmpeg writer.

    Frames are streamed directly to ffmpeg via a pipe and written in a single pass.

    This effectively works as a slideshow input to ffmpeg with the fps passed as
    ``-framerate``, so see also `their notes on frame rates`_ for further details.

    .. _their notes on frame rates: https://trac.ffmpeg.org/wiki/Slideshow#Framerates
    """
    def _args(self): ...

class FFMpegFileWriter(FFMpegBase, FileMovieWriter):
    """
    File-based ffmpeg writer.

    Frames are written to temporary files on disk and then stitched together at the end.

    This effectively works as a slideshow input to ffmpeg with the fps passed as
    ``-framerate``, so see also `their notes on frame rates`_ for further details.

    .. _their notes on frame rates: https://trac.ffmpeg.org/wiki/Slideshow#Framerates
    """
    supported_formats: Incomplete
    def _args(self): ...

class ImageMagickBase:
    """
    Mixin class for ImageMagick output.

    This is a base class for the concrete `ImageMagickWriter` and
    `ImageMagickFileWriter` classes, which define an ``input_names`` attribute
    (or property) specifying the input names passed to ImageMagick.
    """
    _exec_key: str
    _args_key: str
    def _supports_transparency(self): ...
    def _args(self): ...
    @classmethod
    def bin_path(cls): ...
    @classmethod
    def isAvailable(cls): ...

class ImageMagickWriter(ImageMagickBase, MovieWriter):
    """
    Pipe-based animated gif writer.

    Frames are streamed directly to ImageMagick via a pipe and written
    in a single pass.
    """
    input_names: str

class ImageMagickFileWriter(ImageMagickBase, FileMovieWriter):
    """
    File-based animated gif writer.

    Frames are written to temporary files on disk and then stitched
    together at the end.
    """
    supported_formats: Incomplete
    input_names: Incomplete

def _included_frames(frame_count, frame_format, frame_dir): ...
def _embedded_frames(frame_list, frame_format):
    """frame_list should be a list of base64-encoded png files"""

class HTMLWriter(FileMovieWriter):
    """Writer for JavaScript-based HTML movies."""
    supported_formats: Incomplete
    @classmethod
    def isAvailable(cls): ...
    embed_frames: Incomplete
    default_mode: Incomplete
    _bytes_limit: Incomplete
    def __init__(self, fps: int = 30, codec: Incomplete | None = None, bitrate: Incomplete | None = None, extra_args: Incomplete | None = None, metadata: Incomplete | None = None, embed_frames: bool = False, default_mode: str = 'loop', embed_limit: Incomplete | None = None) -> None: ...
    _saved_frames: Incomplete
    _total_bytes: int
    _hit_limit: bool
    _clear_temp: bool
    def setup(self, fig, outfile, dpi: Incomplete | None = None, frame_dir: Incomplete | None = None) -> None: ...
    def grab_frame(self, **savefig_kwargs): ...
    def finish(self) -> None: ...

class Animation:
    """
    A base class for Animations.

    This class is not usable as is, and should be subclassed to provide needed
    behavior.

    .. note::

        You must store the created Animation in a variable that lives as long
        as the animation should run. Otherwise, the Animation object will be
        garbage-collected and the animation stops.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        The figure object used to get needed events, such as draw or resize.

    event_source : object, optional
        A class that can run a callback when desired events
        are generated, as well as be stopped and started.

        Examples include timers (see `TimedAnimation`) and file
        system notifications.

    blit : bool, default: False
        Whether blitting is used to optimize drawing.  If the backend does not
        support blitting, then this parameter has no effect.

    See Also
    --------
    FuncAnimation,  ArtistAnimation
    """
    _draw_was_started: bool
    _fig: Incomplete
    _blit: Incomplete
    frame_seq: Incomplete
    event_source: Incomplete
    _first_draw_id: Incomplete
    _close_id: Incomplete
    def __init__(self, fig, event_source: Incomplete | None = None, blit: bool = False) -> None: ...
    def __del__(self) -> None: ...
    def _start(self, *args) -> None:
        """
        Starts interactive animation. Adds the draw frame command to the GUI
        handler, calls show to start the event loop.
        """
    def _stop(self, *args) -> None: ...
    def save(self, filename, writer: Incomplete | None = None, fps: Incomplete | None = None, dpi: Incomplete | None = None, codec: Incomplete | None = None, bitrate: Incomplete | None = None, extra_args: Incomplete | None = None, metadata: Incomplete | None = None, extra_anim: Incomplete | None = None, savefig_kwargs: Incomplete | None = None, *, progress_callback: Incomplete | None = None):
        """
        Save the animation as a movie file by drawing every frame.

        Parameters
        ----------
        filename : str
            The output filename, e.g., :file:`mymovie.mp4`.

        writer : `MovieWriter` or str, default: :rc:`animation.writer`
            A `MovieWriter` instance to use or a key that identifies a
            class to use, such as 'ffmpeg'.

        fps : int, optional
            Movie frame rate (per second).  If not set, the frame rate from the
            animation's frame interval.

        dpi : float, default: :rc:`savefig.dpi`
            Controls the dots per inch for the movie frames.  Together with
            the figure's size in inches, this controls the size of the movie.

        codec : str, default: :rc:`animation.codec`.
            The video codec to use.  Not all codecs are supported by a given
            `MovieWriter`.

        bitrate : int, default: :rc:`animation.bitrate`
            The bitrate of the movie, in kilobits per second.  Higher values
            means higher quality movies, but increase the file size.  A value
            of -1 lets the underlying movie encoder select the bitrate.

        extra_args : list of str or None, optional
            Extra command-line arguments passed to the underlying movie encoder. These
            arguments are passed last to the encoder, just before the output filename.
            The default, None, means to use :rc:`animation.[name-of-encoder]_args` for
            the builtin writers.

        metadata : dict[str, str], default: {}
            Dictionary of keys and values for metadata to include in
            the output file. Some keys that may be of use include:
            title, artist, genre, subject, copyright, srcform, comment.

        extra_anim : list, default: []
            Additional `Animation` objects that should be included
            in the saved movie file. These need to be from the same
            `.Figure` instance. Also, animation frames will
            just be simply combined, so there should be a 1:1 correspondence
            between the frames from the different animations.

        savefig_kwargs : dict, default: {}
            Keyword arguments passed to each `~.Figure.savefig` call used to
            save the individual frames.

        progress_callback : function, optional
            A callback function that will be called for every frame to notify
            the saving progress. It must have the signature ::

                def func(current_frame: int, total_frames: int) -> Any

            where *current_frame* is the current frame number and *total_frames* is the
            total number of frames to be saved. *total_frames* is set to None, if the
            total number of frames cannot be determined. Return values may exist but are
            ignored.

            Example code to write the progress to stdout::

                progress_callback = lambda i, n: print(f'Saving frame {i}/{n}')

        Notes
        -----
        *fps*, *codec*, *bitrate*, *extra_args* and *metadata* are used to
        construct a `.MovieWriter` instance and can only be passed if
        *writer* is a string.  If they are passed as non-*None* and *writer*
        is a `.MovieWriter`, a `RuntimeError` will be raised.
        """
    def _step(self, *args):
        """
        Handler for getting events. By default, gets the next frame in the
        sequence and hands the data off to be drawn.
        """
    def new_frame_seq(self):
        """Return a new sequence of frame information."""
    def new_saved_frame_seq(self):
        """Return a new sequence of saved/cached frame information."""
    def _draw_next_frame(self, framedata, blit) -> None: ...
    def _init_draw(self) -> None: ...
    def _pre_draw(self, framedata, blit) -> None: ...
    def _draw_frame(self, framedata) -> None: ...
    def _post_draw(self, framedata, blit) -> None: ...
    def _blit_draw(self, artists) -> None: ...
    def _blit_clear(self, artists) -> None: ...
    _blit_cache: Incomplete
    _drawn_artists: Incomplete
    _resize_id: Incomplete
    def _setup_blit(self) -> None: ...
    def _on_resize(self, event) -> None: ...
    def _end_redraw(self, event) -> None: ...
    _base64_video: Incomplete
    _video_size: Incomplete
    def to_html5_video(self, embed_limit: Incomplete | None = None):
        '''
        Convert the animation to an HTML5 ``<video>`` tag.

        This saves the animation as an h264 video, encoded in base64
        directly into the HTML5 video tag. This respects :rc:`animation.writer`
        and :rc:`animation.bitrate`. This also makes use of the
        *interval* to control the speed, and uses the *repeat*
        parameter to decide whether to loop.

        Parameters
        ----------
        embed_limit : float, optional
            Limit, in MB, of the returned animation. No animation is created
            if the limit is exceeded.
            Defaults to :rc:`animation.embed_limit` = 20.0.

        Returns
        -------
        str
            An HTML5 video tag with the animation embedded as base64 encoded
            h264 video.
            If the *embed_limit* is exceeded, this returns the string
            "Video too large to embed."
        '''
    _html_representation: Incomplete
    def to_jshtml(self, fps: Incomplete | None = None, embed_frames: bool = True, default_mode: Incomplete | None = None):
        """
        Generate HTML representation of the animation.

        Parameters
        ----------
        fps : int, optional
            Movie frame rate (per second). If not set, the frame rate from
            the animation's frame interval.
        embed_frames : bool, optional
        default_mode : str, optional
            What to do when the animation ends. Must be one of ``{'loop',
            'once', 'reflect'}``. Defaults to ``'loop'`` if the *repeat*
            parameter is True, otherwise ``'once'``.

        Returns
        -------
        str
            An HTML representation of the animation embedded as a js object as
            produced with the `.HTMLWriter`.
        """
    def _repr_html_(self):
        """IPython display hook for rendering."""
    def pause(self) -> None:
        """Pause the animation."""
    def resume(self) -> None:
        """Resume the animation."""

class TimedAnimation(Animation):
    """
    `Animation` subclass for time-based animation.

    A new frame is drawn every *interval* milliseconds.

    .. note::

        You must store the created Animation in a variable that lives as long
        as the animation should run. Otherwise, the Animation object will be
        garbage-collected and the animation stops.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        The figure object used to get needed events, such as draw or resize.
    interval : int, default: 200
        Delay between frames in milliseconds.
    repeat_delay : int, default: 0
        The delay in milliseconds between consecutive animation runs, if
        *repeat* is True.
    repeat : bool, default: True
        Whether the animation repeats when the sequence of frames is completed.
    blit : bool, default: False
        Whether blitting is used to optimize drawing.
    """
    _interval: Incomplete
    _repeat_delay: Incomplete
    _repeat: Incomplete
    def __init__(self, fig, interval: int = 200, repeat_delay: int = 0, repeat: bool = True, event_source: Incomplete | None = None, *args, **kwargs) -> None: ...
    frame_seq: Incomplete
    event_source: Incomplete
    def _step(self, *args):
        """Handler for getting events."""

class ArtistAnimation(TimedAnimation):
    """
    `TimedAnimation` subclass that creates an animation by using a fixed
    set of `.Artist` objects.

    Before creating an instance, all plotting should have taken place
    and the relevant artists saved.

    .. note::

        You must store the created Animation in a variable that lives as long
        as the animation should run. Otherwise, the Animation object will be
        garbage-collected and the animation stops.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        The figure object used to get needed events, such as draw or resize.
    artists : list
        Each list entry is a collection of `.Artist` objects that are made
        visible on the corresponding frame.  Other artists are made invisible.
    interval : int, default: 200
        Delay between frames in milliseconds.
    repeat_delay : int, default: 0
        The delay in milliseconds between consecutive animation runs, if
        *repeat* is True.
    repeat : bool, default: True
        Whether the animation repeats when the sequence of frames is completed.
    blit : bool, default: False
        Whether blitting is used to optimize drawing.
    """
    _drawn_artists: Incomplete
    _framedata: Incomplete
    def __init__(self, fig, artists, *args, **kwargs) -> None: ...
    def _init_draw(self) -> None: ...
    def _pre_draw(self, framedata, blit) -> None:
        """Clears artists from the last frame."""
    def _draw_frame(self, artists) -> None: ...

class FuncAnimation(TimedAnimation):
    """
    `TimedAnimation` subclass that makes an animation by repeatedly calling
    a function *func*.

    .. note::

        You must store the created Animation in a variable that lives as long
        as the animation should run. Otherwise, the Animation object will be
        garbage-collected and the animation stops.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        The figure object used to get needed events, such as draw or resize.

    func : callable
        The function to call at each frame.  The first argument will
        be the next value in *frames*.   Any additional positional
        arguments can be supplied using `functools.partial` or via the *fargs*
        parameter.

        The required signature is::

            def func(frame, *fargs) -> iterable_of_artists

        It is often more convenient to provide the arguments using
        `functools.partial`. In this way it is also possible to pass keyword
        arguments. To pass a function with both positional and keyword
        arguments, set all arguments as keyword arguments, just leaving the
        *frame* argument unset::

            def func(frame, art, *, y=None):
                ...

            ani = FuncAnimation(fig, partial(func, art=ln, y='foo'))

        If ``blit == True``, *func* must return an iterable of all artists
        that were modified or created. This information is used by the blitting
        algorithm to determine which parts of the figure have to be updated.
        The return value is unused if ``blit == False`` and may be omitted in
        that case.

    frames : iterable, int, generator function, or None, optional
        Source of data to pass *func* and each frame of the animation

        - If an iterable, then simply use the values provided.  If the
          iterable has a length, it will override the *save_count* kwarg.

        - If an integer, then equivalent to passing ``range(frames)``

        - If a generator function, then must have the signature::

             def gen_function() -> obj

        - If *None*, then equivalent to passing ``itertools.count``.

        In all of these cases, the values in *frames* is simply passed through
        to the user-supplied *func* and thus can be of any type.

    init_func : callable, optional
        A function used to draw a clear frame. If not given, the results of
        drawing from the first item in the frames sequence will be used. This
        function will be called once before the first frame.

        The required signature is::

            def init_func() -> iterable_of_artists

        If ``blit == True``, *init_func* must return an iterable of artists
        to be re-drawn. This information is used by the blitting algorithm to
        determine which parts of the figure have to be updated.  The return
        value is unused if ``blit == False`` and may be omitted in that case.

    fargs : tuple or None, optional
        Additional arguments to pass to each call to *func*. Note: the use of
        `functools.partial` is preferred over *fargs*. See *func* for details.

    save_count : int, optional
        Fallback for the number of values from *frames* to cache. This is
        only used if the number of frames cannot be inferred from *frames*,
        i.e. when it's an iterator without length or a generator.

    interval : int, default: 200
        Delay between frames in milliseconds.

    repeat_delay : int, default: 0
        The delay in milliseconds between consecutive animation runs, if
        *repeat* is True.

    repeat : bool, default: True
        Whether the animation repeats when the sequence of frames is completed.

    blit : bool, default: False
        Whether blitting is used to optimize drawing.  Note: when using
        blitting, any animated artists will be drawn according to their zorder;
        however, they will be drawn on top of any previous artists, regardless
        of their zorder.

    cache_frame_data : bool, default: True
        Whether frame data is cached.  Disabling cache might be helpful when
        frames contain large objects.
    """
    _args: Incomplete
    _func: Incomplete
    _init_func: Incomplete
    _save_count: Incomplete
    _iter_gen: Incomplete
    _tee_from: Incomplete
    _cache_frame_data: Incomplete
    _save_seq: Incomplete
    def __init__(self, fig, func, frames: Incomplete | None = None, init_func: Incomplete | None = None, fargs: Incomplete | None = None, save_count: Incomplete | None = None, *, cache_frame_data: bool = True, **kwargs) -> None: ...
    def new_frame_seq(self): ...
    _old_saved_seq: Incomplete
    def new_saved_frame_seq(self): ...
    _drawn_artists: Incomplete
    def _init_draw(self) -> None: ...
    def _draw_frame(self, framedata): ...

def _validate_grabframe_kwargs(savefig_kwargs) -> None: ...
