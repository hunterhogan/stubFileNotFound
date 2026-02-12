import enum
from _typeshed import Incomplete
from collections.abc import Generator
from fontTools import designspaceLib
from fontmake.compatibility import CompatibilityChecker as CompatibilityChecker
from fontmake.errors import FontmakeError as FontmakeError, TTFAError as TTFAError
from fontmake.ttfautohint import ttfautohint as ttfautohint

logger: Incomplete
timer: Incomplete
PUBLIC_PREFIX: str
GLYPHS_PREFIX: str
KEEP_GLYPHS_OLD_KEY: Incomplete
REMOVE_GLYPHS_OLD_KEY: Incomplete
KEEP_GLYPHS_NEW_KEY: Incomplete
REMOVE_GLYPHS_NEW_KEY: Incomplete
GLYPH_EXPORT_KEY: Incomplete
COMPAT_CHECK_KEY: Incomplete
INTERPOLATABLE_OUTPUTS: Incomplete
AUTOHINTING_PARAMETERS: Incomplete
INSTANCE_LOCATION_KEY: str
INSTANCE_FILENAME_KEY: str
UFO_STRUCTURE_EXTENSIONS: Incomplete

class CurveConversion(enum.Enum):
    ALL_CUBIC_TO_QUAD = 'cu2qu'
    MIXED_CUBIC_TO_QUAD = 'mixed'
    KEEP_QUAD = 'keep-quad'
    KEEP_CUBIC = 'keep-cubic'
    @classmethod
    def default(cls): ...
    @property
    def convertCubics(self): ...
    @property
    def allQuadratic(self): ...

def needs_subsetting(ufo): ...

class FontProject:
    """Provides methods for building fonts."""
    validate_ufo: Incomplete
    def __init__(self, timing=None, verbose=None, validate_ufo: bool = False) -> None: ...
    def open_ufo(self, path): ...
    @staticmethod
    def _fix_ufo_path(path, ufo_structure):
        """Normalizes UFO path and updates extension suffix to match its structure"""
    def save_ufo_as(self, font, path, ufo_structure: str = 'package', indent_json: bool = False) -> None: ...
    def build_master_ufos(self, glyphs_path, designspace_path=None, master_dir=None, instance_dir=None, family_name=None, mti_source=None, write_skipexportglyphs: bool = True, generate_GDEF: bool = True, ufo_structure: str = 'package', indent_json: bool = False, glyph_data=None, save_ufos: bool = True):
        """Build UFOs and designspace from Glyphs source."""
    def add_mti_features_to_master_ufos(self, mti_source, masters) -> None: ...
    def build_otfs(self, ufos, **kwargs) -> None:
        """Build OpenType binaries with CFF outlines."""
    def build_ttfs(self, ufos, **kwargs) -> None:
        """Build OpenType binaries with TrueType outlines."""
    def _build_interpolatable_masters(self, designspace, ttf, use_production_names=None, reverse_direction: bool = True, ttf_curves=..., conversion_error=None, feature_writers=None, cff_round_tolerance=None, debug_feature_file=None, fea_include_dir=None, flatten_components: bool = False, filters=None, auto_use_my_metrics: bool = False, **kwargs): ...
    def build_interpolatable_ttfs(self, designspace, **kwargs):
        """Build OpenType binaries with interpolatable TrueType outlines
        from DesignSpaceDocument object.
        """
    def build_interpolatable_otfs(self, designspace, **kwargs):
        """Build OpenType binaries with interpolatable TrueType outlines
        from DesignSpaceDocument object.
        """
    def build_variable_fonts(self, designspace: designspaceLib.DesignSpaceDocument, variable_fonts: str = '.*', output_path=None, output_dir=None, ttf: bool = True, optimize_gvar: bool = True, optimize_cff=..., use_production_names=None, reverse_direction: bool = True, ttf_curves=..., conversion_error=None, feature_writers=None, cff_round_tolerance=None, debug_feature_file=None, fea_include_dir=None, flatten_components: bool = False, filters=None, auto_use_my_metrics: bool = False, drop_implied_oncurves: bool = False, variable_features: bool = True, **kwargs):
        """Build OpenType variable fonts from masters in a designspace."""
    def _iter_compile(self, ufos, ttf: bool = False, debugFeatureFile=None, **kwargs) -> Generator[Incomplete]: ...
    def save_otfs(self, ufos, ttf: bool = False, is_instance: bool = False, autohint=None, subset=None, use_production_names=None, subroutinize=None, optimize_cff=..., cff_round_tolerance=None, remove_overlaps: bool = True, overlaps_backend=None, reverse_direction: bool = True, ttf_curves=..., conversion_error=None, feature_writers=None, interpolate_layout_from=None, interpolate_layout_dir=None, output_path=None, output_dir=None, debug_feature_file=None, inplace: bool = True, cff_version: int = 1, subroutinizer=None, flatten_components: bool = False, filters=None, generate_GDEF: bool = True, fea_include_dir=None, auto_use_my_metrics: bool = True, drop_implied_oncurves: bool = False, skip_export_glyphs=None) -> None:
        '''Build OpenType binaries from UFOs.

        Args:
            ufos: Font objects to compile.
            ttf: If True, build fonts with TrueType outlines and .ttf extension.
            is_instance: If output fonts are instances, for generating paths.
            autohint (Union[bool, None, str]): Parameters to provide to ttfautohint.
                If set to None (default), the UFO lib is scanned for autohinting parameters.
                If nothing is found, the autohinting step is skipped. The lib key is
                "com.schriftgestaltung.customParameter.InstanceDescriptorAsGSInstance.TTFAutohint options".
                If set to False, then no autohinting takes place whether or not the
                source specifies \'TTFAutohint options\'. If True, it runs ttfautohint
                with no additional options.
            subset: Whether to subset the output according to data in the UFOs.
                If not provided, also determined by flags in the UFOs.
            use_production_names: Whether to use production glyph names in the
                output. If not provided, determined by flags in the UFOs.
            subroutinize: If True, subroutinize CFF outlines in output.
            cff_round_tolerance (float): controls the rounding of point
                coordinates in CFF table. It is defined as the maximum absolute
                difference between the original float and the rounded integer
                value. By default, all floats are rounded to integer (tolerance
                0.5); a value of 0 completely disables rounding; values in
                between only round floats which are close to their integral
                part within the tolerated range. Ignored if ttf=True.
            remove_overlaps: If True, remove overlaps in glyph shapes.
            overlaps_backend: name of the library to remove overlaps. Can be
                either "booleanOperations" (default) or "pathops".
            reverse_direction: If True, reverse contour directions when
                compiling TrueType outlines.
            ttf_curves: Choose between "cu2qu" (default), "mixed", "keep-quad" or
                "keep-cubic". NOTE: cubics in TTF use glyf v1 which is still draft!
            conversion_error: Error to allow when converting cubic CFF contours
                to quadratic TrueType contours.
            feature_writers: list of ufo2ft-compatible feature writer classes
                or pre-initialized objects that are passed on to ufo2ft
                feature compiler to generate automatic feature code. The
                default value (None) means that ufo2ft will use its built-in
                default feature writers (for kern, mark, mkmk, etc.). An empty
                list ([]) will skip any automatic feature generation.
            interpolate_layout_from: A DesignSpaceDocument object to give varLib
                for interpolating layout tables to use in output.
            interpolate_layout_dir: Directory containing the compiled master
                fonts to use for interpolating binary layout tables.
            output_path: output font file path. Only works when the input
                \'ufos\' list contains a single font.
            output_dir: directory where to save output files. Mutually
                exclusive with \'output_path\' argument.
            flatten_components: If True, flatten nested components to a single
                level.
            filters: list of ufo2ft-compatible filter classes or
                pre-initialized objects that are passed on to ufo2ft
                pre-processor to modify the glyph set. The filters are either
                pre-filters or post-filters, called before or after the default
                filters. The default filters are format specific and some can
                be disabled with other arguments.
            auto_use_my_metrics: whether to automatically set USE_MY_METRICS glyf
                component flags (0x0200). Not needed unless the font has hinted metrics.
            drop_implied_oncurves: drop on-curve points that can be implied when exactly
                in the middle of two off-curve points (TrueType only; default: False).
        '''
    def _save_interpolatable_fonts(self, designspace, output_dir, ttf) -> None: ...
    def subset_otf_from_ufo(self, otf_path, ufo) -> None:
        '''Subset a font using "Keep Glyphs"/"Remove Glyphs" custom parameters,
        and export flags as set by glyphsLib.

        "Export Glyphs" is currently not supported:
        https://github.com/googlei18n/glyphsLib/issues/295.
        '''
    def run_from_glyphs(self, glyphs_path, designspace_path=None, master_dir=None, instance_dir=None, family_name=None, mti_source=None, write_skipexportglyphs: bool = True, generate_GDEF: bool = True, glyph_data=None, output=(), output_dir=None, interpolate: bool = False, **kwargs) -> None:
        '''Run toolchain from Glyphs source.

        Args:
            glyphs_path: Path to source file.
            designspace_path: Output path of generated designspace document.
                By default it\'s "<family_name>[-<base_style>].designspace".
            master_dir: Directory where to save UFO masters (default:
                "master_ufo").
            instance_dir: Directory where to save UFO instances (default:
                "instance_ufo").
            family_name: If provided, uses this family name in the output.
            mti_source: Path to property list file containing a dictionary
                mapping UFO masters to dictionaries mapping layout table
                tags to MTI source paths which should be compiled into
                those tables.
            glyph_data: A list of GlyphData XML file paths.
            kwargs: Arguments passed along to run_from_designspace.
        '''
    def _instance_ufo_path(self, instance, designspace_path, output_dir=None, ufo_structure: str = 'package'):
        """Return an instance path, optionally overriding output dir or extension"""
    def interpolate_instance_ufos(self, designspace, include=None, round_instances: bool = False, expand_features_to_instances: bool = False, fea_include_dir=None, ufo_structure: str = 'package', indent_json: bool = False, save_ufos: bool = True, output_path=None, output_dir=None) -> Generator[Incomplete]:
        """Interpolate master UFOs with Instantiator and return instance UFOs.

        Args:
            designspace: a DesignSpaceDocument object containing sources and
                instances.
            include (str): optional regular expression pattern to match the
                DS instance 'name' attribute and only interpolate the matching
                instances.
            round_instances (bool): round instances' coordinates to integer.
            expand_features_to_instances: parses the master feature file, expands all
                include()s and writes the resulting full feature file to all instance
                UFOs. Use this if you share feature files among masters in external
                files. Otherwise, the relative include paths can break as instances
                may end up elsewhere. Only done on interpolation.
        Returns:
            generator of ufoLib2.Font objects corresponding to the UFO instances.
        Raises:
            FontmakeError: instances could not be prepared for interpolation or
                interpolation failed.
            ValueError: an instance descriptor did not have a filename attribute set.
        """
    def run_from_designspace(self, designspace, output=(), interpolate: bool = False, variable_fonts: str = '.*', masters_as_instances: bool = False, interpolate_binary_layout: bool = False, round_instances: bool = False, feature_writers=None, variable_features: bool = True, filters=None, expand_features_to_instances: bool = False, check_compatibility=None, auto_use_my_metrics=None, **kwargs):
        '''Run toolchain from a DesignSpace document to produce either static
        instance fonts (ttf or otf), interpolatable or variable fonts.

        Args:
            designspace: Path to designspace or DesignSpaceDocument object.
            interpolate: If True output all instance fonts, otherwise just
                masters. If the value is a string, only build instance(s) that
                match given name. The string is compiled into a regular
                expression and matched against the "name" attribute of
                designspace instances using `re.fullmatch`.
            variable_fonts: if True output all variable fonts, otherwise if the
                value is a string, only build variable fonts that match the
                given filename. As above, the string is compiled into a regular
                expression and matched against the "filename" attribute of
                designspace variable fonts using `re.fullmatch`.
            masters_as_instances: If True, output master fonts as instances.
            interpolate_binary_layout: Interpolate layout tables from compiled
                master binaries.
            round_instances: apply integer rounding when interpolating static
                instance UFOs.
            kwargs: Arguments passed along to run_from_ufos.

        Raises:
            TypeError: "variable" or "interpolatable" outputs are incompatible
                with arguments "interpolate", "masters_as_instances", and
                "interpolate_binary_layout".
        '''
    def _run_from_designspace_static(self, designspace, outputs, interpolate: bool = False, masters_as_instances: bool = False, interpolate_binary_layout: bool = False, round_instances: bool = False, feature_writers=None, expand_features_to_instances: bool = False, fea_include_dir=None, ufo_structure: str = 'package', indent_json: bool = False, output_path=None, output_dir=None, **kwargs) -> None: ...
    def _run_from_designspace_interpolatable(self, designspace, outputs, variable_fonts: str = '.*', variable_features: bool = True, output_path=None, output_dir=None, **kwargs): ...
    def run_from_ufos(self, ufos, output=(), **kwargs) -> None:
        """Run toolchain from UFO sources.

        Args:
            ufos: List of UFO sources, as either paths or opened objects.
            output: List of output formats to generate.
            kwargs: Arguments passed along to save_otfs.
        """
    @staticmethod
    def _search_instances(designspace, pattern): ...
    def _font_name(self, ufo):
        """Generate a postscript-style font name."""
    def _output_dir(self, ext, is_instance: bool = False, interpolatable: bool = False, autohinted: bool = False, is_variable: bool = False):
        """Generate an output directory.

        Args:
            ext: extension string.
            is_instance: The output is instance font or not.
            interpolatable: The output is interpolatable or not.
            autohinted: The output is autohinted or not.
            is_variable: The output is variable font or not.
        Return:
            output directory string.
        """
    def _output_path(self, ufo_or_font_name, ext, is_instance: bool = False, interpolatable: bool = False, autohinted: bool = False, is_variable: bool = False, output_dir=None, suffix=None):
        """Generate output path for a font file with given extension."""
    def _designspace_full_source_locations(self, designspace):
        '''Map "full" sources\' paths to their locations in a designspace.

        \'Sparse layer\' sources only contributing glyph outlines but no
        info/kerning/features are ignored.
        '''
    def _closest_location(self, location_map, target):
        """Return path of font whose location is closest to target."""

def _varLib_finder(source, directory: str = '', ext: str = 'ttf'):
    """Finder function to be used with varLib.build to find master TTFs given
    the filename of the source UFO master as specified in the designspace.
    It replaces the UFO directory with the one specified in 'directory'
    argument, and replaces the file extension with 'ext'.
    """
def _normpath(fname): ...
def _ensure_parent_dir(path): ...
