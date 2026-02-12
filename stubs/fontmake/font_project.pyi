from collections.abc import Generator
from typing import Literal, Any
from fontTools import designspaceLib
import enum

logger = ...
timer = ...
PUBLIC_PREFIX = ...
GLYPHS_PREFIX = ...
KEEP_GLYPHS_OLD_KEY = ...
REMOVE_GLYPHS_OLD_KEY = ...
KEEP_GLYPHS_NEW_KEY = ...
REMOVE_GLYPHS_NEW_KEY = ...
GLYPH_EXPORT_KEY = ...
COMPAT_CHECK_KEY = ...
INTERPOLATABLE_OUTPUTS = ...
AUTOHINTING_PARAMETERS = ...
INSTANCE_LOCATION_KEY = ...
INSTANCE_FILENAME_KEY = ...
UFO_STRUCTURE_EXTENSIONS = ...
class CurveConversion(enum.Enum):
    ALL_CUBIC_TO_QUAD = ...
    MIXED_CUBIC_TO_QUAD = ...
    KEEP_QUAD = ...
    KEEP_CUBIC = ...
    @classmethod
    def default(cls) -> Literal[CurveConversion.ALL_CUBIC_TO_QUAD]:
        ...

    @property
    def convertCubics(self) -> bool:
        ...

    @property
    def allQuadratic(self) -> bool:
        ...



def needs_subsetting(ufo) -> bool:
    ...

class FontProject:
    """Provides methods for building fonts."""

    def __init__(self, timing=..., verbose=..., validate_ufo=...) -> None:
        ...

    def open_ufo(self, path): # -> Font:
        ...

    def save_ufo_as(self, font, path, ufo_structure=..., indent_json=...) -> None:
        ...

    @timer()
    def build_master_ufos(self, glyphs_path, designspace_path=..., master_dir=..., instance_dir=..., family_name=..., mti_source=..., write_skipexportglyphs=..., generate_GDEF=..., ufo_structure=..., indent_json=..., glyph_data=..., save_ufos=...) -> Any:
        """Build UFOs and designspace from Glyphs source."""

    @timer()
    def add_mti_features_to_master_ufos(self, mti_source, masters) -> None:
        ...

    def build_otfs(self, ufos, **kwargs) -> None:
        """Build OpenType binaries with CFF outlines."""

    def build_ttfs(self, ufos, **kwargs) -> None:
        """Build OpenType binaries with TrueType outlines."""

    def build_interpolatable_ttfs(self, designspace, **kwargs):
        """Build OpenType binaries with interpolatable TrueType outlines from DesignSpaceDocument object."""

    def build_interpolatable_otfs(self, designspace, **kwargs):
        """Build OpenType binaries with interpolatable TrueType outlines from DesignSpaceDocument object."""

    def build_variable_fonts(self, designspace: designspaceLib.DesignSpaceDocument, variable_fonts: str = ..., output_path=..., output_dir=..., ttf=..., optimize_gvar=..., optimize_cff=..., use_production_names=..., reverse_direction=..., ttf_curves=..., conversion_error=..., feature_writers=..., cff_round_tolerance=..., debug_feature_file=..., fea_include_dir=..., flatten_components=..., filters=..., auto_use_my_metrics=..., drop_implied_oncurves=..., variable_features=..., **kwargs) -> dict[Any, Any] | None:
        """Build OpenType variable fonts from masters in a designspace."""

    @timer()
    def save_otfs(self, ufos, ttf=..., is_instance=..., autohint=..., subset=..., use_production_names=..., subroutinize=..., optimize_cff=..., cff_round_tolerance=..., remove_overlaps=..., overlaps_backend=..., reverse_direction=..., ttf_curves=..., conversion_error=..., feature_writers=..., interpolate_layout_from=..., interpolate_layout_dir=..., output_path=..., output_dir=..., debug_feature_file=..., inplace=..., cff_version=..., subroutinizer=..., flatten_components=..., filters=..., generate_GDEF=..., fea_include_dir=..., auto_use_my_metrics=..., drop_implied_oncurves=..., skip_export_glyphs=...):
        """Build OpenType binaries from UFOs.

        Args:
            ufos: Font objects to compile.
            ttf: If True, build fonts with TrueType outlines and .ttf extension.
            is_instance: If output fonts are instances, for generating paths.
            autohint (Union[bool, None, str]): Parameters to provide to ttfautohint.
                If set to None (default), the UFO lib is scanned for autohinting parameters.
                If nothing is found, the autohinting step is skipped. The lib key is
                "com.schriftgestaltung.customParameter.InstanceDescriptorAsGSInstance.TTFAutohint options".
                If set to False, then no autohinting takes place whether or not the
                source specifies 'TTFAutohint options'. If True, it runs ttfautohint
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
                'ufos' list contains a single font.
            output_dir: directory where to save output files. Mutually
                exclusive with 'output_path' argument.
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
        """

    def subset_otf_from_ufo(self, otf_path, ufo) -> None:
        """Subset a font using "Keep Glyphs"/"Remove Glyphs" custom parameters, and export flags as set by glyphsLib.

        "Export Glyphs" is currently not supported:
        https://github.com/googlei18n/glyphsLib/issues/295.
        """

    def run_from_glyphs(self, glyphs_path, designspace_path=..., master_dir=..., instance_dir=..., family_name=..., mti_source=..., write_skipexportglyphs=..., generate_GDEF=..., glyph_data=..., output=..., output_dir=..., interpolate=..., **kwargs) -> None:
        """Run toolchain from Glyphs source.

        Args:
            glyphs_path: Path to source file.
            designspace_path: Output path of generated designspace document.
                By default it's "<family_name>[-<base_style>].designspace".
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
        """

    def interpolate_instance_ufos(self, designspace, include=..., round_instances=..., expand_features_to_instances=..., fea_include_dir=..., ufo_structure=..., indent_json=..., save_ufos=..., output_path=..., output_dir=...): # -> Generator[Font, Any, None]:
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

        Returns
        -------
            generator of ufoLib2.Font objects corresponding to the UFO instances.

        Raises
        ------
            FontmakeError: instances could not be prepared for interpolation or
                interpolation failed.
            ValueError: an instance descriptor did not have a filename attribute set.
        """

    def run_from_designspace(self, designspace, output=..., interpolate=..., variable_fonts: str = ..., masters_as_instances=..., interpolate_binary_layout=..., round_instances=..., feature_writers=..., variable_features=..., filters=..., expand_features_to_instances=..., check_compatibility=..., auto_use_my_metrics=..., **kwargs) -> None:
        """Run toolchain from a DesignSpace document to produce either static instance fonts (ttf or otf), interpolatable or variable fonts.

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

        Raises
        ------
            TypeError: "variable" or "interpolatable" outputs are incompatible
                with arguments "interpolate", "masters_as_instances", and
                "interpolate_binary_layout".
        """

    def run_from_ufos(self, ufos, output=..., **kwargs) -> None:
        """Run toolchain from UFO sources.

        Args:
            ufos: List of UFO sources, as either paths or opened objects.
            output: List of output formats to generate.
            kwargs: Arguments passed along to save_otfs.
        """



