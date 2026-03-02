from .glyphData import feq as feq, fne as fne, pt as pt, stem as stem
from .hintstate import (
	glyphHintState as glyphHintState, hintSegment as hintSegment, instanceStemState as instanceStemState, links as links,
	stemValue as stemValue)
from .logging import logging_reconfig as logging_reconfig, set_log_parameters as set_log_parameters
from .overlap import removeOverlap as removeOverlap
from .report import GlyphReport as GlyphReport
from _typeshed import Incomplete
from abc import abstractmethod
from typing import NamedTuple
import abc

class GlyphPE(NamedTuple):
    glyph: Incomplete
    pe: Incomplete

class LocDict(NamedTuple):
    l: Incomplete
    u: Incomplete
    used: Incomplete

class dimensionHinter(metaclass=abc.ABCMeta):
    """
    Common hinting implementation inherited by vertical and horizontal
    variants
    """
    @staticmethod
    def diffSign(a, b): ...
    @staticmethod
    def sameSign(a, b): ...
    StemLimit: int
    MaxStemDist: int
    InitBigDist: Incomplete
    BigDistFactor: Incomplete
    MinDist: int
    GhostWidth: int
    GhostLength: int
    GhostVal: int
    GhostSpecial: int
    BendLength: int
    BendTangent: float
    Theta: float
    MinValue: float
    MaxValue: float
    PruneA: int
    PruneB: float
    PruneC: int
    PruneD: int
    PruneValue: float
    PruneFactor: float
    PruneFactorLt: Incomplete
    PruneDistance: int
    MuchPF: int
    VeryMuchPF: int
    CPfrac: float
    ConflictValMin: float
    ConflictValMult: float
    BandMargin: int
    MaxFlare: int
    MaxBendMerge: int
    MinHintElementLength: int
    AutoLinearCurveFix: bool
    MaxFlex: int
    FlexLengthRatioCutoff: float
    FlexCand: int
    SCurveTangent: float
    OppoFlatMax: int
    FlatMin: int
    ExtremaDist: int
    NearFuzz: int
    NoOverlapPenalty: Incomplete
    GapDistDenom: int
    CloseMerge: int
    MaxMerge: int
    MinHighSegVal: Incomplete
    SFactor: int
    SpcBonus: int
    SpecialCharBonus: int
    GhostFactor: Incomplete
    DeltaDiffMin: float
    DeltaDiffReport: int
    NoSegScore: float
    FlexStrict: bool
    HasFlex: bool
    options: Incomplete
    fddicts: Incomplete
    gllist: Incomplete
    glyph: Incomplete
    report: Incomplete
    name: Incomplete
    isMulti: bool
    def __init__(self, options) -> None: ...
    fddict: Incomplete
    Bonus: Incomplete
    Pruning: bool
    hs: Incomplete
    def setGlyph(self, fddicts, report, gllist, name, clearPrev: bool = True) -> None:
        """Initialize the state for processing a specific glyph"""
    def resetForHinting(self) -> None:
        """Reset state for rehinting same glyph"""
    class glIter:
        """A pathElement set iterator for the glyphData object list"""
        gll: Incomplete
        il: Incomplete
        def __init__(self, gllist, glidx=None) -> None: ...
        def __next__(self): ...
        def __iter__(self): ...
    def __iter__(self, glidx=None): ...
    @abstractmethod
    def startFlex(self): ...
    @abstractmethod
    def stopFlex(self): ...
    @abstractmethod
    def startHint(self): ...
    @abstractmethod
    def stopHint(self): ...
    @abstractmethod
    def startStemConvert(self): ...
    @abstractmethod
    def stopStemConvert(self): ...
    @abstractmethod
    def startMaskConvert(self): ...
    @abstractmethod
    def stopMaskConvert(self): ...
    @abstractmethod
    def isV(self): ...
    @abstractmethod
    def segmentLists(self): ...
    @abstractmethod
    def dominantStems(self): ...
    @abstractmethod
    def isCounterGlyph(self): ...
    @abstractmethod
    def inBand(self, loc, isBottom: bool = False): ...
    @abstractmethod
    def hasBands(self): ...
    @abstractmethod
    def aDesc(self): ...
    @abstractmethod
    def isSpecial(self, lower: bool = False): ...
    @abstractmethod
    def checkTfm(self): ...
    def linearFlexOK(self): ...
    def addFlex(self, force: bool = True, inited: bool = False) -> None:
        """Path-level interface to add flex hints to current glyph"""
    def tryFlex(self, gl, c):
        """pathElement-level interface to add flex hints to current glyph"""
    def markFlex(self, cl) -> None: ...
    keepHints: bool
    BigDist: Incomplete
    def calcHintValues(self, lnks, force: bool = True, tryCounter: bool = True) -> None:
        """
        Top-level method for calculating stem hints for a glyph in one
        dimension
        """
    def handleOverlap(self): ...
    def addSegment(self, fr, to, loc, pe1, pe2, typ, desc, mid: bool = False) -> None: ...
    def CPFrom(self, cp2, cp3):
        """Return point cp3 adjusted relative to cp2 by CPFrac"""
    def CPTo(self, cp0, cp1):
        """Return point cp1 adjusted relative to cp0 by CPFrac"""
    def adjustDist(self, v, q): ...
    def testTan(self, p):
        """Test angle of p (treated as vector) relative to BendTangent"""
    @staticmethod
    def interpolate(q, v0, q0, v1, q1): ...
    def flatQuo(self, p1, p2, doOppo: bool = False):
        """
        Returns a measure of the flatness of the line between p1 and p2

        1 means exactly flat wrt dimension a (or o if doOppo)
        0 means not interestingly flat in dimension a. (or o if doOppo)
        Intermediate values represent degrees of interesting flatness
        """
    def testBend(self, p0, p1, p2):
        """Test of the angle between p0-p1 and p1-p2"""
    def isCCW(self, p0, p1, p2):
        """
        Returns true if p0 -> p1 -> p2 is counter-clockwise in glyph space.
        """
    def relPosition(self, c, lower: bool = False):
        """
        Return value indicates whether c is in the upper (or lower)
        subpath of the glyph (assuming a strict ordering of subpaths
        in this dimension)
        """
    def doBendsNext(self, c) -> None:
        '''
        Adds a BEND segment (short segments marking somewhat flat
        areas) at the end of a spline. In some cases the segment is
        added in both "directions"
        '''
    def doBendsPrev(self, c) -> None:
        '''
        Adds a BEND segment (short segments marking somewhat flat
        areas) at the start of a spline. In some cases the segment is
        added in both "directions"
        '''
    def nodeIsFlat(self, c, doPrev: bool = False):
        """
        Returns true if the junction of this spline and the next
        (or previous) is sufficiently flat, measured by OppoFlatMax
        and FlatMin
        """
    def sameDir(self, c, doPrev: bool = False):
        """
        Returns True if the next (or previous) spline continues in roughly
        the same direction as c
        """
    def extremaSegment(self, pe, extp, extt, isMn):
        """
        Given a curved pathElement pe and a point on that spline extp at
        t == extt, calculates a segment intersecting extp where all portions
        of the segment are within ExtremaDist of pe
        """
    def pickSpot(self, p0, p1, dist, pp0, pp1, prv, nxt):
        """
        Picks a segment location based on candidates p0 and p1 and
        other locations and metrics picked from the spline and
        the adjacent splines. Locations within BlueValue bands are
        priviledged.
        """
    def cpDirection(self, p0, p1, p2):
        '''
        Utility function for detecting singly-inflected curves.
        See original C code or "Fast Detection o the Geometric Form of
        Two-Dimensional Cubic Bezier Curves" by Stephen Vincent
        '''
    def prepForSegs(self) -> None: ...
    def genSegs(self) -> None:
        """
        Calls genSegsForPathElement for each pe and cleans up the
        generated segment lists
        """
    def genSegsForPathElement(self, c) -> None:
        '''
        Calculates and adds segments for pathElement c. These segments
        indicate "flat" areas of the glyph in the relevant dimension
        weighted by segment length.
        '''
    def limitSegs(self) -> None: ...
    def showSegs(self) -> None:
        """
        Adds a debug log message for each generated segment.
        This information is redundant with the genSegs info except that
        it shows the result of processing with compactLists(),
        remExtraBends(), etc.
        """
    def genStemVals(self) -> None:
        """
        Pairs segments of opposite direction and adds them as potential
        stems weighted by evalPair(). Also adds ghost stems for segments
        within BlueValue bands
        """
    def evalPair(self, ls, us):
        '''
        Calculates the initial "value" and "special" weights of a potential
        stem.

        Stems in one BlueValue band are given a spc boost but stems in
        both are ignored (presuambly because the Blues and OtherBlues are
        sufficient for hinting).

        Otherwise the value is based on:
           o The distance between the segment locations
           o The segment lengths
           o the extent of segment overlap (in the opposite direction)
           o Segment "bonus" values
        '''
    def stemMiss(self, ls, us):
        """
        Adds an info message for each stem within two em-units of a dominant
        stem width
        """
    def addStemValue(self, lloc, uloc, val, spc, lseg, useg) -> None:
        """Adapts the stem parameters into a stemValue object and adds it"""
    def insertStemValue(self, sv, note: str = 'add') -> None:
        """
        Adds a stemValue object into the stemValues list in sort order,
        skipping redundant GHOST stems
        """
    def combineStemValues(self) -> None:
        """
        Adjusts the values of stems with the same locations to give them
        each the same combined value.
        """
    def pruneStemVals(self) -> None:
        """
        Prune (remove) candidate stems based on comparisons to other stems.
        """
    def closeSegs(self, s1, s2):
        """
        Returns true if the segments (and the path between them)
        are within CloseMerge of one another
        """
    def prune(self, sv, other_sv, desc) -> None:
        '''
        Sets the pruned property on sv and logs it and the "better" stemValue
        '''
    def highestStemVals(self) -> None:
        """
        Associates each segment in both lists with the highest related stemVal,
        pruning stemValues with no association
        """
    def findHighestValForSegs(self, segl, isU) -> None:
        """Associates each segment in segl with the highest related stemVal"""
    def findHighestVal(self, seg, isU, locFlag):
        """Finds the highest stemVal related to seg"""
    def considerValForSeg(self, sv, seg, isU):
        """Utility test for findHighestVal"""
    def findBestValues(self) -> None:
        """
        Looks among stemValues with the same locations and finds the one
        with the highest spc/val. Assigns that stemValue to the .best
        property of that set
        """
    def replaceVals(self, oldl, oldu, newl, newu, newbest) -> None:
        '''
        Finds each stemValue at oldl, oldu and gives it a new "best"
        stemValue reference and its val and spc.
        '''
    def mergeVals(self) -> None:
        """
        Finds stem pairs with sides close to one another (in different
        senses) and uses replaceVals() to substitute one for another
        """
    def limitVals(self) -> None:
        """
        Limit the number of stem values in a dimension
        """
    def checkVals(self) -> None:
        """Reports stems with widths close to a dominant stem width"""
    def findLineSeg(self, loc, isBottom: bool = False):
        """Returns LINE segments with the passed location"""
    def reportStems(self) -> None:
        '''Reports stem zones and char ("alignment") zones'''
    def mainVals(self) -> None:
        """Picks set of highest-valued non-overlapping stems"""
    def mainOK(self, spc, val, hasHints, prevBV):
        """Utility test for mainVals"""
    def tryCounterHinting(self):
        """
        Attempts to counter-hint the dimension with the first three
        (e.g. highest value) mainValue stems
        """
    def addBBox(self, doSubpaths: bool = False):
        """
        Adds the top and bottom (or left and right) sides of the glyph
        as a stemValue -- serves as a backup hint stem when few are found

        When called with doSubpaths == True adds stem hints for the
        top/bottom or right/left of each subpath
        """
    def markStraySegs(self) -> None:
        """
        highestStemVals() may not assign a hintval to a given segment.
        Once the list of stems has been arrived at we go through each
        looking for stems where the segment on one side is unassigned
        and assign it to that stem.
        """
    def convertToStemLists(self):
        """
        This method builds up the information needed to mostly get away from
        looking at stem values when distributing hintmasks.

        hs.stems: Tuple of arrays of the eventual hstems or vstems objects
                  that will be copied into the glyphData object, one array
                  per gllist.
        """
    def calcInstanceStems(self, glidx) -> None: ...
    def bestLocation(self, sidx, ul, iSSl, hs0): ...
    def unconflict(self, sc, curSet=None, pinSet=None): ...
    def convertToMasks(self) -> None:
        """
        This method builds up the information needed to mostly get away from
        looking at stem values when distributing hintmasks.

        hs.stemOverlaps: A map of which stems overlap with which other stems.
        hs.ghostCompat: [i][j] is true if stem i is a ghost and stem j can
                        substitute for it.
        """
    def makePEMask(self, pestate, c) -> None:
        """Convert the hints desired by pathElement to a conflict-free mask"""
    def OKToRem(self, loc, spc): ...

class hhinter(dimensionHinter):
    def startFlex(self) -> None:
        """Make pt.a map to x and pt.b map to y"""
    def stopFlex(self) -> None: ...
    topPairs: Incomplete
    bottomPairs: Incomplete
    def startHint(self) -> None:
        """
        Make pt.a map to x and pt.b map to y and store BlueValue bands
        for easier processing
        """
    startStemConvert = startFlex
    startMaskConvert = startFlex
    stopHint = stopFlex
    stopStemConvert = stopFlex
    stopMaskConvert = stopFlex
    def dominantStems(self): ...
    def isV(self):
        """Mark the hinter as horizontal rather than vertical"""
    def inBand(self, loc, isBottom: bool = False):
        """Return true if loc is within the selected set of bands"""
    def hasBands(self): ...
    def isSpecial(self, lower: bool = False): ...
    def aDesc(self): ...
    def checkTfm(self) -> None: ...
    def checkTfmVal(self, sl, pl) -> None: ...
    def checkInsideBands(self, loc, pl): ...
    def checkNearBands(self, loc, pl) -> None: ...
    def segmentLists(self): ...
    def isCounterGlyph(self): ...

class vhinter(dimensionHinter):
    def startFlex(self) -> None: ...
    def stopFlex(self) -> None: ...
    startHint = startFlex
    startStemConvert = startFlex
    startMaskConvert = startFlex
    stopHint = stopFlex
    stopStemConvert = stopFlex
    stopMaskConvert = stopFlex
    def isV(self): ...
    def dominantStems(self): ...
    def inBand(self, loc, isBottom: bool = False): ...
    def hasBands(self): ...
    def isSpecial(self, lower: bool = False):
        """Check the Specials list for the current glyph"""
    def aDesc(self): ...
    def checkTfm(self) -> None: ...
    def segmentLists(self): ...
    def isCounterGlyph(self): ...

class glyphHinter:
    """
    Adapter between high-level autohint.py code and the 1D hinter.
    Also contains code that uses hints from both dimensions, primarily
    for hintmask distribution
    """
    impl: Incomplete
    @classmethod
    def initialize(cls, options, dictRecord, logQueue=None) -> None: ...
    @classmethod
    def hint(cls, name, glyphTuple=None, fdKey=None): ...
    options: Incomplete
    dictRecord: Incomplete
    hHinter: Incomplete
    vHinter: Incomplete
    cnt: int
    taskDesc: str
    FlareValueLimit: int
    MaxHalfMargin: int
    PromotionDistance: int
    def __init__(self, options, dictRecord) -> None: ...
    def getSegments(self, glyph, pe, oppo: bool = False):
        """Returns the list of segments for pe in the requested dimension"""
    def getMasks(self, glyph, pe):
        """
        Returns the masks of hints needed by/desired for pe in each dimension
        """
    def compatiblePaths(self, gllist, fddicts): ...
    def distributeMasks(self, glyph):
        """
        When necessary, chose the locations and contents of hintmasks for
        the glyph
        """
    def buildCounterMasks(self, glyph) -> None:
        """
        For glyph dimensions that are counter-hinted, make a cntrmask
        with all Trues in that dimension (because only h/vstem3 style counter
        hints are supported)
        """
    def joinMasks(self, m, cm, log):
        """
        Try to add the stems in cm to m, or start a new mask if there are
        conflicts.
        """
    def bridgeMasks(self, glyph, o, n, used, pe) -> None:
        """
        For switching hintmasks: Clean up o by adding compatible stems from
        mainMask and add stems from o to n when they are close to pe

        used contains a running map of which stems have ever been included
        in a hintmask
        """
    def mergeMain(self, glyph): ...
    def cleanupUnused(self, gllist, usedmasks) -> None: ...
    def delUnused(self, l, ml) -> None:
        """If ml[d][i] is False delete that entry from ml[d]"""
    def listHintInfo(self, glyph) -> None:
        """
        Output debug messages about which stems are associated with which
        segments
        """
    def remFlares(self, glyph) -> None:
        """
        When two paths are witin MaxFlare and connected by a path that
        also stays within MaxFlare, and both desire different stems,
        (sometimes) remove the lower-valued stem of the pair
        """
    def isFlare(self, loc, glyph, c, n):
        """Utility function for remFlares"""
    def isUSeg(self, loc, uloc, lloc): ...
    def reportRemFlare(self, pe, pe2, desc) -> None: ...
    def otherInstanceStems(self, gllist): ...
    def otherInstanceMasks(self, gllist): ...
