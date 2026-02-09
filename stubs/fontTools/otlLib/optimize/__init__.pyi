from fontTools.otlLib.optimize.gpos import compact as compact, COMPRESSION_LEVEL as COMPRESSION_LEVEL
from fontTools.ttLib import TTFont as TTFont

def main(args=None) -> None:
    """Optimize the layout tables of an existing font"""
