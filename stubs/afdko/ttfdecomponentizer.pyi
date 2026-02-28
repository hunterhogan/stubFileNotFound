from afdko.fdkutils import get_font_format as get_font_format

__version__: str

def get_options(args): ...
def process_font(input_path, output_path=None, verbose: bool = False):
    """
    De-componentize a single font at input_path, saving to output_path (or
    input_path if None)
    """
def main(args=None): ...
